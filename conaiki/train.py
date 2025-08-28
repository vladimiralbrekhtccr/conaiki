# train_qwen_omni_trainer.py
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import math

import torch
import torchaudio
import numpy as np 
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import (
    TrainingArguments as HFTrainingArguments,
    Trainer,
    set_seed,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# User ARGs
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ScriptArgs:
    model_path: str = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/models/rezised_Qwen2.5-Omni-7B"
    output_dir: str = "./qwen_omni_finetuned_common_voice_for_qwen_train_less_than_3_sec"

    train_data_path: str = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/conaiki/data/common_voice_for_qwen_train/less_than_3_sec/processed/streaming_chunks_padded.jsonl"
    audio_root: str = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/conaiki/data/common_voice_for_qwen_train/less_than_3_sec/processed"
    # val_data_path: Optional[str] = "data/val_data.jsonl"
    max_seq_length: int = 512 # TODO: think about 512 or less tokens, we don't need to train on large max_seq_length
    label_masking: str = "assistant_only"

    # NEW: streaming-chunk setup
    chunk_sec: float = 0.5
    cumulative_audio: bool = True  # use 0..t cumulative audio per step
    # optional: weight gate loss
    gate_loss_weight: float = 1.0

    # Freezes
    freeze_vision_encoder: bool = True # always True
    freeze_llm_backbone: bool = True # always True
    freeze_audio_encoder: bool = True # always True
    freeze_conaiki_modules: bool = False # TODO: later check the time module


    # Unfreeze
    unfreeze_lm_head: str = "UNFREEZE" # either FREEZE or UNFREEZE
    unfreeze_embed_tokens: str = "UNFREEZE" # embed_tokens inside llm_backbone so it will be freezed if {ScriptArgs.freeze_llm_backbone==True}
    unfreeze_audio_projector: str = "UNFREEZE" 
    unfreeze_conaiki_gate_proj: str = "UNFREEZE"
    

    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "qwen-omni-training"



# ──────────────────────────────────────────────────────────────────────────────
# # Datasets torch
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# ## Text
# ──────────────────────────────────────────────────────────────────────────────
class TextConversationDataset(Dataset):
    def __init__(self, data_path: str, processor: Qwen2_5OmniProcessor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length
        self.rows: List[List[Dict]] = []
        with open(data_path, "r") as f:
            for line in f:
                j = json.loads(line.strip())
                self.rows.append(j["conversations"])
        logger.info(f"Loaded {len(self.rows)} rows from {data_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        conv = self.rows[idx]

        # нормализуем контент под processor.apply_chat_template
        formatted = []
        for msg in conv:
            content = msg["content"] if not isinstance(msg.get("content"), str) else [
                {"type": "text", "text": msg["content"]}
            ]
            formatted.append({"role": msg["role"], "content": content})

        inputs = self.processor.apply_chat_template(
            [formatted],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        # убираем батчевую ось
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # labels (игнор падов)
        pad_id = self.processor.tokenizer.pad_token_id or 151643
        labels = inputs["input_ids"].clone()
        labels[labels == pad_id] = -100
        inputs["labels"] = labels
        return inputs

# ──────────────────────────────────────────────────────────────────────────────
# ## Audio
# ──────────────────────────────────────────────────────────────────────────────

WAIT, TRANSLATE= 0, 1   # gate classes
GATE_TOKENS = ["<WAIT>", "<TRANSLATE>"]

class ConaikiAudioDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: Qwen2_5OmniProcessor,
        max_seq_length: int = 1024,
        label_masking: str = "assistant_only",
        chunk_sec: float = 0.5,
        cumulative_audio: bool = True,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.label_masking = label_masking
        self.chunk_sec = float(chunk_sec)
        self.cumulative_audio = bool(cumulative_audio)

        self.rows = []
        with open(data_path, "r") as f:
            for line in f:
                self.rows.append(json.loads(line.strip()))

        self._sr = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", 16000)

        # Build (row_idx, chunk_idx) index without fully decoding audio
        self._index = []
        for i, row in enumerate(self.rows):
            audio_path = self._get_audio_path(row)
            try:
                info = torchaudio.info(audio_path)
                num_frames = info.num_frames
                sr = info.sample_rate
            except Exception:
                # Fallback: load headers by decoding (rare)
                wav, sr = torchaudio.load(audio_path)
                num_frames = wav.shape[-1]

            dur = num_frames / sr
            n_chunks = max(1, math.ceil(dur / self.chunk_sec))
            # map each chunk
            for k in range(n_chunks):
                self._index.append((i, k, n_chunks))

        print(f"[ConaikiAudioDataset] {len(self.rows)} clips → {len(self._index)} chunk-samples")

    def __len__(self):
        return len(self._index)

    def _get_audio_path(self, row: Dict) -> str:
        if "conversations" in row:
            for m in row["conversations"]:
                cont = m.get("content", [])
                if isinstance(cont, list):
                    for seg in cont:
                        if seg.get("type") == "audio":
                            return seg.get("audio_url")
            raise ValueError("No audio found in 'conversations' item.")
        return row["audio_path"]  # simple schema

    def _load_audio(self, path: str):
        wav, sr = torchaudio.load(path)  # [C, S]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)
        if sr != self._sr:
            wav = torchaudio.functional.resample(wav, sr, self._sr)
            sr = self._sr
        return wav, sr

    def _slice_wave(self, wav: torch.Tensor, sr: int, k: int) -> torch.Tensor:
        """Return chunk k (0-based). If cumulative=True, return [0 : (k+1)*chunk_sec]."""
        if self.cumulative_audio:
            end = int((k + 1) * self.chunk_sec * sr)
            end = min(end, wav.shape[-1])
            return wav[:end]
        else:
            t0 = int(k * self.chunk_sec * sr)
            t1 = int((k + 1) * self.chunk_sec * sr)
            t1 = min(t1, wav.shape[-1])
            return wav[t0:t1]

    def _build_conversation(self, row: Dict, is_final: bool) -> List[Dict]:
        if "conversations" in row:
            conv = []
            transcript = ""
            for m in row["conversations"]:
                if m["role"] == "system":
                    conv.append(m)  # optional, keep or drop
                elif m["role"] == "user":
                    cont = []
                    if isinstance(m.get("content"), str):
                        cont.append({"type": "text", "text": m["content"]})
                    else:
                        for seg in m["content"]:
                            if seg.get("type") in ("audio", "text"):
                                cont.append(seg)
                    conv.append({"role": "user", "content": cont})
                elif m["role"] == "assistant":
                    # pull transcript text if present
                    t = "".join(seg["text"] for seg in m.get("content", []) if seg.get("type") == "text")
                    if t:
                        transcript = t

            # inject our gate token turn
            if is_final:
                # gate token + transcript
                asst_text = "<TRANSLATE>"
                if transcript:
                    asst_text = asst_text + " " + transcript
                conv.append({"role": "assistant", "content": [{"type": "text", "text": asst_text}]})
            else:
                conv.append({"role": "assistant", "content": [{"type": "text", "text": "<WAIT>"}]})
            return conv

        # simple schema
        user_cont = []
        if row.get("prompt"):
            user_cont.append({"type": "text", "text": row["prompt"]})
        user_cont.append({"type": "audio", "audio_url": row["audio_path"]})
        conv = [{"role": "user", "content": user_cont}]

        if is_final and row.get("assistant"):
            conv.append({"role": "assistant", "content": [{"type": "text", "text": "<TRANSLATE> " + row["assistant"]}]})
        else:
            conv.append({"role": "assistant", "content": [{"type": "text", "text": "<WAIT>"}]})
        return conv


    def __getitem__(self, idx):
        row_idx, k, n_chunks = self._index[idx]
        row = self.rows[row_idx]
        audio_path = self._get_audio_path(row)

        # 1) Build conversation (assistant text only on final chunk)
        is_final = (k == n_chunks - 1)
        formatted = self._build_conversation(row, is_final=is_final)

        text = self.processor.apply_chat_template(
            [formatted],
            add_generation_prompt=False,
            tokenize=False,
        )

        # 2) Load & slice audio
        wav, sr = self._load_audio(audio_path)
        wav_k = self._slice_wave(wav, sr, k)
        wav_np = wav_k.detach().cpu().float().numpy()

        # 3) Let processor do BOTH: tokens + audio features (+ its own attention masks)
        proc = self.processor(
            text=text,
            audio=[wav_np],            # IMPORTANT: singular 'audio' and list for batch
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",              # don't override its audio padding policy
            truncation=True,
            max_length=self.max_seq_length,  # text max length only
        )

        # squeeze batch dim
        input_ids = proc["input_ids"].squeeze(0)
        attention_mask = proc["attention_mask"].squeeze(0)
        input_features = proc["input_features"].squeeze(0)              # [n_mels, T]
        feat_mask = proc["feature_attention_mask"].squeeze(0).long()    # [T]

        # 4) Labels: assistant-only (final chunk has text, others have none)
        pad_id = self.processor.tokenizer.pad_token_id or 151643
        labels = build_labels(
            input_ids=input_ids,
            pad_id=pad_id,
            formatted_msgs=formatted,
            tokenizer=self.processor.tokenizer,
            mode=self.label_masking,
            # set to None (default) to TRAIN on <WAIT>/<TRANSLATE>;
            # or pass SPECIAL_IDS to ignore them in the LM loss.
            ignore_gate_token_ids=None,
        )

        # 5) Gate label
        gate_label = TRANSLATE if is_final else WAIT

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,           # [feat_dim, T]
            "feature_attention_mask": feat_mask,        # [T]
            "labels": labels,
            "gate_labels": torch.tensor(gate_label, dtype=torch.long),
        }

WAIT, TRANSLATE = 0, 1
LABEL_MAP = {"WAIT": WAIT, "TRANSLATE": TRANSLATE, 0: WAIT, 1: TRANSLATE}

class StreamingGateDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor,
        max_seq_length: int = 512,
        # accept unused args to keep your call site unchanged
        label_masking: str = None,
        chunk_sec: float = None,
        cumulative_audio: bool = True,
        # optional override root where audio files live
        audio_root: str | None = None,
        verify_n: int = 5,   # sanity check a few paths at init
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length

        # where the JSONL lives
        self.json_dir = Path(data_path).expanduser().resolve().parent
        self.audio_root = Path(audio_root).expanduser().resolve() if audio_root else None
        self.sr_target = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", 16000)

        def _resolve_path(p: str) -> Path:
            """Return absolute existing path for p (which may be relative)."""
            p = os.path.normpath(os.path.expanduser(p))
            pp = Path(p)
            # If already absolute and exists, use it.
            if pp.is_absolute() and pp.exists():
                return pp

            # Try audio_root, then JSON dir, then CWD as last resort.
            candidates = []
            if self.audio_root is not None:
                candidates.append(self.audio_root / p)
            candidates.append(self.json_dir / p)
            candidates.append(Path.cwd() / p)

            for cand in candidates:
                if cand.exists():
                    return cand.resolve()

            # Give a helpful error with attempted locations.
            attempted = "\n  ".join(str(c.resolve()) for c in candidates)
            raise FileNotFoundError(
                f"Could not resolve audio path '{p}'. Tried:\n  {attempted}"
            )

        # Load samples
        self.samples = []
        with open(data_path, "r") as f:
            for line in f:
                j = json.loads(line.strip())
                gate = j.get("gate_label")
                if gate not in LABEL_MAP:
                    try:
                        gate = int(gate)
                    except Exception:
                        raise ValueError(f"Unrecognized gate_label: {j.get('gate_label')}")
                self.samples.append({
                    "audio_path": _resolve_path(j["audio_path"]),
                    "gate": LABEL_MAP[gate],
                    "chunk_index": j.get("chunk_index"),
                    "total_chunks": j.get("total_chunks"),
                    "chunk_duration_sec": j.get("chunk_duration_sec"),
                    "transcript": j.get("transcript", ""),
                    "original_audio": j.get("original_audio", None),
                })

        # Quick existence / readability sanity check
        for s in self.samples[:verify_n]:
            if not Path(s["audio_path"]).exists():
                raise FileNotFoundError(f"Audio file not found: {s['audio_path']}")  # should be caught above already

        wait_n = sum(1 for s in self.samples if s["gate"] == WAIT)
        trans_n = len(self.samples) - wait_n
        print(f"[StreamingGateDataset] loaded {len(self.samples)} chunks | WAIT={wait_n} TRANSLATE={trans_n}")
        print(f"[StreamingGateDataset] base json_dir={self.json_dir}")
        if self.audio_root:
            print(f"[StreamingGateDataset] override audio_root={self.audio_root}")

    def __len__(self):
        return len(self.samples)

    def _load_chunk(self, path: Path):
        wav, sr = torchaudio.load(str(path))  # [C,S]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)
        wav = wav.squeeze(0)  # [S]
        if sr != self.sr_target:
            wav = torchaudio.functional.resample(wav, sr, self.sr_target)
            sr = self.sr_target
        return wav, sr

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav, sr = self._load_chunk(item["audio_path"])
        wav_np = wav.detach().cpu().float().numpy()

        gate_token = "<WAIT>" if item["gate"] == WAIT else "<TRANSLATE>"

        # minimal convo: user(audio) + assistant(<GATE>)
        # audio_url is just a placeholder; real waveform passed in `audio=[wav_np]`
        conv = [
            {"role": "system", "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}]},
            {"role": "user", "content": [{"type": "audio", "audio_url": str(item["audio_path"])}]},
            {"role": "assistant", "content": [{"type": "text", "text": gate_token}]},
        ]
        text = self.processor.apply_chat_template([conv], add_generation_prompt=False, tokenize=False)

        proc = self.processor(
            text=text,
            audio=[wav_np],
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
        )

        labels = torch.full_like(proc["input_ids"].squeeze(0), -100)

        return {
            "input_ids": proc["input_ids"].squeeze(0),
            "attention_mask": proc["attention_mask"].squeeze(0),
            "input_features": proc["input_features"].squeeze(0),
            "feature_attention_mask": proc["feature_attention_mask"].squeeze(0).long(),
            "labels": labels,
            "gate_labels": torch.tensor(item["gate"], dtype=torch.long),
        }





# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────

def freeze_modules(model, sargs: ScriptArgs):
    if sargs.freeze_vision_encoder and hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False
        logger.info("Froze vision encoder")

    if sargs.freeze_audio_encoder and hasattr(model, "audio_tower"):
        for p in model.audio_tower.parameters():
            p.requires_grad = False
        logger.info("Froze audio encoder")

    if sargs.freeze_llm_backbone and hasattr(model, "model"):
        for p in model.model.parameters():
            p.requires_grad = False
        logger.info("Froze LLM backbone")

    if sargs.freeze_conaiki_modules:
        for n, p in model.named_parameters():
            if "conaiki" in n.lower():
                p.requires_grad = False
        logger.info("Froze conaiki modules")

# TODO: answer the question why small model usually have {tied} embeddings while larger not. 
# For example: Qwen3
def unfreeze_modules(model, sargs: ScriptArgs):
    
    if sargs.unfreeze_embed_tokens=="UNFREEZE" and hasattr(model.model, "embed_tokens"):
        for p in model.model.embed_tokens.parameters():
            p.requires_grad = True
    else:
        logger.info("model.model.embed_tokens() -> freezed")
    
    if sargs.unfreeze_lm_head=="UNFREEZE" and hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
    elif sargs.unfreeze_lm_head=="FREEZE":
        for p in model.lm_head.parameters():
            p.requires_grad = False
        logger.info("model.lm_head() -> freezed")

    if sargs.unfreeze_audio_projector=="UNFREEZE" and hasattr(model.audio_tower, "proj"):
        for n, p in model.audio_tower.named_parameters():
            if "proj" in n:
                p.requires_grad = True
    else:
        logger.info("model.audui_tower.proj() -> freezed")
    
    if sargs.unfreeze_conaiki_gate_proj=="UNFREEZE" and hasattr(model, "conaiki_gate"):
        for p in model.conaiki_gate.parameters():
            p.requires_grad = True
    else:
        logger.info("model.conaiki_gate() -> freezed")

def _find_subseq(haystack, needle, start=0):
    n, m = len(haystack), len(needle)
    if m == 0:
        return start
    for i in range(start, n - m + 1):
        if haystack[i:i+m] == needle:
            return i
    return -1

def build_labels(input_ids, pad_id, formatted_msgs, tokenizer,
                 mode="assistant_only", ignore_gate_token_ids=None, mask_all=False):
    """
    mode: "assistant_only" | "last_assistant_only" | "all"
    mask_all=True -> return a full -100 tensor (Phase A).
    """
    if mask_all:
        return torch.full_like(input_ids, -100)

    if mode == "all":
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        return labels

    full_ids = input_ids.tolist()
    labels = torch.full_like(input_ids, -100)

    # collect assistant text chunks
    chunks = []
    for msg in formatted_msgs:
        if msg["role"] != "assistant":
            continue
        txt = "".join(seg["text"] for seg in msg["content"] if seg.get("type") == "text")
        if txt:
            chunks.append(txt)
    if mode == "last_assistant_only" and chunks:
        chunks = [chunks[-1]]

    cursor = 0
    for chunk in chunks:
        for variant in (chunk, "\n"+chunk, chunk+"\n", "\n"+chunk+"\n"):
            sub_ids = tokenizer(variant, add_special_tokens=False).input_ids
            pos = _find_subseq(full_ids, sub_ids, start=cursor)
            if pos != -1:
                end = pos + len(sub_ids)
                labels[pos:end] = input_ids[pos:end]
                cursor = end - 1
                break

    labels[input_ids == pad_id] = -100
    if ignore_gate_token_ids:
        for sid in ignore_gate_token_ids:
            labels[input_ids == sid] = -100
    return labels




# ──────────────────────────────────────────────────────────────────────────────
# Main training script
# ──────────────────────────────────────────────────────────────────────────────
def main():
    sargs = ScriptArgs()
    if sargs.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", sargs.wandb_project)

    #  - remove_unused_columns=False → не выкидывать кастомные ключи из батча.
    hf_args = HFTrainingArguments(
        output_dir=sargs.output_dir,
        
        # precision / CUDA
        bf16=True,                              
        dataloader_num_workers=4, # TODO: change to correct

        # оптимизация / шаги
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=100,

        # логгинг/оценка/сейвы
        logging_steps=1,
        # evaluation_strategy="steps" if sargs.val_data_path else "no",
        # eval_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_safetensors=True,
        save_total_limit=10,
        num_train_epochs=3,

        deepspeed="ds_config.json",             # в конфиге: "stage3_gather_16bit_weights_on_model_save": true for saving properly.

        # репорты
        report_to=["wandb"] if sargs.use_wandb else [],

        # прочее
        remove_unused_columns=False, # ?
        seed=sargs.seed,
    )

    # фиксируем сид
    set_seed(sargs.seed)

    # модель/процессор
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        # TODO: question to AI how low_cpu_mem_usage=True helps with training speed?
        # ANSWER: as I understand 
        sargs.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None, trust_remote_code=True 
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(sargs.model_path)
    
    # TODO: question how gradient_checkpointing helps? Also is it True that it's 20-30% slower to train with grad_check = ON
    model.config.use_cache = False # if doing grad_check add cache as well
    model.gradient_checkpointing_enable()
    freeze_modules(model, sargs)
    unfreeze_modules(model, sargs)

    model.config.gate_loss_weight = sargs.gate_loss_weight  # weight for the gate_loss

    # # TODO: do we need this?
    # if hasattr(model, "conaiki_align_modules"):
    #     model.conaiki_align_modules()


    train_ds = StreamingGateDataset(
        data_path=sargs.train_data_path,
        processor=processor,
        max_seq_length=sargs.max_seq_length,
        label_masking=sargs.label_masking,
        chunk_sec=sargs.chunk_sec,
        cumulative_audio=sargs.cumulative_audio,
        audio_root=sargs.audio_root,

    )
    # eval_ds = (ConaikiAudioDataset(
    #     data_path=sargs.val_data_path,
    #     processor=processor,
    #     max_seq_length=sargs.max_seq_length,
    #     label_masking=sargs.label_masking,
    #     chunk_sec=sargs.chunk_sec,
    #     cumulative_audio=sargs.cumulative_audio,
    # ) if sargs.val_data_path else None)



    # простой коллатор: мы уже паддим в датасете до max_length
    def _pad_time(x, T_max):
        # x: [feat_dim, T]  → pad on the right along time
        feat_dim, T = x.shape
        if T == T_max:
            return x
        return F.pad(x, (0, T_max - T))

    def _pad_mask(m, T_max):
        # m: [T]  → pad zeros to the right
        T = m.shape[0]
        if T == T_max:
            return m
        return F.pad(m, (0, T_max - T), value=0)

    def collate_fn(batch):
        out = {}
        # text fields (already same length due to your text max_length)
        out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
        out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
        out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
        # audio: pad to max T in this batch
        T_max = max(b["feature_attention_mask"].shape[0] for b in batch)
        out["input_features"] = torch.stack([_pad_time(b["input_features"], T_max) for b in batch], dim=0)
        out["feature_attention_mask"] = torch.stack([_pad_mask(b["feature_attention_mask"], T_max) for b in batch], dim=0)
        # gate labels
        if "gate_labels" in batch[0]:
            out["gate_labels"] = torch.stack([b["gate_labels"] for b in batch], dim=0)
        return out

    # Trainer
    trainer = Trainer(
        model=model,
        args=hf_args,
        train_dataset=train_ds,
        # eval_dataset=eval_ds,
        processing_class=processor,
        data_collator=collate_fn,
    )

    trainer.train() # we can provide (resume_from_checkpoint=...)

    final_dir = Path(sargs.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    logger.info(f"Saved final model to {final_dir}")

    processor.save_pretrained(str(final_dir))
    logger.info("Done.")


if __name__ == "__main__":
    main()
