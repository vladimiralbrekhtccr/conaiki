# train_qwen_omni_trainer.py
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torchaudio
import numpy as np 
from torch.utils.data import Dataset
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
    model_path: str = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/models/Qwen2.5-Omni-7B"
    output_dir: str = "./qwen_omni_finetuned" # it's run name and out_dir

    # Данные
    train_data_path: str = "data/audio/a_train_data.jsonl" # or "data/train_data.jsonl"
    val_data_path: Optional[str] = "data/val_data.jsonl"
    max_seq_length: int = 2048
    label_masking: str = "assistant_only" # or "assistant_only" | "last_assistant_only" | "all"

    # Заморозка модулей
    freeze_vision_encoder: bool = True
    freeze_llm_backbone: bool = True
    freeze_audio_encoder: bool = True
    freeze_conaiki_modules: bool = False  

    # Прочее
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
        print("Conaiki_1")
        print(formatted[0])

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

class AudioOnlyDataset(Dataset):
    """
    Produces batches with:
      - input_ids, attention_mask
      - input_features [feat_dim, T_fixed]  (padded/truncated)
      - feature_attention_mask [T_fixed]
      - labels
    Works with your Qwen2.5-Omni model forward(native audio path).
    """

    def __init__(
        self,
        data_path: str,
        processor: Qwen2_5OmniProcessor,
        max_seq_length: int = 2048,
        max_audio_frames: int = 4000,         # tune this for your GPU / audio lengths
        label_masking: str = "assistant_only"# "assistant_only" | "last_assistant_only" | "all"
        # default_system: Optional[str] = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.max_audio_frames = max_audio_frames
        self.label_masking = label_masking
        self.rows: List[Dict] = []

        with open(data_path, "r") as f:
            for line in f:
                self.rows.append(json.loads(line.strip()))

        # self.default_system = default_system

        # handy for resampling if using torchaudio
        self._target_sr = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", 16000)

        print(f"[AudioOnlyDataset] loaded {len(self.rows)} rows from {data_path}")

    def __len__(self):
        return len(self.rows)

    def _load_audio(self, path: str):
        wav, sr = torchaudio.load(path)                      # [C, S]
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)                  # mono
        wav = wav.squeeze(0)                                 # [S]
        if sr != self._target_sr:
            wav = torchaudio.functional.resample(wav, sr, self._target_sr)
            sr = self._target_sr
        return wav, sr

    def _build_conversation(self, row: Dict) -> (List[Dict], str):
        """
        Returns (formatted_msgs, audio_path)
        """
        if "conversations" in row:
            conv = row["conversations"]
            # normalize content for apply_chat_template
            formatted = []
            audio_path = None
            for msg in conv:
                if isinstance(msg.get("content"), str):
                    content = [{"type": "text", "text": msg["content"]}]
                else:
                    content = []
                    for seg in msg.get("content", []):
                        if seg.get("type") == "audio":
                            audio_path = seg.get("audio_url")
                            content.append(seg)
                        elif seg.get("type") == "text":
                            content.append(seg)
                        else:
                            # ignore images/videos in this audio-only dataset
                            pass
                formatted.append({"role": msg["role"], "content": content})
            if audio_path is None:
                raise ValueError("No audio found in conversations item.")
            return formatted, audio_path

        # simple item: {"audio_path": "...", "assistant": "...", "prompt": "...?"}
        audio_path = row["audio_path"]
        assistant_text = row.get("assistant", "")
        user_prompt = row.get("prompt", "")
        #system_msg = self.default_system or row.get("system", None)

        formatted = []
        # if system_msg:
        #     formatted.append({"role": "system", "content": [{"type": "text", "text": system_msg}]})
        # user: only audio; optionally prepend text prompt
        user_content = []
        if user_prompt:
            user_content.append({"type": "text", "text": user_prompt})
        user_content.append({"type": "audio", "audio_url": audio_path})
        formatted.append({"role": "user", "content": user_content})
        # assistant: target
        if assistant_text:
            formatted.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})
        return formatted, audio_path

    def __getitem__(self, idx):
        row = self.rows[idx]

        # 1) build conversation text that includes the audio token
        formatted, audio_path = self._build_conversation(row)
        text = self.processor.apply_chat_template(
            [formatted],
            add_generation_prompt=False,
            tokenize=False,      # important: pass raw text to processor
        )

        # 2) load audio -> mono, target sr
        wav, sr = self._load_audio(audio_path)     # wav: torch.Tensor [S] or np.ndarray [S]
        if isinstance(wav, torch.Tensor):
            wav_np = wav.detach().cpu().float().numpy()
        else:
            wav_np = wav.astype(np.float32, copy=False)

        # 3) processor does BOTH: tokenize text + extract audio features
        proc_out = self.processor(
            text=text,
            audio=[wav_np],          # singular 'audio'
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",    # processor enforces this anyway
            truncation=True,
            max_length=self.max_seq_length,  # text max length; audio handled separately
        )

        # squeeze batch
        input_ids      = proc_out["input_ids"].squeeze(0)
        attention_mask = proc_out["attention_mask"].squeeze(0)
        input_features = proc_out["input_features"].squeeze(0)          # [feat_dim, T]
        # DO NOT trust/use processor-provided feature_attention_mask after we resize
        # feat_mask      = proc_out.get("feature_attention_mask").squeeze(0)  # ← remove

        # 4) fix time length deterministically and REBUILD mask from the final T
        T    = input_features.shape[-1]
        Ttgt = self.max_audio_frames

        if T > Ttgt:
            input_features = input_features[..., :Ttgt]
            feat_mask = torch.ones(Ttgt, dtype=torch.long)
        elif T < Ttgt:
            pad = Ttgt - T
            input_features = torch.nn.functional.pad(input_features, (0, pad))
            feat_mask = torch.cat(
                [torch.ones(T, dtype=torch.long), torch.zeros(pad, dtype=torch.long)],
                dim=0
            )
        else:
            # T == Ttgt → build mask to EXACTLY match features
            feat_mask = torch.ones(Ttgt, dtype=torch.long)

        # 6) labels (assistant-only by default)
        pad_id = self.processor.tokenizer.pad_token_id or 151643
        labels = build_labels(
            input_ids=input_ids,
            pad_id=pad_id,
            formatted_msgs=formatted,
            tokenizer=self.processor.tokenizer,
            mode=self.label_masking,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,         # [feat_dim, T_fixed]
            "feature_attention_mask": feat_mask,      # [T_fixed]
            "labels": labels,
        }



# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────
def fix_model_config(model):
    # переносим vocab_size наверх, если он был в text_config
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "vocab_size"):
        model.config.vocab_size = model.config.text_config.vocab_size
        logger.info(f"Set vocab_size to {model.config.vocab_size}")

    if not getattr(model.config, "pad_token_id", None):
        if hasattr(model.config, "text_config") and getattr(model.config.text_config, "pad_token_id", None):
            model.config.pad_token_id = model.config.text_config.pad_token_id
        else:
            model.config.pad_token_id = 151643
        logger.info(f"Set pad_token_id to {model.config.pad_token_id}")
    return model


def freeze_modules(model, args: ScriptArgs):
    # распаковывать ничего не нужно — Trainer сам это сделает
    if args.freeze_vision_encoder and hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False
        logger.info("Froze vision encoder")

    if args.freeze_audio_encoder and hasattr(model, "audio_tower"):
        for p in model.audio_tower.parameters():
            p.requires_grad = False
        logger.info("Froze audio encoder")

    if args.freeze_llm_backbone and hasattr(model, "model"):
        for p in model.model.parameters():
            p.requires_grad = False
        logger.info("Froze LLM backbone")

    if args.freeze_conaiki_modules:
        for n, p in model.named_parameters():
            if "conaiki" not in n.lower():
                p.requires_grad = False
        logger.info("Training only *conaiki* modules")


def _find_subseq(haystack, needle, start=0):
    n, m = len(haystack), len(needle)
    if m == 0:
        return start
    for i in range(start, n - m + 1):
        if haystack[i:i+m] == needle:
            return i
    return -1

def build_labels(input_ids, pad_id, formatted_msgs, tokenizer, mode="assistant_only"):
    """
    mode: "assistant_only" | "last_assistant_only" | "all"
    """
    if mode == "all":
        labels = input_ids.clone()
        labels[labels == pad_id] = -100
        return labels

    full_ids = input_ids.tolist()
    labels = torch.full_like(input_ids, -100)

    # gather assistant text chunks
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
        dataloader_num_workers=0, # TODO: change to correct

        # оптимизация / шаги
        per_device_train_batch_size=1,
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
        save_steps=100,
        save_safetensors=True,
        save_total_limit=3,
        num_train_epochs=1,

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
        sargs.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None # TODO: question to AI how low_cpu_mem_usage=True helps with training speed?
    )
    model = fix_model_config(model)
    model.gradient_checkpointing_enable()  # TODO: why and  what is that?
    freeze_modules(model, sargs)

    processor = Qwen2_5OmniProcessor.from_pretrained(sargs.model_path)

    # датасеты
    label_masking_arg = sargs.label_masking
    train_ds = AudioOnlyDataset(
        data_path=sargs.train_data_path,
        processor=processor,
        max_seq_length=sargs.max_seq_length,
        max_audio_frames=4000,             # tune for your audio duration # ??
        label_masking=label_masking_arg,
    )
    eval_ds = (AudioOnlyDataset(
        data_path=sargs.val_data_path,
        processor=processor,
        max_seq_length=sargs.max_seq_length,
        max_audio_frames=4000,
        label_masking=label_masking_arg,
    ) if sargs.val_data_path else None)


    # простой коллатор: мы уже паддим в датасете до max_length
    def collate_fn(features):
        # fixed-length tensors → simple stack
        out = {}
        for k in features[0].keys():
            out[k] = torch.stack([f[k] for f in features], dim=0)
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
