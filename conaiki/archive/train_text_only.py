# train_qwen_omni_trainer.py
import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import torch
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
# Пользовательские аргументы (всё, чего нет в HF TrainingArguments)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ScriptArgs:
    model_path: str = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/models/Qwen2.5-Omni-7B"
    output_dir: str = "./qwen_omni_finetuned" # it's run name and out_dir

    # Данные
    train_data_path: str = "data/train_data.jsonl"
    val_data_path: Optional[str] = "data/val_data.jsonl"
    max_seq_length: int = 2048

    # Заморозка модулей
    freeze_vision_encoder: bool = True
    freeze_llm_backbone: bool = True
    freeze_audio_encoder: bool = True
    train_conaiki_modules_only: bool = False  # тренировать только свои модули

    # Прочее
    seed: int = 42
    use_wandb: bool = True
    wandb_project: str = "qwen-omni-training"


# ──────────────────────────────────────────────────────────────────────────────
# Датасеты
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
# Audio Only Dataset
# ──────────────────────────────────────────────────────────────────────────────

import os, json
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset

# optional: torchaudio first, librosa fallback
try:
    import torchaudio
    HAVE_TORCHAUDIO = True
except Exception:
    HAVE_TORCHAUDIO = False
    import librosa
    import numpy as np

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
        processor,
        max_seq_length: int = 2048,
        max_audio_frames: int = 4000,         # tune this for your GPU / audio lengths
        label_masking: str = "assistant_only",# "assistant_only" | "last_assistant_only" | "all"
        default_system: Optional[str] = None,
    ):
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.max_audio_frames = max_audio_frames
        self.label_masking = label_masking
        self.rows: List[Dict] = []

        with open(data_path, "r") as f:
            for line in f:
                self.rows.append(json.loads(line.strip()))

        self.default_system = default_system

        # handy for resampling if using torchaudio
        self._target_sr = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", 16000)

        print(f"[AudioOnlyDataset] loaded {len(self.rows)} rows from {data_path}")

    def __len__(self):
        return len(self.rows)

    def _load_audio(self, path: str):
        if HAVE_TORCHAUDIO:
            wav, sr = torchaudio.load(path)                      # [C, S]
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)                  # mono
            wav = wav.squeeze(0)                                 # [S]
            if sr != self._target_sr:
                wav = torchaudio.functional.resample(wav, sr, self._target_sr)
                sr = self._target_sr
            return wav, sr
        else:
            y, sr = librosa.load(path, sr=self._target_sr, mono=True)
            return torch.from_numpy(y), sr

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

        # simple item: {"audio_path": "...", "assistant_text": "...", "prompt": "...?"}
        audio_path = row["audio_path"]
        assistant_text = row.get("assistant_text", "")
        user_prompt = row.get("prompt", "")
        system_msg = self.default_system or row.get("system", None)

        formatted = []
        if system_msg:
            formatted.append({"role": "system", "content": [{"type": "text", "text": system_msg}]})
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
        # 1) prepare conversation text (with audio placeholder)
        formatted, audio_path = self._build_conversation(row)

        text = self.processor.apply_chat_template(
            [formatted],
            add_generation_prompt=False,
            tokenize=False,             # ← we want raw text for processor(...)
        )

        # 2) load audio
        wav, sr = self._load_audio(audio_path)

        # 3) let processor make BOTH text tokens and audio features
        proc_out = self.processor(
            text=text,
            audios=[(wav, sr)],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
        )
        # squeeze batch
        input_ids = proc_out["input_ids"].squeeze(0)                    # [T_text]
        attention_mask = proc_out["attention_mask"].squeeze(0)          # [T_text]
        # native audio path expects [feat_dim, T_audio]
        input_features = proc_out["input_features"].squeeze(0)          # [feat_dim, T_var]
        feat_mask = proc_out.get("feature_attention_mask", None)
        if feat_mask is not None:
            feat_mask = feat_mask.squeeze(0)                             # [T_var]

        # 4) make audio time dimension fixed for easy stacking
        T = input_features.shape[-1]
        Ttgt = self.max_audio_frames
        if T > Ttgt:
            input_features = input_features[..., :Ttgt]
            feat_mask = torch.ones(Ttgt, dtype=torch.long)
        elif T < Ttgt:
            pad = Ttgt - T
            input_features = torch.nn.functional.pad(input_features, (0, pad))  # right-pad time
            feat_mask = torch.cat([torch.ones(T, dtype=torch.long),
                                   torch.zeros(pad, dtype=torch.long)], dim=0)
        else:
            if feat_mask is None:
                feat_mask = torch.ones(T, dtype=torch.long)

        # 5) labels (assistant-only by default)
        pad_id = self.processor.tokenizer.pad_token_id or 151643
        labels = build_labels_assistant_only(
            input_ids=input_ids,
            pad_id=pad_id,
            formatted_msgs=formatted,
            tokenizer=self.processor.tokenizer,
            mode=self.label_masking,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,           # [feat_dim, T_fixed]
            "feature_attention_mask": feat_mask,        # [T_fixed]
            "labels": labels,
        }



# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
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

    if args.train_conaiki_modules_only:
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

def build_labels_assistant_only(input_ids, pad_id, formatted_msgs, tokenizer, mode="assistant_only"):
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
# Основной скрипт — только Trainer, без своего save()
# ──────────────────────────────────────────────────────────────────────────────
def main():
    sargs = ScriptArgs()
    if sargs.use_wandb:
        os.environ.setdefault("WANDB_PROJECT", sargs.wandb_project)

    # HF TrainingArguments. Важно:
    #  - save_strategy="no" → Trainer НЕ будет писать тяжёлые DS-шарды,
    #    а мы сами сохраним финальную модель через trainer.save_model().
    #  - remove_unused_columns=False → не выкидывать кастомные ключи из батча.
    hf_args = HFTrainingArguments(
        output_dir=sargs.output_dir,
        # precision / CUDA
        bf16=True,                              # под bfloat16
        dataloader_num_workers=4,

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

        # DeepSpeed (если нужен ZeRO-3 + сборка весов на сохранении)
        deepspeed="ds_config.json",             # в конфиге: "stage3_gather_16bit_weights_on_model_save": true

        # репорты
        report_to=["wandb"] if sargs.use_wandb else [],

        # прочее
        remove_unused_columns=False,
        seed=sargs.seed,
    )

    # фиксируем сид
    set_seed(sargs.seed)

    # модель/процессор
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        sargs.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map=None
    )
    model = fix_model_config(model)
    model.gradient_checkpointing_enable()  # по желанию
    freeze_modules(model, sargs)

    processor = Qwen2_5OmniProcessor.from_pretrained(sargs.model_path)

    # датасеты
    train_ds = TextConversationDataset(sargs.train_data_path, processor, sargs.max_seq_length)
    eval_ds = TextConversationDataset(sargs.val_data_path, processor, sargs.max_seq_length) if sargs.val_data_path else None

    # простой коллатор: мы уже паддим в датасете до max_length
    def collate_fn(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k in features[0].keys():
            out[k] = torch.stack([f[k] for f in features])
        return out

    # Trainer
    trainer = Trainer(
        model=model,
        args=hf_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,    # Trainer.save_model() сохранит и его
        data_collator=collate_fn,
    )

    # обучение (можно передать resume_from_checkpoint=...)
    trainer.train()

    # как в InternVL: финальное сохранение ТОЛЬКО модели (safetensors)
    final_dir = Path(sargs.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    # save_model сохранит .safetensors + config; tokenizer/processor сохранится автоматом,
    # т.к. мы передали его в `tokenizer=` при создании Trainer.
    trainer.save_model(str(final_dir))
    logger.info(f"Saved final model to {final_dir}")

    # на всякий случай — сохраним процессор отдельно (избыточно, но безопасно)
    processor.save_pretrained(str(final_dir))
    logger.info("Done.")


if __name__ == "__main__":
    main()
