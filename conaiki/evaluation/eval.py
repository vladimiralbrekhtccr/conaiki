import os
import time
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,  # OK to keep; we’ll also handle fallback
)

# ==== CONFIG ====
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

MODEL_PATH = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/conaiki/qwen_omni_finetuned_common_voice_for_qwen_train_less_than_3_sec/final_model"
JSONL_PATH = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/conaiki/data/common_voice_for_qwen/less_than_3_sec/processed/streaming_chunks_padded.jsonl"
TRANSLATE_THRESHOLD = 0.5
USE_AUTOCast = False  # set True if you want cuda autocast(bfloat16) during forward
# =================


def load_and_prep_audio(audio_path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)  # [S]


def _pool_last_audio_token(hidden_states: torch.Tensor,
                           input_ids: torch.LongTensor,
                           attention_mask: torch.LongTensor,
                           audio_token_index: int) -> torch.Tensor:
    """
    hidden_states: [B, T, D] (last hidden layer from the text model)
    input_ids:     [B, T]
    attention_mask:[B, T]
    Returns:       [B, D] pooled at last audio token; fallback = last attended token.
    """
    B, T, D = hidden_states.shape
    device = hidden_states.device

    audio_mask = (input_ids == audio_token_index)  # [B, T]
    has_audio = audio_mask.any(dim=1)              # [B]

    # default: last attended token
    # (find last index where attention_mask==1)
    attn = attention_mask.to(torch.int8)
    rev = torch.flip(attn, dims=[1])                        # [B, T]
    last_from_end = torch.argmax(rev, dim=1)                # [B]
    last_attn_idx = (T - 1) - last_from_end                 # [B]
    last_idx = last_attn_idx.clone()

    # where audio exists, take last audio position instead
    if has_audio.any():
        rev_audio = torch.flip(audio_mask.to(torch.int8), dims=[1])
        last_audio_from_end = torch.argmax(rev_audio, dim=1)
        last_audio_idx = (T - 1) - last_audio_from_end
        last_idx = torch.where(has_audio, last_audio_idx, last_attn_idx)

    gather_idx = last_idx.view(B, 1, 1).expand(B, 1, D)     # [B,1,D]
    pooled = hidden_states.gather(dim=1, index=gather_idx).squeeze(1)
    return pooled  # [B, D]


@torch.inference_mode()
def get_gate_prediction(model, processor, wav_tensor: torch.Tensor, system_prompt: dict) -> torch.Tensor:
    """
    Returns probabilities over [WAIT, TRANSLATE].
    Works if the model natively returns gate_logits OR if we have to compute them
    from hidden_states + model.conaiki_gate.
    """
    target_sr = processor.feature_extractor.sampling_rate

    # conversation (minimal, mirrors training structure on the user/audio side)
    conversation = [
        system_prompt,
        {"role": "user", "content": [{"type": "audio", "audio_url": "placeholder.wav"}]},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)

    # Prepare inputs (single example)
    inputs = processor(
        text=text,
        audio=[wav_tensor.cpu().numpy()],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,
    )

    # Move to the right device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Try the fast path: model returns gate_logits directly (your patched forward).
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(model.device.type == "cuda" and USE_AUTOCast)):
            outputs = model(**inputs, return_gate_logits=True)
        if hasattr(outputs, "gate_logits") and outputs.gate_logits is not None:
            probs = torch.softmax(outputs.gate_logits.float(), dim=-1).squeeze(0)
            return probs
    except TypeError:
        # The model forward might not accept return_gate_logits; we’ll fall back below.
        pass

    # Fallback: compute gate_logits ourselves from hidden states.
    if not hasattr(model, "conaiki_gate"):
        raise RuntimeError(
            "Model does not expose 'gate_logits' AND has no 'conaiki_gate' module. "
            "Load your custom class (with gate head) or re-export the model with that head."
        )

    # We need last hidden states BEFORE lm_head; ask the model for hidden_states.
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(model.device.type == "cuda" and USE_AUTOCast)):
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden_states; cannot build gate logits fallback.")

    # The last entry in hidden_states is the last layer of the text model (before lm_head).
    # NOTE: For Qwen* decoders, hidden_states is a list of all layer outputs. Pick the last.
    last_h = outputs.hidden_states[-1]  # [B, T, D]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    audio_idx = getattr(model.config, "audio_token_index", None)
    if audio_idx is None:
        # If the config doesn’t have it, try a reasonable fallback.
        # But ideally, this should exist in Qwen-Omni configs.
        audio_idx = 151666  # (example) – replace with your actual audio token id if known.

    pooled = _pool_last_audio_token(last_h, input_ids, attention_mask, audio_idx)  # [B, D]
    gate_logits = model.conaiki_gate(pooled)  # [B, 2] for WAIT/TRANSLATE (or [B, C] if you set C=2)
    probs = torch.softmax(gate_logits.float(), dim=-1).squeeze(0)
    return probs


def evaluate_gate_model():
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from: {MODEL_PATH} ...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,   # important if your class was saved with custom code
    ).to(device).eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    print("Model and processor loaded.")

    system_prompt = {
        "role": "system",
        "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
    }

    # Load dataset
    print(f"Loading evaluation data from: {JSONL_PATH}")
    jsonl_path = Path(JSONL_PATH).expanduser().resolve()
    base_dir = jsonl_path.parent
    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    WAIT_IDX, TRANS_IDX = 0, 1
    label_map = {"WAIT": WAIT_IDX, "TRANSLATE": TRANS_IDX}

    correct = 0
    total = 0
    cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    print(f"\nStarting evaluation on {len(samples)} audio chunks...")
    pbar = tqdm(samples, desc="Evaluating Gate Predictions")
    for sample in pbar:
        try:
            true_lbl_s = sample["gate_label"]
            true_lbl = label_map[true_lbl_s]

            audio_rel = sample["audio_path"]
            audio_path = (base_dir / audio_rel).as_posix()

            wav = load_and_prep_audio(audio_path, processor.feature_extractor.sampling_rate)
            probs = get_gate_prediction(model, processor, wav, system_prompt)

            p_translate = probs[TRANS_IDX].item()
            pred_lbl = TRANS_IDX if p_translate >= TRANSLATE_THRESHOLD else WAIT_IDX

            correct += int(pred_lbl == true_lbl)
            if   pred_lbl == TRANS_IDX and true_lbl == TRANS_IDX: cm["tp"] += 1
            elif pred_lbl == WAIT_IDX  and true_lbl == WAIT_IDX:  cm["tn"] += 1
            elif pred_lbl == TRANS_IDX and true_lbl == WAIT_IDX:  cm["fp"] += 1
            elif pred_lbl == WAIT_IDX  and true_lbl == TRANS_IDX: cm["fn"] += 1

            total += 1

        except Exception as e:
            print(f"\nSkipping sample due to error: {e} | Sample: {sample.get('audio_path')}")

    # Report
    print("\n--- Evaluation Complete ---")
    acc = (correct / total * 100.0) if total else 0.0
    print(f"Overall Accuracy: {acc:.2f}% ({correct} / {total})")

    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    print("\nConfusion Matrix (Positive = TRANSLATE):")
    print(f"  TP: {tp:4d}   TN: {tn:4d}   FP: {fp:4d}   FN: {fn:4d}")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\nMetrics (TRANSLATE):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    print(f"\nTotal runtime: {time.perf_counter() - start:.2f} s")


if __name__ == "__main__":
    evaluate_gate_model()