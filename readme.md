# Conaiki — Simultaneous Speech-to-Text Translation (S2TT)

**TL;DR:** Conaiki streams **audio → translation** with **\~0.65–0.85 s** first-word latency and **continuous** target text thereafter. We freeze **Whisper** as the audio encoder, add a tiny **MLP projector** (+ time/type conditioning), and use a **decoder-only LLM** (Qwen-Thinker) that reads **audio tokens** and decides, every \~40–80 ms, whether to output the next translation token or wait—via two control tokens: **`<TRANSLATE>`** / **`<NOT_TRANSLATE>`**. Optional **`<COMMIT>`** freezes clauses so we can prune old audio. Barge-in (interrupt awareness) is a runtime toggle.

---

## 1) Scope & Goals

* **Mode:** **Simultaneous** S2TT only (real-time).
* **Single model:** One LLM backbone does the decisioning and translation; **no multi-model orchestration**.
* **Inputs:** Streaming mic or file → **audio chunks**.
* **Outputs:** **Streaming target text** (optionally TTS later; out of scope here).
* **Barge-in toggle:**

  * **OFF** (broadcast): keep translating through background speech.
  * **ON** (interactive): immediately pause emission on competing speech.

**Non-goals:** offline batch translation; speech synthesis; multi-turn dialog policy.

---

## 2) High-Level System

1. **Overlap-and-center windowing**

   * Window **W=1.8 s**, stride **S=0.24 s**, trust center **C=0.6 s**.
   * Encode W for context, **publish only the center**, and within that only the **new right-tail** (no duplicates).
   * Algorithmic look-ahead ≈ (W−C)/2 = **0.6 s**.

2. **Audio encoder (frozen Whisper)**

   * Mel → Whisper encoder → frame features (10 ms hop).
   * We **don’t** change Whisper.

3. **Projector (tiny MLP) + conditioning**

   * Pack **K** frames → 1 audio token (e.g., K=4 ⇒ \~25 tokens/s).
   * MLP: `d_enc → d_model` (+ **time embedding** from Fourier features; **type embedding** = audio vs text).

4. **LLM backbone (decoder-only)**

   * Qwen2.5-Omni-Thinker text stack.
   * Reads **audio tokens** as prefix, then interleaves **actions** and **text**:

     * Action tokens: **`<TRANSLATE>`**, **`<NOT_TRANSLATE>`** (+ optional **`<COMMIT>`**, **`<EOS>`**).
     * When the action is `<TRANSLATE>`, the LLM emits 1–3 target tokens; when `<NOT_TRANSLATE>`, it waits for more audio.

5. **Monotonicity**

   * Enforced by **incremental feeding**: only past audio exists in the prefix; no future peeking.

---

## 3) Runtime Dataflow (Streaming)

1. **Mic buffer → windows** every **S** seconds.
2. **Whisper encodes** each W-second window.
3. **Center extract (C)** → keep only \[t−C/2, t+C/2] within the window.
4. **New-tail filter** → publish frames with time > `t_published`; update `t_published`.
5. **Pack & project** → audio tokens with time/type conditioning.
6. **Append tokens** to LLM prefix; **decode tick** (\~40–80 ms):

   * Predict **`<TRANSLATE>`** vs **`<NOT_TRANSLATE>`**.
   * If `<TRANSLATE>`, emit 1–3 target tokens (greedy/beam).
   * Optionally output **`<COMMIT>`** at clause boundary → freeze text; **prune** old audio (keep \~6–8 s).
   * If **`<EOS>`**, end turn.

**Latency profile:**

* First word: **0.6 s** (look-ahead) + **0.05–0.25 s** (compute/patience) ⇒ **\~0.65–0.85 s**.
* Steady lag while streaming: **\~0.2–0.4 s** typical; **\~0.6–0.8 s** for verb-final languages.

---

## 4) Interface & Tokens

**Special tokens added to the text tokenizer**

* Actions: `<TRANSLATE>`, `<NOT_TRANSLATE>`, (optional) `<COMMIT>`, `<EOS>`
* Mode/Tags: `<SRC=xx>`, `<TGT=yy>`, optional `<LAG=600ms>`

**Patched `forward` (conceptual):**

```python
out = model.forward(
  input_ids=step_text_ids,                 # action or small burst of text
  past_key_values=past_kv, use_cache=True, return_dict=True,
  # Conaiki external path:
  external_audio_embeds=audio_ext_step,    # [B, Ta_new, d_ext]
  external_audio_times=times_step          # [B, Ta_new] or [Ta_new]
)
```

* We **bypass** the internal audio merge when `external_audio_embeds` is present.
* Projector/time/type modules live **inside** the class; they map `d_ext → d_model` and add temporal conditioning.
* **KV cache** ensures incremental decoding; **no custom mask** required since we never feed future audio.

**Barge-in (runtime):**

* **ON:** upon energy spike/overlap, force `<NOT_TRANSLATE>` for a few ticks.
* **OFF:** ignore; continue translating (broadcast mode).

---

## 5) Minimal Streaming Loop (pseudocode)

```python
past_kv = None
t_published = 0.0
print_prefix = ""

while mic.is_open():
    # 1) slide window, encode, extract center, publish only new right-tail
    center_frames, frame_times = whisper_center_tail(W, C, S, t_published)
    t_published = max(t_published, frame_times.max())

    # 2) pack & project to audio tokens
    audio_ext_step, times_step = pack_and_project(center_frames, frame_times)

    # 3) action (gate) — MVP rule or learned policy
    action = "<TRANSLATE>" if gate_confident() else "<NOT_TRANSLATE>"
    step_ids = tok.encode(action, return_tensors="pt").to(device)

    # 4) LLM step
    out = model.forward(
        input_ids=step_ids,
        past_key_values=past_kv, use_cache=True, return_dict=True,
        external_audio_embeds=audio_ext_step,
        external_audio_times=times_step,
    )
    past_kv = out.past_key_values

    # 5) if action==TRANSLATE → emit 1–3 tokens
    if action == "<TRANSLATE>":
        for _ in range(burst_len):
            out2 = continue_decode(model, past_kv)  # normal next-token step
            y = argmax(out2.logits[:, -1, :])
            print_prefix += tok.decode([y], skip_special_tokens=False)
            past_kv = out2.past_key_values

    # 6) optional commit: freeze & prune old audio tokens
    if should_commit():
        prune_kv_audio_to_last_seconds(past_kv, seconds=6)
```

---

## 6) Design Knobs (defaults that work well)

* **Window W:** 1.8 s (↑W = ↑stability, ↑latency)
* **Center C:** 0.6 s (trusted band; ↓C = ↑update rate)
* **Stride S:** 0.24 s (must satisfy **S ≤ C**)
* **Pack K:** 4 frames → 1 token (\~25 tok/s)
* **Patience K:** 2–3 ticks before first `<TRANSLATE>`
* **Prune horizon:** keep last 6–8 s of audio in KV after `<COMMIT>`
* **LAG token:** `<LAG=600ms>` to control the latency‐quality trade-off (optional)

---

## 7) Training (when we move beyond MVP)

* **Supervision:** build teacher actions from **wait-k / monotonic** alignment (ASR→MT pipeline or aligner).
* **Losses:**

  * CE for **text tokens** (when action is `<TRANSLATE>`),
  * CE for **action tokens** at action slots,
  * **Latency regularizer** (Average Lagging / DAL), optional and controlled by `<LAG=…>`.
* **Curriculum:** start with longer look-ahead (e.g., LAG=900 ms), then reduce toward 600 ms.
* **Augmentations:** noise / reverb / speed-perturb to improve robustness.

---

## 8) Evaluation

* **Translation quality:** BLEU, chrF, COMET on simultaneous references.
* **Latency:** Average Lagging (AL), Differentiable AL (DAL), Average Proportion (AP).
* **Stability:** re-edits per minute (should be ≈0 after commit), punctuation stability, clause boundary accuracy.
* **Barge-in behavior:** time-to-pause (ms), false pauses.

---

## 9) Failure Modes & Mitigations

* **Too early starts** → add **patience K**; stronger latency regularizer; increase W.
* **Hall noise / crosstalk** → barge-in **ON**; VAD gating before `<TRANSLATE>`.
* **Verb-final latency** → allow slightly larger LAG for those language pairs.
* **Memory growth** → emit **`<COMMIT>`** more often; prune audio KV to last 6–8 s.

---

## 10) Implementation Notes (what we actually changed)

* **Patched Qwen-Thinker class** to accept:

  * `external_audio_embeds: [B, Ta, d_ext]`, `external_audio_times: [B, Ta]`
  * Internal **projector MLP** (`d_ext → d_model`), **time Linear**(16→d\_model), **type Embedding**
  * **External path** concatenates `[audio_proj , text_embeds]` and calls the text model with **KV cache**.
* **No changes** to Whisper; it stays frozen.
* **No custom attention mask** needed (monotonicity achieved by incremental feeding).
* **Two control tokens** added to **tokenizer** (plus optional commit/EOS).

**Dtype/Device safety:** all Conaiki modules and inputs are cast to the **text embedding’s** device/dtype; run projector/time `Linear` in their own dtype and cast addends on sum.

---

## 11) Roadmap (MVP → Production)

* **MVP (what we have):**

  * Frozen Whisper; projector + time/type; two-token gate via tokens; streaming loop; optional `<COMMIT>`.
* **Next:**

  * Learned gate with supervision; `<COMMIT>` training; multi-pair fine-tuning; LAG control.
  * Light **LoRA** adapters on the text stack for S2TT specialization.
  * Optional TTS stage for full S2ST.

---

## 12) Quick FAQ

* **Why overlap-and-center?** Stability: each published slice had extra left/right context.
* **Do we lose data?** No—publish only the **new right-tail**; ensure **S ≤ C**; warm-start and right-flush.
* **Is it truly simultaneous?** Yes—audio tokens stream in; no future peeking; first words \~0.7 s.

---

### One-screen summary (hand-off)

* **Input:** live audio stream
* **Encode:** Whisper (W=1.8 s, C=0.6 s, S=0.24 s) → center tail frames
* **Tokens:** pack frames (≈25 tok/s) → projector + time/type
* **Backbone:** decoder-only LLM reads audio tokens; every \~50 ms chooses `<TRANSLATE>` or `<NOT_TRANSLATE>`; emits target tokens; `<COMMIT>` to freeze & prune
* **Latency:** first word \~0.65–0.85 s; steady lag \~0.2–0.4 s
* **Toggle:** barge-in ON/OFF

That’s Conaiki in a nutshell—single-architecture, real-time S2TT, clean interfaces, production-ready knobs.







# Conaiki: Current Model Overview (with `gate_head`) v_2

## 1) Modules (single model, no orchestration)

* **Audio encoder (Whisper-style / Qwen Omni audio tower)**

  * 128-mel → Conv1d (stride 1) → Conv1d (stride 2) → 32 Transformer encoder layers (d=1280) → `ln_post`.
  * We **tap pre-projection features** (`[B, T_enc, 1280]`) optionally after `avg_pooler` to halve frame rate.
* **Time embedding**

  * Fixed 16-d Fourier features of frame timestamps → `Linear(16→d_model)`; injected into audio tokens.
* **MLP Adapter (`conaiki_adapter`)**

  * `Linear(1280→d_model=3584) → SiLU → Linear(3584→3584)` to map encoder tokens into the LLM token space.
* **LLM backbone**

  * Qwen-style 28-layer decoder (hidden 3584) + `lm_head`. Initially **frozen**.
* **Gate head (`gate_head`)**

  * Tiny MLP on the **last hidden state** (`h_t = hidden_states[:, -1, :]`) each hop:

    * `Linear(d_model→256) → ReLU → Linear(256→3)` → logits for **{SILENCE, WAIT, TRANSLATE}**.
  * Optionally apply hysteresis (τ\_on / τ\_off) and patience K externally.
* **KV management**

  * Keep \~2–3 s of recent **audio tokens** in KV; keep **all target text** (or compress long histories); prune beyond horizon.

> Variant (drop-in): a **U-Net decider** on mel/post-conv features can replace `gate_head`. Our *current* path uses `gate_head` for simplicity and tight LLM coupling.

---

## 2) Streaming schedule (hop/window policy)

* **Hop** `S = 0.6 s` (model ticks every 0.6 s).
* **Window** `W = 1.8 s` centered on the hop; **trust center** `C = 0.6 s`.
* **Tail/Head** for boundary safety: keep **tail \~0.2–0.3 s** uncommitted; include **head \~0.1–0.2 s** when committing.

Why: overlap gives the encoder context; trusting only the center reduces boundary errors. Tail+head prevent cutting words in half.

---

## 3) Inference loop (what happens each 0.6 s)

1. **Buffer audio**; compute mel → pass through encoder up to our chosen tap (e.g., after `avg_pooler`) to get `[B, T_new, 1280]`.
2. **Adapt to LLM space:** `audio_tok = conaiki_adapter(encoder_tok) + time_embed`.
3. **Probe LLM:** concatenate `audio_tok` after prior audio tokens; run one forward step (can include a dummy probe token) to get `hidden_states` and logits; the **gate\_head** reads `hidden_states[:, -1, :]` and outputs action probabilities.
4. **Decision:**

   * **SILENCE:** do nothing (optionally decay/prune).
   * **WAIT:** buffer more audio; no text emitted.
   * **TRANSLATE:** define commit at the trusted center; **append only the new audio tokens** since the last commit (plus small head) to the LLM context; **generate a short burst** (e.g., 3–10 tokens or until punctuation).
5. **Housekeeping:**

   * **KV prune** old audio (keep \~2–3 s).
   * **Tail replay:** retain \~0.2–0.3 s at the right edge uncommitted so the next segment starts clean.
   * **De-dup** translated text if the next commit re-covers a couple of words (simple suffix–prefix check).

**Barge-in toggle:**

* **On:** allow commits even during overlapping speakers when acoustic evidence supports it.
* **Off:** raise τ\_on or add VAD gating so the model waits for clear talkspurts.

---

## 4) Training objectives (two losses)

Let hops be indexed by `t`, with gold actions `a*_t ∈ {SILENCE, WAIT, TRANSLATE}` and target text segments `{Y_i}` aligned to the **TRANSLATE** boundaries.

1. **Gate loss (per hop)**

   * `p_t = softmax(gate_head(h_t))`.
   * **Cross-entropy:** `L_gate = Σ_t CE(p_t, a*_t)`.
   * Optional:

     * **Entropy reg** (discourage indecision): `+ β · H(p_t)` with small `β` (e.g., 0.01–0.05).
     * **Class weighting** if SILENCE dominates.

2. **LM loss (only on commit hops)**

   * For each commit `i`, condition the LLM on **all committed audio tokens up to τ\_i**, then compute:
   * `L_lm = Σ_i Σ_k CE(softmax(lm_head(h_{i,k})), y_{i,k})`, masking non-commit hops completely.
   * Teacher forcing on target `Y_i`; no loss on WAIT/SILENCE steps.

**Which params get gradients?**

* Stage-1: **train `conaiki_adapter`, time layer, and `gate_head` only**; freeze encoder & LLM (+ `lm_head`).
* Stage-2 (optional): unfreeze **`lm_head`** and top-K LLM layers (or use **LoRA**) for better lexical choice.
* Stage-3 (optional): small-LR end-to-end finetune on your domain.

---

## 5) Data & labels (what a sample looks like)

* Input audio + source/target text with **word/phrase alignments** or at least **punctuation-anchored clause boundaries**.
* Build 0.6 s hops with a 1.8 s window. For each hop:

  * Gold **action** at the **center**:

    * **TRANSLATE** if the center aligns to an end-of-clause (pause/punct) since last commit.
    * **WAIT** inside speech regions.
    * **SILENCE** during VAD-negative stretches.
  * For TRANSLATE hops, attach the **target segment** `Y_i` (tokens) spanning `(last_commit, center]` (+ small head).

**Toy JSON (conceptual):**

```json
{
  "audio": "talk.wav",
  "src_lang": "RU",
  "tgt_lang": "EN",
  "hop_s": 0.6,
  "window_s": 1.8,
  "examples": [
    { "t_center": 0.6, "action": "WAIT" },
    { "t_center": 1.2, "action": "WAIT" },
    { "t_center": 1.8, "action": "TRANSLATE",
      "segment": {"t_start": 0.0, "t_end": 1.9},
      "target_tokens": ["I", "arrived", "yesterday", "."]
    },
    { "t_center": 2.4, "action": "WAIT" },
    { "t_center": 3.0, "action": "TRANSLATE",
      "segment": {"t_start": 2.0, "t_end": 3.1},
      "target_tokens": ["It", "was", "great", "."]
    }
  ]
}
```

---

## 6) Why a separate `gate_head` (vs predicting action tokens)

**Pros**

* Doesn’t compete with the language vocabulary; avoids spurious action tokens in text; **cleaner gradients** for decisions.
* Allows **independent thresholds/hysteresis** and **entropy regularization**.
* You can **freeze the LLM** and still train a strong decider.

**Cons**

* The gate doesn’t directly benefit from LM token probabilities; slight risk of premature commits (mitigated by tail & peek-veto if desired).

---

## 7) Failure modes & mitigations

* **Boundary cut (start of next word clipped):** keep **tail replay** and **dedup** on the target side.
* **Over-waiting (latency spike):** cap patience; force commit if no TRANSLATE in N hops.
* **Rapid speakers / fillers:** raise τ\_on; require K consecutive ready signals.
* **Long sessions (10+ minutes):** KV pruning of old audio; periodic **state compaction**; chunked decoding.

---

## 8) What another model should search for (keywords)

* “Simultaneous speech translation gating”, “wait-k vs adaptive policy”, “commitment policy ST”,
* “monotonic attention streaming”, “MAESTRO simultaneous translation”, “prefix-to-prefix translation”,
* “end-to-end S2TT streaming Whisper adapter”, “boundary detection for ST”,
* “average lagging (AL) metric”, “latency-quality tradeoff ST”, “gate head MLP simultaneous translation”.

---

## 9) Mini example (timeline)

> Source: «Я приехал вчера.»
> Hops at 0.6 s:

* 0.6s: encoder tokens → **gate=WAIT** → no text.
* 1.2s: **gate=WAIT** → no text.
* 1.8s: pause detected → **gate=TRANSLATE** → adapter tokens for (0.0–1.9s) → LLM emits: “I arrived yesterday.”
  KV prunes old audio; keep \~0.2 s tail for safety.
