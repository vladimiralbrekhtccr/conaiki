import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

PATH = "/scratch/vladimir_albrekht/projects/18_august_25_conaiki/qwen_omni/models/Qwen2.5-Omni-7B"

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    PATH, torch_dtype="auto", device_map="auto"
)
processor = Qwen2_5OmniProcessor.from_pretrained(PATH)

conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are Qwen..."}]},
    {"role": "user",   "content": [{"type": "audio", "path": "/scratch/vladimir_albrekht/projects/18_august_25_conaiki/qwen_omni/inference_test/rustem_1.wav"}]},
]

# Build inputs directly from the chat—no custom utils needed
inputs = processor.apply_chat_template(
    [conversation],                            # batch of 1
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    padding=True
).to(model.device)

# Text-only output (no audio is even possible with Thinker-only)
text_ids = model.generate(**inputs)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text[0])
