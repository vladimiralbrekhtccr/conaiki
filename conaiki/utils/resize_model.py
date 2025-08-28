# import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] ='1'
# from transformers import Qwen2_5OmniProcessor
# from transformers import Qwen2_5OmniThinkerForConditionalGeneration

# # PATH = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/conaiki/qwen_omni_finetuned/checkpoint-135"
# PATH = "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/models/Qwen2.5-Omni-7B"
# model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
#     PATH, torch_dtype=torch.bfloat16, device_map=None
# ).to("cuda")

# processor = Qwen2_5OmniProcessor.from_pretrained(PATH)
# n_gate = sum(p.numel() for p in model.conaiki_gate.parameters() if p.requires_grad)
# print(f"Trainable gate params: {n_gate}")

# # First test: standard generation
# conversation = [
#     {"role": "user", "content": [{"type": "text", "text": "Who rescues whom at the abandoned observatory in episode 3?"}]},
# ]

# # conversation = [
# #     {"role": "user", "content": [{"type": "audio", "path": "/raid/vladimir_albrekht/projects/conaiki/qwen_omni/conaiki/data/audio/audios/rustem_1.wav"}]},
# # ]

# inputs = processor.apply_chat_template(
#     [conversation],
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
#     padding=True
# ).to(model.device)

# # Standard generation (only token IDs)
# text_ids = model.generate(**inputs)
# text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print("Standard generation:", text[0])

def resize_token_embeds(model, processor):
    """
    you can do it
    """
    SPECIALS = ["<WAIT>", "<TRANSLATE>"]
    tok = processor.tokenizer
    old_tok_n = len(tok)
    added = tok.add_special_tokens({"additional_special_tokens": SPECIALS})
    new_tok_n = len(tok)

    emb = model.model.embed_tokens 
    old_emb_n, d = emb.weight.shape

    print(f"tokenizer: {old_tok_n} -> {new_tok_n} (added={added}) | emb rows={old_emb_n}")

    def init_rows(weight, rows, pad_idx=None):
        if not rows:
            return
        with torch.no_grad():
            for r in rows:
                if r is None or r < 0 or r >= weight.shape[0]:
                    continue
                torch.nn.init.normal_(weight[r], std=0.02)
            # TODO: doing this will just ruin the model performance and make it dumpest model in the world lol
            # probably it's because from {151665 and <} have same value and by coping embeddings it's from embed_tokens we are creating something unusual for the model 
            # TODO: come up with some solution
            # for r in rows:
            #     model.lm_head.weight[r].copy_(model.get_input_embeddings().weight[r])
            
    
    # Our Case B: tokenizer <= embeddings -> DO NOT SHRINK; just init the newly allocated token rows
    # The new special tokens took IDs at the end of the tokenizer space
    new_ids = tok.convert_tokens_to_ids(SPECIALS)
    # Reinit only those rows (they already exist in the embedding matrix)
    init_rows(model.get_input_embeddings().weight, new_ids, pad_idx=getattr(model.config, "pad_token_id", None))
    # keep config.vocab_size = max(current emb rows, tokenizer size); do NOT reduce it

    print("final shapes:",
        tuple(model.get_input_embeddings().weight.shape),
        "lm_head:" if hasattr(model, "lm_head") else "",
        (tuple(model.lm_head.weight.shape) if hasattr(model, "lm_head") else "N/A"))
    print("special IDs:", tok.convert_tokens_to_ids(SPECIALS))
    model.config.vocab_size = model.config.text_config.vocab_size
resize_token_embeds(model, processor)