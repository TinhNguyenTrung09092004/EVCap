# EVCap with a GPT-2 decoder

This branch replaces EVCap's Vicuna-13B decoder with GPT-2 Small. Everything
else - the external visual-name memory, retrieval and fusion - is the original
design. See the `main` branch for the unmodified code, and
[kaggle/README.md](kaggle/README.md) for how to run it.

## Language model

`gpt2` (GPT-2 Small, 124M, 12 layers, hidden size 768), loaded from the Hugging Face hub with `AutoModelForCausalLM` in fp32. The `main` branch hardcoded `vicuna-13b-v1.3` in fp16.

## Trained vs frozen

Trained (0.621312M parameters):

| Module | Shape | Note |
| --- | --- | --- |
| `query_tokens` | 32 x 768 | image Q-Former queries |
| `query_tokens_txt` | 8 x 768 | text Q-Former queries |
| `llama_proj` | 768 -> 768 | Q-Former output -> decoder embedding space |

Frozen: EVA-CLIP ViT-G + `ln_vision` (fp16), both Q-Formers (BERT weights from BLIP-2 `blip2_pretrained_flant5xxl`), and all of GPT-2. Same trainable set as the original EVCap; only `llama_proj` shrinks, since its output width follows the decoder hidden size (768 instead of Vicuna's 5120 - this is why the original paper reports 3.97M trainable parameters and this branch ~0.62M).

## Changes made to swap Vicuna-13B -> GPT-2

- **Loading**: `LlamaForCausalLM` / `LlamaTokenizer` -> `AutoModelForCausalLM` /
  `AutoTokenizer`; the vendored `models/modeling_llama.py` is deleted. GPT-2 has
  no `pad_token` and no `bos_token_id`, so both fall back to `eos_token`
  ([evcap.py:105-109](models/evcap.py#L105-L109)).
- **Embedding lookup**: every `llama_model.model.embed_tokens(...)` call becomes
  `EVCap.embed_tokens`, a wrapper over `get_input_embeddings()` that works for
  GPT-2 (`transformer.wte`) and LLaMA alike
  ([evcap.py:156-159](models/evcap.py#L156-L159)); `search.word_embed` does the
  same.
- **Precision**: the decoder runs in fp32 under AMP; `model.dtype` is replaced by
  `_lm_dtype(model)` in [search.py](search.py#L307-L309). The ViT stays fp16.
- **transformers v5**: slow tokenizer classes and Q-Former helpers removed
  upstream are vendored in [models/hf_compat.py](models/hf_compat.py).

The prompt is unchanged (`prompts/prompt_evcap.txt` with the
`###Human: ... ###Assistant:` template).
