# Adapted from https://github.com/mosaicml/benchmarks/blob/main/llm/src/hf_causal_lm.py

from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2LMHeadModel, GPT2Model
from transformers.models.gptj.modeling_gptj import (GPTJBlock,
                                                    GPTJForCausalLM,
                                                    GPTJModel)
from transformers.models.gpt_neo.modeling_gpt_neo import (GPTNeoBlock,
                                                          GPTNeoForCausalLM,
                                                          GPTNeoModel)
from transformers.models.gpt_neox.modeling_gpt_neox import (GPTNeoXModel,
                                                            GPTNeoXForCausalLM,
                                                            GPTNeoXLayer)


_SUPPORTED_HF_MODELS = (
    GPT2Model,
    GPTJModel,
    GPTNeoModel,
    GPTNeoXModel,
)

_WEIGHT_TIED_HF_MODELS = (
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
)


_HF_MODEL_BLOCKS = (
    GPT2Block,
    GPTJBlock,
    GPTNeoBlock,
    GPTNeoXLayer,
)


def is_fsdp_able(model) -> bool:
    return isinstance(model, _SUPPORTED_HF_MODELS)


def prepare_hf_model_for_fsdp(model):
    assert is_fsdp_able(model), f"FSDP support only available for {list(_SUPPORTED_HF_MODELS)}"
    # When using the HF Causal LM models,
    # the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block
    if isinstance(model, _WEIGHT_TIED_HF_MODELS):
        model.transformer._fsdp_wrap = False
        model.transformer.wte._fsdp_wrap = False
        model.lm_head._fsdp_wrap = False

    # FSDP Wrap and Activation Checkpoint every GPT2Block
    model.fsdp_wrap_fn = lambda module: isinstance(module, _HF_MODEL_BLOCKS)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, _HF_MODEL_BLOCKS)