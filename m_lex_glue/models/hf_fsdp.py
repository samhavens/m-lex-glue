# Adapted from https://github.com/mosaicml/benchmarks/blob/main/llm/src/hf_causal_lm.py
from typing import Tuple
import transformers
from torch import nn
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM, GPTJModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM, GPTNeoModel
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel, GPTNeoXForCausalLM

from m_lex_glue.utils import findattr


_SUPPORTED_HF_MODELS = (
    GPT2Model,
    GPTJModel,
    GPTNeoModel,
    GPTNeoXModel,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoForCausalLM,
    GPTNeoXForCausalLM,
)


# PLAN: use this method and a few others to replace the lookup
# "does this model work" and "what blocks do we wrap"
def hf_get_causal_base_model(model: transformers.AutoModelForCausalLM) -> nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ("transformer", "model.decoder", "gpt_neox")
    return findattr(model, decoder_attrs)


def hf_get_lm_head(model: transformers.AutoModelForCausalLM) -> nn.Module:
    """Returns the language modeling (lm) head of the specified HuggingFace
    transformers model.
    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


def hf_get_causal_hidden_layers(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
    """
    hidden_layers_attrs = (
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_tied_embedding_weights(model: nn.Module) -> nn.Module:
    """Returns the embeddings which are weight tied layers of the specified model.
    NOTE: Different model configurations have different embedding attribute names.
        - wte: (GPT2LMHeadModel, GPTJForCausalLM, GPTNeoForCausalLM)
        - word_embeddings: (BloomForCausalLM)
        - embed_tokens: (OPTForCausalLM)
        - GPT NeoX doesn't weight tie
    """
    tied_embedding_attrs = (
        "wte",
        "word_embeddings",
        "embed_tokens",
    )
    return findattr(model, tied_embedding_attrs)


# REWRITE THIS TO NOT USE LOOKUP!
# Or at least add BLOOM and OPT
# also... T5??
def is_fsdp_able(model) -> bool:
    causal_base_model = hf_get_causal_base_model(model)
    return isinstance(model, _SUPPORTED_HF_MODELS) or isinstance(causal_base_model, _SUPPORTED_HF_MODELS)


def prepare_hf_model_for_fsdp(model):
    """
    Wrap any model for FSDP which follows one of the 3 existing conventions on HF for decoder only LLMs
    """
    assert is_fsdp_able(model), f"FSDP support not available for this model"
    causal_base_model = hf_get_causal_base_model(model)
    model_block = hf_get_causal_hidden_layers(model)[0]
    lm_head = hf_get_causal_hidden_layers(model)
    tied_embeddings = hf_get_tied_embedding_weights(causal_base_model)
    # When using the HF Causal LM models,
    # the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This is a hurdle for FSDP because they need to be in the same FSDP block
    # These lines ensures that both modules stay together in the top-most block
    if tied_embeddings is not None:
        causal_base_model._fsdp_wrap = False
        tied_embeddings._fsdp_wrap = False
        lm_head._fsdp_wrap = False

    # FSDP Wrap and Activation Checkpoint every GPT2Block
    model.fsdp_wrap_fn = lambda module: isinstance(module, model_block)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, model_block)
