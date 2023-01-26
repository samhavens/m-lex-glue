# Adapted from https://github.com/mosaicml/benchmarks/blob/main/llm/src/hf_causal_lm.py
# some helper functions from https://github.com/CarperAI/trlx/blob/main/trlx/utils/modeling.py

from typing import Tuple
import transformers
from torch import nn
from transformers.models.bloom.modeling_bloom import BloomForCausalLM, BloomModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model
from transformers.models.gptj.modeling_gptj import GPTJForCausalLM, GPTJModel
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM, GPTNeoModel
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXModel
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel

from m_lex_glue.utils import findattr


_DEFINITELY_SUPPORTED_HF_MODELS = (
    BloomForCausalLM,
    BloomModel,
    GPT2LMHeadModel,
    GPT2Model,
    GPTJForCausalLM,
    GPTJModel,
    GPTNeoForCausalLM,
    GPTNeoModel,
    GPTNeoXForCausalLM,
    GPTNeoXModel,
    OPTForCausalLM,
    OPTModel,
)


def hf_get_base_model(model: transformers.AutoModelForCausalLM) -> nn.Module:
    """Returns the backbone of the specified HuggingFace transformer model
    (i.e. without the task-specific head)
    
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


def is_fsdp_able(model) -> bool:
    """Check if this is a known FSDP-able model"""
    base_model = hf_get_base_model(model)
    return (
        isinstance(model, _DEFINITELY_SUPPORTED_HF_MODELS)
        or isinstance(base_model, _DEFINITELY_SUPPORTED_HF_MODELS)
    )


def prepare_hf_model_for_fsdp(model):
    """
    Wrap any model for FSDP which follows one of the 3 existing conventions on HF for decoder only LLMs
    """
    if not is_fsdp_able(model):
        print(f"attempting to FSDP wrap {model.name_or_path}, but this model has not been tested with FSDP")

    causal_base_model = hf_get_base_model(model)
    model_block = hf_get_causal_hidden_layers(model)[0]
    block_type = type(model_block)
    lm_head = hf_get_causal_hidden_layers(model)
    tied_embeddings = hf_get_tied_embedding_weights(causal_base_model)

    # When using HF LM models, the weights of the self.lm_head and self.transformer.wte are tied.
    # This tying occurs inside the `self.post_init()` function.
    # This means the lm_head and wte need to be in the same FSDP block. By setting ._fsdp_wrap = False,
    # we do not wrap those modules, so their parameters get flattened as part of the parent module,
    # which makes sure they don't get split apart
    if model.config.tie_word_embeddings and lm_head is not None:
        causal_base_model._fsdp_wrap = False
        tied_embeddings._fsdp_wrap = False
        lm_head._fsdp_wrap = False

    # FSDP Wrap and Activation Checkpoint every block (e.g. GPT2Block, GPTNeoBlock, etc)
    model.fsdp_wrap_fn = lambda module: isinstance(module, block_type)
    model.activation_checkpointing_fn = lambda module: isinstance(
        module, block_type)
