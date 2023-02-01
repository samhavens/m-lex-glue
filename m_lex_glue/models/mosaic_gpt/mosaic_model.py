import os
from collections import OrderedDict
from urllib.parse import urlparse

import einops
import torch
from composer.metrics.nlp import LanguageCrossEntropy
from composer.utils.file_helpers import get_file
from composer.utils.object_store import S3ObjectStore
from omegaconf import DictConfig
from transformers import AutoTokenizer

from m_lex_glue.metrics.modified_metrics import RougeWithDetokenizer
from m_lex_glue.models.mosaic_gpt.modeling_mosaic_gpt import ComposerMosaicGPT


def get_checkpoint_name_from_path(path: str) -> str:
    """To go from checkpoint name to path, replace | with /"""
    return path.lstrip('/').replace('/', '|')


def download_starting_checkpoint(starting_checkpoint_load_path: str,
                                 local_pretrain_checkpoints_folder: str) -> str:
    """Downloads the pretrained checkpoints to start from.

    Currently only supports S3 and URLs
    """
    load_object_store = None
    parsed_path = urlparse(starting_checkpoint_load_path)
    if parsed_path.scheme == 's3':
        load_object_store = S3ObjectStore(bucket=parsed_path.netloc)

    download_path = parsed_path.path if parsed_path.scheme == 's3' else starting_checkpoint_load_path
    os.makedirs(local_pretrain_checkpoints_folder, exist_ok=True)
    local_path = os.path.join(local_pretrain_checkpoints_folder,
                              get_checkpoint_name_from_path(parsed_path.path))
    if not os.path.exists(local_path):
        get_file(destination=local_path,
                 path=download_path.lstrip('/'),
                 object_store=load_object_store,
                 progress_bar=True)

    return local_path



def get_mosaic_model_hf_wrap(cfg: DictConfig) -> ComposerMosaicGPT:
    """This returns a Composer Model from a checkpoint or randomly initialized

    ...which wraps a HuggingFace model, which wraps MosaicGPT

    Currently only defined for summarization; need to write task-specific heads (or convert tasks to seq2seq)

    If you specify attn_impl: torch but the model was trained with triton attention, this will
    convert state_dict key names as needed
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        padding_side="left",
        use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.max_seq_length

    train_metrics = [LanguageCrossEntropy(cfg.model.vocab_size)]
    eval_metrics = [
        LanguageCrossEntropy(cfg.model.vocab_size),
        RougeWithDetokenizer(detokenizer=tokenizer),
    ]

    cm = ComposerMosaicGPT(cfg, tokenizer=tokenizer, train_metrics=train_metrics, eval_metrics=eval_metrics)
    cm.tokenizer = tokenizer

    checkpoint_path = cfg.get('starting_checkpoint_load_path', None)
    local_checkpoint_path = cfg.get('local_pretrain_checkpoints_folder', None)

    if checkpoint_path is None and local_checkpoint_path is None:
        print('Specified MosaicGPT model without checkpoint path, '
              'using randomly initialized weights.')
        print('If this is not what you expected, please set '
              '`starting_checkpoint_load_path` and `local_pretrain_checkpoints_folder`')
        return cm

    # load checkpoint, converting key names if needed
    checkpoint_path = download_starting_checkpoint(checkpoint_path, local_checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    try:
        # will work if the pretrained model and current config have the same attn_impl
        cm.model.load_state_dict(checkpoint['state']['model'])
    except RuntimeError as e:
        # if we are loading Triton trained weights into Torch, need to manipulate a bit
        if (
            'Missing key(s) in state_dict: "model.transformer.blocks.0.causal_attn.mhsa.in_proj_weight"' in str(e) 
            and 'Unexpected key(s) in state_dict: "model.transformer.blocks.0.causal_attn.mhsa.Wqkv.weight"' in str(e)
        ):
            # converting from triton to torch
            triton_2_torch = {
                "Wqkv.weight": "in_proj_weight",
                "Wqkv.bias": "in_proj_bias"
            }
            t2t_keys = triton_2_torch.keys()
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state']['model'].items():
                if any(ttk in k for ttk in t2t_keys):
                    for ttk, ttv in triton_2_torch.items():
                        if ttk in k:
                            new_k = k.replace(ttk, ttv)
                            new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            # now need to reshape attn_mask from [1, 12, 1, 8192] to [12, 8192, 8192]
            torch_mask = einops.rearrange(new_state_dict['model.attn_mask'], '1 h 1 s -> h s')
            torch_mask = einops.repeat(torch_mask, 'h s -> h s_again s', s_again=8192)
            new_state_dict['model.attn_mask'] = torch_mask
            cm.model.load_state_dict(new_state_dict)

    return cm
