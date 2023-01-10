import os
from urllib.parse import urlparse

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
    """This returns a Composer Model, which wraps a HuggingFace model, which wraps MosaicGPT

    Currently only defined for summarization; need to write task-specific heads (or convert tasks to seq2seq)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name,
        padding_side="left",
        use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = cfg.max_seq_len

    train_metrics = [LanguageCrossEntropy(cfg.model.vocab_size)]
    eval_metrics = [
        LanguageCrossEntropy(cfg.model.vocab_size),
        RougeWithDetokenizer(detokenizer=tokenizer),
    ]

    cm = ComposerMosaicGPT(cfg, tokenizer=tokenizer, train_metrics=train_metrics, eval_metrics=eval_metrics)
    cm.tokenizer = tokenizer

    checkpoint_path = download_starting_checkpoint(cfg.starting_checkpoint_load_path, cfg.local_pretrain_checkpoints_folder)
    checkpoint = torch.load(checkpoint_path)
    cm.model.load_state_dict(checkpoint['state']['model'])

    return cm
