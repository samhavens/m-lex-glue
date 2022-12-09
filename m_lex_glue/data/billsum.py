import functools
import logging
from itertools import chain
from typing import List, Tuple, Union, cast

import datasets
import numpy as np
import transformers
from composer.core import Evaluator
from composer.core.types import Dataset
from composer.utils import dist
from transformers.tokenization_utils_fast import (PreTrainedTokenizer,
                                                  PreTrainedTokenizerFast)
from torch.utils.data import DataLoader

from m_lex_glue.data.tasks import summarization
from m_lex_glue.metrics.modified_metrics import RougeWithDetokenizer

log = logging.getLogger(__name__)


def get_prefix_suffix(tokenizer) -> Tuple[str, str]:
    """Task prefixes / suffixes are in the papers for (m)t5, but a good summary is
    https://huggingface.co/course/chapter7/5?fw=pt#models-for-text-summarization"""
    if "mt5" in tokenizer.name_or_path:
        model_style = "mt5"
    elif any(clue in tokenizer.name_or_path for clue in ["t5", "ul2", "t0", "bart"]):
        model_style = "t5"
    elif "gpt" in tokenizer.name_or_path:
        model_style = "gpt"

    if model_style == "t5":
        prefix = "summarize: "
        suffix = "\n"
    elif model_style == "gpt":
        # prefix = ""
        # suffix = " TL;DR "
        # suffix is a pain to implement with truncation and padding, use t5-style for now
        prefix = "summarize: "
        suffix = "\n"
    else:
        prefix, suffix = "", ""
    return prefix, suffix


def get_summarization_preprocessor(tokenizer, max_seq_length):
    """BillSum https://arxiv.org/pdf/1910.00523.pdf
    The BillSum corpus focuses on mid-length legislation from 5,000 to 20,000 character in length
    Summaries are ~200 to 5,000 characters"""
    text_column = summarization[1]
    summary_column = summarization[2]

    prefix, suffix = get_prefix_suffix(tokenizer)

    def preprocess_summary_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp + suffix for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding="max_length", truncation=True)

        labels = tokenizer(targets, max_length=max_seq_length, padding="max_length", truncation=True)

        # replace all tokenizer.pad_token_id in the labels by -100 to
        # ignore padding in the loss
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_summary_function


def get_clm_preprocessor(tokenizer, max_seq_length):
    text_column = summarization[1]
    summary_column = summarization[2]

    prefix, suffix = get_prefix_suffix(tokenizer)

    def preprocess_clm_function(examples):
        # remove pairs where at least one record is None
        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        inputs = [prefix + inp + suffix for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding="max_length", truncation=True)

        # @TODO make this configurable
        # WHY DOES THIS CAUSE ROUGE TO FAIL if we use model max length??
        # might be a CPU-only error
        block_size = max_seq_length # tokenizer.model_max_length # 512  # 2048
        concatenated_examples = {k: list(chain(*model_inputs[k])) for k in model_inputs.keys()}
        total_length = len(concatenated_examples[list(model_inputs.keys())[0]])
        # We drop the small remainder, we could add padding
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        return result

    return preprocess_clm_function


def create_clm_dataset(
    task: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    split: str,
    max_seq_length: int = 512,
    max_retries: int = 10,
    num_workers: int = 0,
):
    if (max_seq_length % 8) != 0:
        log.warning('For performance, a max_seq_length as a multiple of 8 is recommended.')

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        task,
        split=split if split == "train" else "test",  # eval split is called test; normalize
        download_config=download_config,
    )

    columns_to_remove = list(summarization)

    log.info(f'Starting tokenization by preprocessing over {num_workers} threads!')

    assert isinstance(dataset, datasets.Dataset)

    pre_process_fn = get_clm_preprocessor(tokenizer, max_seq_length)
    dataset = dataset.map(
        pre_process_fn,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        load_from_cache_file=True,
    )
    dataset.set_format(type="torch")
    return dataset


def create_summarization_dataset(
    task: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    split: str,
    max_seq_length: int = 512,
    max_retries: int = 10,
    num_workers: int = 0,
):
    """Needs to be cleaned up, is the same fn as create_clm_dataset but for the preprocessor"""
    if (max_seq_length % 8) != 0:
        log.warning('For performance, a max_seq_length as a multiple of 8 is recommended.')

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        task,
        split=split if split == "train" else "test",  # eval split is called test; normalize
        download_config=download_config,
    )
    columns_to_remove = list(summarization)

    log.info(f'Starting tokenization by preprocessing over {num_workers} threads!')

    assert isinstance(dataset, datasets.Dataset)

    pre_process_fn = get_summarization_preprocessor(tokenizer, max_seq_length)
    dataset = dataset.map(
        pre_process_fn,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        load_from_cache_file=False,
    )
    dataset.set_format(type="torch")
    return dataset


def labeled_collate(data, mode: str, collator=transformers.default_data_collator):
    """This adds an extra `eval_mode` key to batches, so in model.eval_forward we can use that mode to decide
    whether to do one forward pass (CLM) or greedy decoding (seq2seq)"""
    collate_output = collator(data)
    batch_size = collate_output['labels'].shape[0]
    eval_mode = [mode] * batch_size # must be same size or else data_spec raises an error
    collate_output['eval_mode'] = np.array(eval_mode)  # composer checks .shape so use array bc string payload
    return collate_output


def build_summarization_dataloaders(
    clm_dataset,
    summarization_dataset,
    device_batch_size,
    drop_last=False,
    **kwargs
) -> List[Evaluator]:
    """Build the evaluation sets for summarization.

    Composer does not having an easy way to know when to generate or do one forward pass,
    so we attach information to the batch about what dataset it is so the model can use that to
    decide how to run the eval forward pass"""
    clm_dataset = cast(Dataset, clm_dataset)
    summarization_dataset = cast(Dataset, summarization_dataset)

    print("\nSummarization Dataloader Shapes:")
    print("CLM dataloader", {k: v.shape for k, v in clm_dataset[0].items()})
    print("Seq2seq dataloader", {k: v.shape for k, v in summarization_dataset[1].items()})
    print("\n")

    clm_dl =  DataLoader(
        dataset=clm_dataset,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(clm_dataset, drop_last=drop_last, shuffle=False),
        collate_fn=functools.partial(labeled_collate, mode='clm'),
        **kwargs,
    )

    sum_dl =  DataLoader(
        dataset=clm_dataset,
        batch_size=device_batch_size,
        sampler=dist.get_sampler(clm_dataset, drop_last=drop_last, shuffle=False),
        collate_fn=functools.partial(labeled_collate, mode='seq2seq'),
        **kwargs,
    )

    # We actually return Evaluators so the loader is tied to a metric
    # we don't just use the string "RougeWithDetokenizer" because we want there to be an ImportError
    # if the name of the metric changes
    # NOTE be very careful that metric_names is a list... it won't fail if you put in a string, but it won't work either
    return [
        Evaluator(label="LanguageCrossEntropy", dataloader=clm_dl, metric_names=["LanguageCrossEntropy"]),
        Evaluator(label=RougeWithDetokenizer.__name__, dataloader=sum_dl, metric_names=[RougeWithDetokenizer.__name__]),
    ]