import logging
from typing import List, Union

import datasets
from datasets.naming import INVALID_WINDOWS_CHARACTERS_IN_PATH
from composer.utils import dist
from transformers.tokenization_utils_fast import PreTrainedTokenizer, PreTrainedTokenizerFast


# this is crazy
def _new_fingerprint_validate(fingerprint: str, max_length=64):
    """
    Make sure the fingerprint is a non-empty string that is not longer that max_length=64 by default,
    so that the fingerprint can be used to name cache files without issues.
    """
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Invalid fingerprint '{fingerprint}': it should be a non-empty string.")
    for invalid_char in INVALID_WINDOWS_CHARACTERS_IN_PATH:
        if invalid_char in fingerprint:
            # raise ValueError(
            print(
                f"Invalid fingerprint. Bad characters from black list '{INVALID_WINDOWS_CHARACTERS_IN_PATH}' found in '{fingerprint}'. "
                f"They could create issues when creating cache files."
            )
    if len(fingerprint) > max_length:
        raise ValueError(
            f"Invalid fingerprint. Maximum lenth is {max_length} but '{fingerprint}' has length {len(fingerprint)}."
            "It could create issues when creating cache files."
        )
# work around ValueError: Invalid fingerprint. Bad characters from black list '<>:/\|?*' found in 'ledgar-microsoft/deberta-v2-xlarge-mnli-tokenization-validation'.
# They could create issues when creating cache files.
datasets.fingerprint.validate_fingerprint = _new_fingerprint_validate


single_label = (None, "text", "label")  # none, str, int
multi_label = (None, "text", "labels")  # none, str, List[int]
multiple_choice_qa = ("context", "endings", "label")  # str, str, int


task_example_types = {
    'case_hold': multiple_choice_qa,
    'ecthr_a': multi_label,
    'ecthr_b': multi_label,
    'eurlex': multi_label,
    'ledgar': single_label,
    'scotus': single_label,
    'unfair_tos': multi_label,
}

log = logging.getLogger(__name__)


def format_casehold_input(context: str, endings: List[str]) -> str:
    t = "Context: "
    t += context
    t += "\n\n"
    t += "Which of the following follow from the context?\n"
    for i, end in enumerate(endings):
        t += f"{i}: {end}\n"
    return t


def format_casehold_batched(context: List[str], endings: List[List[str]]):
    return [format_casehold_input(c, e) for c, e in zip(context, endings)]


# this assumes we want to do classification, not treat as seq2seq
# todo add seq2seq support
def create_lexglue_dataset(
    task: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    split: str,
    max_seq_length: int = 512,
    max_retries: int = 10,
    num_workers: int = 0,
):

    if task not in task_example_types:
        raise ValueError(f'task ({task}) must be one of {task_example_types.keys()}')

    if (max_seq_length % 8) != 0:
        log.warning('For performance, a max_seq_length as a multiple of 8 is recommended.')

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)
    dataset = datasets.load_dataset(
        'lex_glue',
        task,
        split=split,
        download_config=download_config,
    )

    log.info(f'Starting tokenization by preprocessing over {num_workers} threads!')
    example_type = task_example_types[task]

    def tokenize_function(inp):
        # truncates sentences to max_length or pads them to max_length

        if example_type == multiple_choice_qa:
            context = inp[example_type[0]]
            endings = inp[example_type[1]]
            if isinstance(context, list):
                text = format_casehold_batched(context, endings)
            else:
                text = format_casehold_input(context, endings)
        else:
            text = inp[example_type[1]]

        return tokenizer(
            text=text,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
        )

    columns_to_remove = ['text'] if example_type != multiple_choice_qa else ["context", "endings"]

    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        new_fingerprint=f'{task}-{tokenizer.name_or_path}-tokenization-{split}',
        load_from_cache_file=True,
    )
    return dataset
