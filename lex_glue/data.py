import logging
from typing import List, Union

import datasets
from composer.utils import dist
from transformers.tokenization_utils_fast import PreTrainedTokenizer, PreTrainedTokenizerFast


_single_label = (None, "text", "label")  # none, str, int
_multi_label = (None, "text", "labels")  # none, str, List[int]
_multiple_choice_qa = ("context", "endings", "label")  # str, str, int


_task_example_types = {
    'case_hold': _multiple_choice_qa,
    'ecthr_a': _multi_label,
    'ecthr_b': _multi_label,
    'eurlex': _multi_label,
    'ledgar': _single_label,
    'scotus': _single_label,
    'unfair_tos': _multi_label,
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

    if task not in _task_example_types:
        raise ValueError(f'task ({task}) must be one of {_task_example_types.keys()}')

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
    example_type = _task_example_types[task]

    def tokenize_function(inp):
        # truncates sentences to max_length or pads them to max_length

        if example_type == _multiple_choice_qa:
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

    columns_to_remove = ['text'] if example_type != _multiple_choice_qa else ["context", "endings"]

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
