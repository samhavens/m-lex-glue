import logging
from typing import List, Union

import datasets
import torch
from composer.utils import dist
from transformers.tokenization_utils_fast import PreTrainedTokenizer, PreTrainedTokenizerFast

from m_lex_glue.casehold_helpers import MultipleChoiceDataset, format_casehold_batched, format_casehold_input
from m_lex_glue.labels import TASK_NAME_TO_LABELS


multi_class = (None, "text", "label")  # none, str, int
multi_label = (None, "text", "labels")  # none, str, List[int]
multiple_choice_qa = ("context", "endings", "label")  # str, str, int


task_example_types = {
    'case_hold': multiple_choice_qa,
    'ecthr_a': multi_label,
    'ecthr_b': multi_label,
    'eurlex': multi_label,
    'ledgar': multi_class,
    'scotus': multi_class,
    'unfair_tos': multi_label,
}

log = logging.getLogger(__name__)


def get_preprocessor(task, example_type, tokenizer, max_seq_length):

    def tokenize_function(inp):
        if example_type == multiple_choice_qa:
            # the only MCQA is Casehold, and we already returned that dataloader
            # we would get to this branch once we do seq2seq eval
            context = inp[example_type[0]]
            endings = inp[example_type[1]]
            if isinstance(context, list):
                text = format_casehold_batched(context, endings)
            else:
                text = format_casehold_input(context, endings)
        else:
            text = inp[example_type[1]]

        batch = tokenizer(
            text=text,
            padding='max_length',
            max_length=max_seq_length,
            truncation=True,
        )

        if example_type == multi_label:
            label_list = [i for i, _ in enumerate(TASK_NAME_TO_LABELS[task])]
            # id2label = {idx:label for idx, label in enumerate(TASK_NAME_TO_LABELS[task])}
            # label2id = {label:idx for idx, label in enumerate(TASK_NAME_TO_LABELS[task])}
            # shape (batch_size, num_labels)
            labels_matrix = torch.zeros((len(text), len(label_list)), dtype=torch.float)
            for idx, labels in enumerate(inp['targets']):
                labels_matrix[idx] = torch.tensor(
                    [1 if label in labels else 0 for label in label_list],
                    dtype=torch.float
                )

            batch["labels"] = labels_matrix
        return batch

    return tokenize_function


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

    if task == "case_hold":
        # For now we treat this like sequence classification, using
        # AutoModelForMultipleChoice models
        return MultipleChoiceDataset(
            tokenizer=tokenizer,
            task=task,
            max_seq_length=max_seq_length,
            split=split,
        )

    dataset = datasets.load_dataset(
        'lex_glue',
        task,
        split=split,
        download_config=download_config,
    )

    log.info(f'Starting tokenization by preprocessing over {num_workers} threads!')
    example_type = task_example_types[task]

    columns_to_remove = ['text'] if example_type != multiple_choice_qa else ["context", "endings"]

    assert isinstance(dataset, datasets.Dataset)

    pre_process_fn = get_preprocessor(task, example_type, tokenizer, max_seq_length)

    if example_type == multi_label:
        # the dataset which we download has stored information 
        dataset = dataset.rename_column("labels", "targets")  # it forces `label` column to be CLassList
        columns_to_remove.append("targets")

    dataset = dataset.map(
        pre_process_fn,
        batched=True,
        num_proc=None if num_workers == 0 else num_workers,
        batch_size=1000,
        remove_columns=columns_to_remove,
        # new_fingerprint=f'{task}-{tokenizer.name_or_path}-tokenization-{split}',
        load_from_cache_file=True,
    )

    dataset.set_format(type="torch")

    return dataset
