import logging
from typing import  Union

import datasets
import torch
from composer.utils import dist
from transformers.tokenization_utils_fast import PreTrainedTokenizer, PreTrainedTokenizerFast
from m_lex_glue.data.billsum import get_summarization_preprocessor

from m_lex_glue.data.casehold_helpers import MultipleChoiceDataset, format_casehold_batched, format_casehold_input
from m_lex_glue.data.tasks import multi_label, multiple_choice_qa, summarization, lex_glue_tasks, task_example_types
from m_lex_glue.labels import TASK_NAME_TO_LABELS


log = logging.getLogger(__name__)


def get_preprocessor(task, example_type, tokenizer, max_seq_length):

    if example_type == summarization:
        return get_summarization_preprocessor(tokenizer, max_seq_length)

    def tokenize_function(examples):
        if example_type == multiple_choice_qa:
            # the only MCQA is Casehold, and we already returned that dataloader
            # we would get to this branch once we do seq2seq eval
            context = examples[example_type[0]]
            endings = examples[example_type[1]]
            if isinstance(context, list):
                text = format_casehold_batched(context, endings)
            else:
                text = format_casehold_input(context, endings)
        else:
            text = examples[example_type[1]]

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
            for idx, labels in enumerate(examples['targets']):
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
    """This actually is an extended version of lexglue containing billsum"""

    if task not in lex_glue_tasks:
        raise ValueError(f'task ({task}) must be one of {lex_glue_tasks}')

    if (max_seq_length % 8) != 0:
        log.warning('For performance, a max_seq_length as a multiple of 8 is recommended.')

    log.info(f'Loading {task.upper()} on rank {dist.get_global_rank()}')
    download_config = datasets.DownloadConfig(max_retries=max_retries)

    example_type = task_example_types[task]

    if task == "case_hold":
        # For now we treat this like sequence classification, using
        # AutoModelForMultipleChoice models
        columns_to_remove = [multiple_choice_qa[0], multiple_choice_qa[1]]
        return MultipleChoiceDataset(
            tokenizer=tokenizer,
            task=task,
            max_seq_length=max_seq_length,
            split=split,
        )
    else:
        dataset = datasets.load_dataset(
            'lex_glue',
            task,
            split=split,
            download_config=download_config,
        )
        columns_to_remove = ['text']

    log.info(f'Starting tokenization by preprocessing over {num_workers} threads!')
    

    assert isinstance(dataset, datasets.Dataset)

    pre_process_fn = get_preprocessor(task, example_type, tokenizer, max_seq_length)

    if example_type == multi_label:
        # the dataset which we download is typed in such a way that we cannot coerce the labels column into a Float
        # so here we rename the labels column to targets, so we can name it back to labels as a Float type
        # which both torch and torchmetrics expect
        dataset = dataset.rename_column("labels", "targets")
        columns_to_remove.append("targets")  # type: ignore

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

