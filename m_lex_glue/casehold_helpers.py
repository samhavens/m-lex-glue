# based on https://github.com/coastalcph/lex-glue/blob/main/experiments/casehold_helpers.py
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm
import re

from filelock import FileLock
from transformers import PreTrainedTokenizer
import datasets
import torch
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


def convert_examples_to_features(
    examples: datasets.Dataset,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, ending in enumerate(example['endings']):
            context = example['context']
            inputs = tokenizer(
                context,
                ending,
                add_special_tokens=True,
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )

            choices_inputs.append(inputs)
        
        label = example['label']

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


class MultipleChoiceDataset(Dataset):
    """
    PyTorch multiple choice dataset class
    """

    features: List[InputFeatures]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        task: &task str,
        max_seq_length: int,
        overwrite_cache=False,
        split: str = "train",
    ):
        examples: datasets.arrow_dataset.Dataset = datasets.load_dataset('lex_glue', task, split=split)  # type: ignore
        tokenizer_name = re.sub('[^a-z]+', ' ', tokenizer.name_or_path).title().replace(' ', '')
        cached_features_file = os.path.join(
            '.cache',
            task,
            "cached_{}_{}_{}_{}".format(
                split,
                tokenizer_name,
                str(max_seq_length),
                task,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        if not os.path.exists(os.path.join('.cache', task)):
            if not os.path.exists('.cache'):
                os.mkdir('.cache')
            os.mkdir(os.path.join('.cache', task))
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {task}")
                logger.info("Training examples: %s", len(examples))
                self.features = convert_examples_to_features(
                    examples,
                    max_seq_length,
                    tokenizer,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def format_casehold_input(context: str, endings: List[str]) -> str:
    """For turning casehold into a seq2seq problem"""
    t = "Context: "
    t += context
    t += "\n\n"
    t += "Which of the following follow from the context?\n"
    for i, end in enumerate(endings):
        t += f"{i}: {end}\n"
    return t


def format_casehold_batched(context: List[str], endings: List[List[str]]):
    return [format_casehold_input(c, e) for c, e in zip(context, endings)]
