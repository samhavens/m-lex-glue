from typing import Any, Optional

import torch
import torch.nn.functional as F
from composer.loss import binary_cross_entropy_with_logits
from composer.metrics import CrossEntropy, LossMetric
from composer.metrics.nlp import LanguageCrossEntropy
from composer.models import HuggingFaceModel
from omegaconf import DictConfig
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1Score
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMultipleChoice, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

from m_lex_glue.data.tasks import (multi_class, multi_label, multiple_choice_qa,
                                   summarization, task_example_types)
from m_lex_glue.labels import TASK_NAME_TO_LABELS
from m_lex_glue.metrics.modified_metrics import (FloatAccuracy, FloatF1,
                                                 RougeWithDetokenizer)
from m_lex_glue.models.gpt_for_multiple_choice import (GPT2ForMultipleChoice,
                                                       GPTJForMultipleChoice)
from m_lex_glue.models.hf_fsdp import prepare_hf_model_for_fsdp


class RougeableComposerHFModel(HuggingFaceModel):
    """This subclass overrides eval_forward,
    which must be passed batches during evaluation with a `eval_mode` key
    This key allows it to know whether to do one forward pass (for CLM) or greedy decoding (for Rouge)"""

    def __init__(
        self,
        *args,
        train_metrics=None,
        val_metrics=None,
        max_length: int = 512,
        do_sample: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_length = max_length
        self.do_sample = do_sample
        self.train_metrics = {
            metric.__class__.__name__: metric for metric in train_metrics
        }  # type: ignore
        if val_metrics:
            self.val_metrics = {
                metric.__class__.__name__: metric for metric in val_metrics
            }
        else:
            self.val_metrics = self.train_metrics

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def get_targets(self, batch: dict):
        targets = torch.roll(batch['labels'], shifts=-1)
        targets[:, -1] = -100
        return targets

    def loss(self, outputs: Tensor, batch: dict):
        targets = self.get_targets(batch)
        return F.cross_entropy(outputs.view(-1, outputs.size(-1)),
                               targets.view(-1),
                               ignore_index=-100)

    def eval_forward(self, batch, outputs: Optional[Any] = None):
        if outputs:
            # In train eval...
            return super().eval_forward(batch=batch, outputs=outputs)

        # eval_mode should be present on eval_batches
        mode = batch.get("eval_mode", ["clm"])[0]

        if "eval_mode" in batch:
            # delete this key as the forward pass calls **batch and will freak out about unexpected keys
            del batch['eval_mode']

        if mode == "clm":
            # eval mode CLM evaluation
            # this is handled fine by the parent class
            return super().eval_forward(batch=batch, outputs=outputs)
        else:
            # "seq2seq" mode

            # nucleus / topk sampling:
            outputs = self.model.generate(
                batch['input_ids'],
                max_new_tokens=self.max_length,
                do_sample=self.do_sample,
                top_p=0.90,
                top_k=0,
                no_repeat_ngram_size=3,
            )

            return outputs


def get_huggingface_model(cfg: DictConfig):
    """Instantiate a single label or multi label SequenceClassification hf transformers model,
    or if the task is case_hold, a MultipleChoice model.

    Also instantiate the tokenizer, and if the model is GPT2, set the PAD token as the EOS token

    Then wrap these as Composer models, with the Tokenizer stored as a property
    """
    tokenizer_name = cfg.get('tokenizer_name', cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False
        if "deberta" in cfg.model_name else True  # crazy byte conversion error
    )

    task_type = task_example_types[cfg.task]
    val_metrics = None

    if task_type == multi_class:
        hf_config = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
            finetuning_task=cfg.task,
        )
        train_metrics = [
            CrossEntropy(),
            Accuracy(),
            F1Score(num_classes=hf_config.num_labels, average="macro")
        ]
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=hf_config,
            ignore_mismatched_sizes=True,  # showed up with DeBERTa
        )
    elif task_type == multi_label:
        hf_config = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
            problem_type="multi_label_classification",
            finetuning_task=cfg.task,
        )
        train_metrics = [
            LossMetric(binary_cross_entropy_with_logits),
            FloatAccuracy(task="multilabel"),
            FloatF1(num_classes=hf_config.num_labels,
                    average="macro",
                    task="multilabel")
        ]
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=hf_config,
            ignore_mismatched_sizes=True,  # showed up with DeBERTa
        )
    elif task_type == multiple_choice_qa:
        hf_config = AutoConfig.from_pretrained(
            cfg.model_name,
            finetuning_task=cfg.task,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
        )
        train_metrics = [
            CrossEntropy(),
            Accuracy(),
            F1Score(num_classes=hf_config.num_labels, average="macro")
        ]
        if 'gpt2' in cfg.model_name:
            model = GPT2ForMultipleChoice.from_pretrained(
                cfg.model_name,
                config=hf_config,
            )
        elif 'gptj' in cfg.model_name:
            model = GPTJForMultipleChoice.from_pretrained(
                cfg.model_name,
                config=hf_config,
            )
        else:
            model = AutoModelForMultipleChoice.from_pretrained(
                cfg.model_name,
                config=hf_config,
            )
    elif task_type == summarization:
        hf_config = AutoConfig.from_pretrained(
            cfg.model_name,
            finetuning_task=cfg.task,
        )
        if "gpt" in cfg.model_name:
            train_metrics = [LanguageCrossEntropy(hf_config.vocab_size)]
            val_metrics = [
                LanguageCrossEntropy(hf_config.vocab_size),
                RougeWithDetokenizer(detokenizer=tokenizer),
            ]
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                config=hf_config,
            )
        elif "opt" in cfg.model_name or "bloom" in cfg.model_name:
            train_metrics = [LanguageCrossEntropy(len(tokenizer))]
            val_metrics = [
                LanguageCrossEntropy(len(tokenizer)),
                RougeWithDetokenizer(detokenizer=tokenizer),
            ]
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                config=hf_config,
            )
        elif "t5" in cfg.model_name:
            print(
                "\n\nt5 is untested! There are likely bugs in this implementation!\n\n"
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.model_name,
                config=hf_config,
            )
            # t5 models have a discrepancy between model size and vocab size
            # this causes issues because we automatically resize the model embeddings,
            # but the hf_config.vocab_size doesn't match, so use len(tokenizer) instead
            #
            # this mismatch causes the warning you see:
            # The number of tokens in the tokenizer and the number of tokens in the model
            # are different. Resizing the model tokenizer to 32100 from 32128.
            train_metrics = [LanguageCrossEntropy(len(tokenizer))]
            val_metrics = [
                LanguageCrossEntropy(len(tokenizer)),
                RougeWithDetokenizer(detokenizer=tokenizer),
            ]
    else:
        raise ValueError(
            f"Your YAML file has the task set to {cfg.task}, which is invalid."
            f"Please use a task in {list(task_example_types.keys())}")

    # GPT2 does not have a PAD token, and that breaks things
    # this lets GPT2 be used in this script without errors
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id  # type: ignore

    # For very large models which will be distributed over multiple GPUs
    fsdp_config = cfg.get('fsdp_config', None)
    if fsdp_config is not None:
        prepare_hf_model_for_fsdp(model)

    if task_type == summarization:
        return RougeableComposerHFModel(
            model=model,
            tokenizer=tokenizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            use_logits=True,
            max_length=cfg.get(
                "summary_max_length",
                512),  # limited by min(corpus_max, model_max_length)
            do_sample=cfg.get("summary_do_sample", True),
        )
    else:
        return HuggingFaceModel(model=model,
                                tokenizer=tokenizer,
                                metrics=train_metrics,
                                use_logits=True)
