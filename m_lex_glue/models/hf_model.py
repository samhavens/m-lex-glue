from omegaconf import DictConfig
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1Score
from transformers import AutoConfig, AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoTokenizer

from composer.loss import binary_cross_entropy_with_logits
from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy, LossMetric

from m_lex_glue.data import multi_class, multi_label, multiple_choice_qa, task_example_types
from m_lex_glue.labels import TASK_NAME_TO_LABELS
from m_lex_glue.models.gpt_for_multiple_choice import GPT2ForMultipleChoice


class ComposerHFModelWithTokenizer(HuggingFaceModel):
    """Attach the tokenizer to the model so it is easily available to pass to the dataloader builder"""
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)


class FloatAccuracy(Accuracy):
    """For multi-label classification, torchmetrics requires that the target tensor be a torch.int
    however, the binary_cross_entropy_with_logits function requires that the target tensor be a torch.float
    So this class converts the target to an int tensor before computing metrics"""
    def update(self, preds: Tensor, target: Tensor) -> None:
        target = target.int()
        return super().update(preds, target)


class FloatF1(F1Score):
    """For multi-label classification, torchmetrics requires that the target tensor be a torch.int
    however, the binary_cross_entropy_with_logits function requires that the target tensor be a torch.float
    So this class converts the target to an int tensor before computing metrics"""
    def update(self, preds: Tensor, target: Tensor) -> None:
        target = target.int()
        return super().update(preds, target)


def get_huggingface_model(cfg: DictConfig):
    """Instantiate a single label or multi label SequenceClassification hf transformers model,
    or if the task is case_hold, a MultipleChoice model.

    Also instantiate the tokenizer, and if the model is GPT2, set the PAD token as the EOS token

    Then wrap these as Composer models, with the Tokenizer stored as a property
    """
    tokenizer_name = cfg.get('tokenizer_name', cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
        use_fast=False if "deberta" in cfg.model_name else True  # crazy byte conversion error
    )
    # todo also allow seq2seq
    if task_example_types[cfg.task] == multi_class:
        hf_config  = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
            finetuning_task=cfg.task,
        )
        metrics = [CrossEntropy(), Accuracy(), F1Score(num_classes=hf_config.num_labels, average="macro")]
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=hf_config,
            use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
            ignore_mismatched_sizes=True,  # showed up with DeBERTa
        )
    elif task_example_types[cfg.task] == multi_label:
        hf_config  = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
            problem_type="multi_label_classification",
            finetuning_task=cfg.task,
        )
        metrics = [
            LossMetric(binary_cross_entropy_with_logits),
            FloatAccuracy(task="multilabel"),
            FloatF1(num_classes=hf_config.num_labels, average="macro", task="multilabel")
        ]
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=hf_config,
            use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
            ignore_mismatched_sizes=True,  # showed up with DeBERTa
        )
    elif task_example_types[cfg.task] == multiple_choice_qa:
        hf_config  = AutoConfig.from_pretrained(
            cfg.model_name,
            finetuning_task=cfg.task,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
        )
        metrics = [CrossEntropy(), Accuracy(), F1Score(num_classes=hf_config.num_labels, average="macro")]
        if 'gpt' in cfg.model_name:
            model = GPT2ForMultipleChoice.from_pretrained(
                cfg.model_name,
                config=hf_config,
                use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
            )
        else:
            model = AutoModelForMultipleChoice.from_pretrained(
                cfg.model_name,
                config=hf_config,
                use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
            )
    else:
        raise ValueError(f"Your YAML file has the task set to {cfg.task}, which is invalid."
                         f"Please use a task in {list(task_example_types.keys())}")

    # GPT2 does not have a PAD token, and that breaks things
    # this lets GPT2 be used in this script without errors
    if "gpt" in tokenizer_name or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id  # type: ignore

    return ComposerHFModelWithTokenizer(model=model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
    
