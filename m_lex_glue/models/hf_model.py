from omegaconf import DictConfig
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1Score
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

from composer.loss import binary_cross_entropy_with_logits
from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy, LossMetric
from composer.metrics.nlp import LanguageCrossEntropy

from m_lex_glue.data.data import multi_class, multi_label, multiple_choice_qa, summarization, task_example_types
from m_lex_glue.labels import TASK_NAME_TO_LABELS
from m_lex_glue.metrics.modified_metrics import FloatAccuracy, FloatF1, RougeWithDetokenizer
from m_lex_glue.models.gpt_for_multiple_choice import GPT2ForMultipleChoice, GPTJForMultipleChoice
from m_lex_glue.models.hf_fsdp import is_fsdp_able, prepare_hf_model_for_fsdp


class ComposerHFModelWithTokenizer(HuggingFaceModel):
    """Attach the tokenizer to the model so it is easily available to pass to the dataloader builder"""
    def __init__(self, *args, train_metrics=None, eval_metrics=None, **kwargs):
        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)
        self.train_metrics = {metric.__class__.__name__: metric for metric in train_metrics}  # type: ignore
        if eval_metrics:
            self.eval_metrics = {metric.__class__.__name__: metric for metric in eval_metrics}
        else:
            self.eval_metrics = self.train_metrics

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.eval_metrics


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

    task = task_example_types[cfg.task]
    eval_metrics = None

    # @TODO also allow seq2seq
    if task == multi_class:
        hf_config  = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
            finetuning_task=cfg.task,
        )
        train_metrics = [CrossEntropy(), Accuracy(), F1Score(num_classes=hf_config.num_labels, average="macro")]
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=hf_config,
            use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
            ignore_mismatched_sizes=True,  # showed up with DeBERTa
        )
    elif task == multi_label:
        hf_config  = AutoConfig.from_pretrained(
            cfg.model_name,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
            problem_type="multi_label_classification",
            finetuning_task=cfg.task,
        )
        train_metrics = [
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
    elif task == multiple_choice_qa:
        hf_config  = AutoConfig.from_pretrained(
            cfg.model_name,
            finetuning_task=cfg.task,
            num_labels=len(TASK_NAME_TO_LABELS[cfg.task]),
        )
        train_metrics = [CrossEntropy(), Accuracy(), F1Score(num_classes=hf_config.num_labels, average="macro")]
        if 'gpt2' in cfg.model_name:
            model = GPT2ForMultipleChoice.from_pretrained(
                cfg.model_name,
                config=hf_config,
                use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
            )
        elif 'gptj' in cfg.model_name:
            model = GPTJForMultipleChoice.from_pretrained(
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
    elif task == summarization:
        hf_config  = AutoConfig.from_pretrained(
            cfg.model_name,
            finetuning_task=cfg.task,
        )
        if 'gpt' in cfg.model_name:
            train_metrics = [LanguageCrossEntropy(hf_config.vocab_size)]
            eval_metrics = [
                LanguageCrossEntropy(hf_config.vocab_size),
                RougeWithDetokenizer(detokenizer=tokenizer),
            ]
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                config=hf_config,
                use_auth_token=cfg.get('use_auth_token', None),
            )
        else:
            # Need to figure out T5 models...
            raise NotImplementedError("T5 models not yet supported!")
            train_metrics = [LanguageCrossEntropy(hf_config.vocab_size)]
            eval_metrics = [
                LanguageCrossEntropy(hf_config.vocab_size),
                RougeWithDetokenizer(detokenizer=tokenizer),
            ]
            model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.model_name,
                config=hf_config,
                use_auth_token=cfg.get('use_auth_token', None),
            )
    else:
        raise ValueError(f"Your YAML file has the task set to {cfg.task}, which is invalid."
                         f"Please use a task in {list(task_example_types.keys())}")

    # GPT2 does not have a PAD token, and that breaks things
    # this lets GPT2 be used in this script without errors
    if "gpt" in tokenizer_name or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id  # type: ignore

    # For very large models which will be distributed over multiple GPUs
    if is_fsdp_able(model):
        prepare_hf_model_for_fsdp(model)

    return ComposerHFModelWithTokenizer(
        model=model,
        tokenizer=tokenizer,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        use_logits=True
    )
    
