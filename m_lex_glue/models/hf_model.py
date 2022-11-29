from omegaconf import DictConfig
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1Score
from transformers import AutoModelForMultipleChoice, AutoModelForSequenceClassification, AutoTokenizer

from transformers.configuration_utils import PretrainedConfig
from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy

from m_lex_glue.data import single_label, multi_label, multiple_choice_qa, task_example_types
from m_lex_glue.models.gpt_for_multiple_choice import GPT2ForMultipleChoice

class ComposerHFModelWithTokenizer(HuggingFaceModel):
    """Attach the tokenizer to the model so it is easily available to pass to the dataloader builder"""
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)


def get_huggingface_model(cfg: DictConfig, hf_config: PretrainedConfig):
    """Instantiate a single label or multi label SequenceClassification hf transformers model,
    or if the task is case_hold, a MultipleChoice model.

    Also instantiate the tokenizer, and if the model is GPT2, set the PAD token as the EOS token

    Then wrap these as Composer models, with the Tokenizer stored as a property
    """

    metrics = [CrossEntropy(), Accuracy(), F1Score(num_classes=hf_config.num_labels, average="macro")]

    tokenizer_name = cfg.get('tokenizer_name', cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
        use_fast=False if "deberta" in cfg.model_name else True  # crazy byte conversion error
    )
    # todo also allow seq2seq
    if task_example_types[cfg.task] == single_label:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            config=hf_config,
            use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
        )
    elif task_example_types[cfg.task] == multi_label:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            problem_type="multi_label_classification",
            config=hf_config,
            use_auth_token=cfg.get('use_auth_token', None),  # for private HF Hub models
        )
    elif task_example_types[cfg.task] == multiple_choice_qa:
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
    
