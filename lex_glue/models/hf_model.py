from omegaconf import DictConfig
from torchmetrics import Accuracy, F1Score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers.configuration_utils import PretrainedConfig
from composer.models import HuggingFaceModel
from composer.metrics import CrossEntropy


class ComposerHFModelWithTokenizer(HuggingFaceModel):
    """Attach the tokenizer to the model so it is easily available to pass to the dataloader builder"""
    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs['tokenizer']
        super().__init__(*args, **kwargs)


def get_huggingface_model(cfg: DictConfig, hf_config: PretrainedConfig):

    metrics = [CrossEntropy(), Accuracy(), F1Score(num_classes=hf_config.num_labels, average="macro")]

    tokenizer_name = cfg.tokenizer_name if cfg.tokenizer_name else cfg.model_name

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        do_lower_case=cfg.do_lower_case if cfg.do_lower_case else False,
        revision=cfg.model_revision if cfg.model_revision else None,
        use_auth_token=True if cfg.use_auth_token else None,  # for private HF Hub models
    )
    # todo also allow seq2seq
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=hf_config,
        revision=cfg.model_revision if cfg.model_revision else None,
        use_auth_token=True if cfg.use_auth_token else None,  # for private HF Hub models
    )

    # GPT2 does not have a PAD token, and that breaks things
    # this lets GPT2 be used in this script without errors
    if "gpt" in tokenizer_name or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return ComposerHFModelWithTokenizer(model=model, tokenizer=tokenizer, metrics=metrics, use_logits=True)
    
