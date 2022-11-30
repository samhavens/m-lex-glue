from omegaconf import OmegaConf as om
import torch

from m_lex_glue.data import create_lexglue_dataset
from m_lex_glue.models.hf_model import get_huggingface_model


if __name__ == "__main__":
    with open('tests/test_unfair_tos.yaml') as f:
        cfg = om.load(f)
    model = get_huggingface_model(cfg)  # really just need the tokenizer
    ds = create_lexglue_dataset("unfair_tos", model.tokenizer, split="test", max_seq_length=128)
    assert len(ds) > 100
    assert ds[0]['labels'].dtype == torch.float, f"labels must be of type float for multiplabel classification, got {ds[0]['labels'].dtype}"
