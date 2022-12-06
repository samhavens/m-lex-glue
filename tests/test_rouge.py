import math
from typing import List
import torch

from m_lex_glue.metrics.modified_metrics import RougeWithDetokenizer



class DummyDetokenizer:
    def __init__(self, token_to_str):
        self.token_to_str = token_to_str
        self.pad_token_id = 0

    def detokenize(self, t: torch.Tensor) -> List[str]:
        l = t.tolist()
        # huggingface tokenizers can decode a batch or single tensor, mock that
        if isinstance(l[0], list):
            return [self.token_to_str[token] for token in l[0]]
        else:
            return [self.token_to_str[token] for token in l]

    def batch_decode(self, t: torch.Tensor, **kwargs) -> List[str]:
        return [" ".join(self.detokenize(t))]


def test_rouge_with_detokenizer():
    # create a sample dictionary of token ids to strings
    token_to_str = {0: "", 1: "hello", 2: "world", 3: "star"}

    # create a sample tensor of token probs for the predictions
    preds = torch.tensor([[[0.05, 0.9, 0.05], [0.1, 0.1, 0.8], [1.0, 0, 0]]])

    # create a sample tensor of token ids for the ground truth labels
    targets = torch.tensor([[1, 2, 3]])

    # instantiate the RougeWithDetokenizer class
    rouge = RougeWithDetokenizer(DummyDetokenizer(token_to_str))

    # call the update method with the sample inputs
    rouge.update(preds, targets)

    # compute the ROUGE scores
    scores = rouge.compute()

    # assert that the computed ROUGE scores match the expected values
    assert math.isclose(scores["rouge1_fmeasure"].item(), 0.8, rel_tol=1e-04)
    assert math.isclose(scores["rouge1_precision"].item(), 1.0, rel_tol=1e-04)
    assert math.isclose(scores["rouge1_recall"].item(), 0.6667, rel_tol=1e-04)
    assert math.isclose(scores["rouge2_fmeasure"].item(), 0.6667, rel_tol=1e-04)
    assert math.isclose(scores["rouge2_precision"].item(), 1.0, rel_tol=1e-04)
    assert math.isclose(scores["rouge2_recall"].item(), 0.5, rel_tol=1e-04)
    assert math.isclose(scores["rougeL_fmeasure"].item(), 0.8, rel_tol=1e-04)
    assert math.isclose(scores["rougeL_precision"].item(), 1.0, rel_tol=1e-04)
    assert math.isclose(scores["rougeL_recall"].item(), 0.6667, rel_tol=1e-04)
    assert math.isclose(scores["rougeLsum_fmeasure"].item(), 0.8, rel_tol=1e-04)
    assert math.isclose(scores["rougeLsum_precision"].item(), 1.0, rel_tol=1e-04)
    assert math.isclose(scores["rougeLsum_recall"].item(), 0.6667, rel_tol=1e-04)


if __name__ == "__main__":
    test_rouge_with_detokenizer()