from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.f_beta import F1Score
from torchmetrics.text.rouge import ROUGEScore


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


class RougeWithDetokenizer(ROUGEScore):
    """AFAICT (see https://docs.mosaicml.com/en/v0.11.1/trainer/evaluation.html)
    this is necessary to connect a composer metric to a tokenizer for detokenizing,
    as preds and target as tensors
    
    I tried to do it with a custom eval dataloader, but preds was still a tensor
    This way we can use the same format as the train data loader"""
    def __init__(
        self,
        detokenizer,
        use_stemmer: bool = False,
        normalizer: Optional[Callable[[str], str]] = None,
        tokenizer: Optional[Callable[[str], Sequence[str]]] = None,
        accumulate: Literal["avg", "best"] = "best",
        rouge_keys: Union[str, Tuple[str, ...]] = ('rouge1', 'rouge2', 'rougeL', 'rougeLsum'),
        **kwargs: Any
    ):
        self.detokenizer = detokenizer
        super().__init__(use_stemmer, normalizer, tokenizer, accumulate, rouge_keys, **kwargs)  # type: ignore

    def update(self, preds: Tensor, target: Tensor) -> None:
        # -100 is special value which means "don't use when computing loss"
        # used for ignoring pad tokens for loss
        target = torch.where(target != -100, target, self.detokenizer.pad_token_id)
        preds_decoded: Sequence[str] = self.detokenizer.batch_decode(
            preds,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        target_decoded: Sequence[str] = self.detokenizer.batch_decode(
            target,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        preds_decoded = [pred.strip() for pred in preds_decoded]
        target_decoded = [label.strip() for label in target_decoded]

        return super().update(preds_decoded, target_decoded)