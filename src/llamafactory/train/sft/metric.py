# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()
        
@dataclass
class ComputeClassification:
    tokenizer: "PreTrainedTokenizer"
    labels: list[str] = ('行业白皮书', '技术文档', '问答互动', '攻略指南', '学术报告', '知识科普', '文献综述', '社交媒体', '生活指南', '商业提案', '文学创作')

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {
            "accuracy": [],
            "precision_macro": [], "recall_macro": [], "f1_macro": [],
            "precision_weighted": [], "recall_weighted": [], "f1_weighted": [],
            "fail": []
        }
        self.score_dict.update(
            {f'precision_{label}': [] for label in self.labels},
        )
        self.score_dict.update(
            {f'recall_{label}': [] for label in self.labels},
        )
        self.score_dict.update(
            {f'f1_{label}': [] for label in self.labels},
        )
        return result
    
    def __post_init__(self):
        self._dump()

    def extract_style_label(self, text):
        """从模型的输出中提取 style_label"""
        match = re.search(r"<style_label>(.*?)</style_label>", text)
        if match:
            _match = match.group(1).strip()
            if _match in self.labels:
                return _match
        return "Unknown"
    
    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)\

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        extracted_preds = [self.extract_style_label(pred) for pred in decoded_preds]
        extracted_labels = [self.extract_style_label(label) for label in decoded_labels]

        fail = len([_ for _ in extracted_preds if _ == "Unknown"]) / len(extracted_preds)
        self.score_dict["fail"].append(fail)

        accuracy = accuracy_score(extracted_labels, extracted_preds)
        self.score_dict["accuracy"].append(accuracy)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            extracted_labels, extracted_preds, average='weighted', labels=self.labels, zero_division=0
        )
        self.score_dict["precision_weighted"].append(precision_weighted)
        self.score_dict["recall_weighted"].append(recall_weighted)
        self.score_dict["f1_weighted"].append(f1_weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            extracted_labels, extracted_preds, average='macro', labels=self.labels, zero_division=0
        )
        self.score_dict["precision_macro"].append(precision_macro)
        self.score_dict["recall_macro"].append(recall_macro)
        self.score_dict["f1_macro"].append(f1_macro)

        precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
            extracted_labels, extracted_preds, average=None, labels=self.labels, zero_division=0
        )

        for i, label in enumerate(self.labels):
            self.score_dict[f"precision_{label}"].append(precision_per[i])
            self.score_dict[f"recall_{label}"].append(recall_per[i])
            self.score_dict[f"f1_{label}"].append(f1_per[i])

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        if compute_result:
            return self._dump()
