"""Hide sexual example sentences in the classroom demo.

Aggregate metrics for the Sexual label stay visible. This filter only removes
rows whose text or labels would put that content on screen.
"""

from __future__ import annotations

import re

import pandas as pd

LABEL_COLUMNS = ("true_labels", "tfidf_predicted_labels", "qarib_predicted_labels", "trigger_label")
TEXT_COLUMNS = ("text", "text_raw")
# Rows found in the review sample whose wording is sexual even though the
# Sexual label is absent. Identifiers only; the sentences are not stored here.
EXTRA_HIDDEN_ROW_IDS = frozenset({
    "564:TF-IDF + Linear SVM:Appearance",
    "1116:QARiB:Cussing",
    "3423:TF-IDF + Linear SVM:Racial",
    "115:QARiB:Violence",
    "3032:TF-IDF + Linear SVM:Violence",
})
_SEXUAL_LABEL = re.compile(r"(?<![A-Za-z])Sexual(?![A-Za-z])")
_SEXUAL_WORD = re.compile(
    r"(?<![\u0600-\u06FF])(?:"
    r"سكس|جنس|جنسي|زنا|زاني|زنوات|زناة|عورة|فخذ(?:ها|ه)?|مثلية|"
    r"اغتصاب|عاهر(?:ة)?|شرموط(?:ة)?|طيز|نيك|ينيك|منيك|خرقها"
    r")(?![\u0600-\u06FF])"
)


def row_is_sexual_example(row: pd.Series) -> bool:
    if str(row.get("row_id", "")) in EXTRA_HIDDEN_ROW_IDS:
        return True
    labels = " ".join(str(row.get(column, "") or "") for column in LABEL_COLUMNS)
    if _SEXUAL_LABEL.search(labels):
        return True
    text = " ".join(str(row.get(column, "") or "") for column in TEXT_COLUMNS)
    return _SEXUAL_WORD.search(text) is not None


def without_sexual_examples(frame: pd.DataFrame, enabled: bool = True) -> pd.DataFrame:
    if not enabled or frame.empty:
        return frame.reset_index(drop=True)
    hidden = frame.apply(row_is_sexual_example, axis=1)
    return frame.loc[~hidden].reset_index(drop=True)
