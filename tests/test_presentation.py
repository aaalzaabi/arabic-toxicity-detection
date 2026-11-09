import pandas as pd

from aratox_nlp.presentation import without_sexual_examples


def test_classroom_filter_removes_sexual_labels_and_keeps_other_rows():
    frame = pd.DataFrame([
        {"row_id": "1", "text": "جملة عادية", "true_labels": "Hatred", "tfidf_predicted_labels": "Hatred", "qarib_predicted_labels": "none", "trigger_label": "Hatred"},
        {"row_id": "2", "text": "جملة أخرى", "true_labels": "Sexual", "tfidf_predicted_labels": "Sexual", "qarib_predicted_labels": "none", "trigger_label": "Cussing"},
        {"row_id": "564:TF-IDF + Linear SVM:Appearance", "text": "جملة محفوظة", "true_labels": "Violence", "tfidf_predicted_labels": "Appearance", "qarib_predicted_labels": "Violence", "trigger_label": "Appearance"},
        {"row_id": "4", "text": "كلمة عورة هنا", "true_labels": "none", "tfidf_predicted_labels": "none", "qarib_predicted_labels": "none", "trigger_label": "Violence"},
    ])
    hidden = without_sexual_examples(frame, enabled=True)
    assert hidden.row_id.tolist() == ["1"]
    shown = without_sexual_examples(frame, enabled=False)
    assert len(shown) == 4
