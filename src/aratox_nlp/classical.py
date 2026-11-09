from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

@dataclass
class ClassicalConfig:
    word_ngram_min: int = 1
    word_ngram_max: int = 3
    char_ngram_min: int = 2
    char_ngram_max: int = 5
    min_df: int = 2
    class_weight: str | None = 'balanced'

def build_features(cfg):
    return FeatureUnion([
        ('word', TfidfVectorizer(analyzer='word', ngram_range=(cfg.word_ngram_min,cfg.word_ngram_max), min_df=cfg.min_df, sublinear_tf=True, dtype=np.float32)),
        ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(cfg.char_ngram_min,cfg.char_ngram_max), min_df=cfg.min_df, sublinear_tf=True, dtype=np.float32)),
    ])

def build_multilabel_svm(cfg): return OneVsRestClassifier(LinearSVC(class_weight=cfg.class_weight))
def build_binary_svm(cfg): return LinearSVC(class_weight=cfg.class_weight)
def decision_to_unit_interval(scores): return 1/(1+np.exp(-np.asarray(scores)))
