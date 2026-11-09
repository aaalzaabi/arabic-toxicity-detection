from dataclasses import dataclass
import re

ARABIC_DIACRITICS = re.compile(r'[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
USER_RE = re.compile(r'(?<!\w)@\w+')
WS_RE = re.compile(r'\s+')

@dataclass(frozen=True)
class ArabicPreprocessConfig:
    remove_urls: bool = True
    remove_usernames: bool = True
    remove_diacritics: bool = True
    remove_tatweel: bool = True
    normalize_alef: bool = True
    normalize_yeh: bool = False
    normalize_ta_marbuta: bool = False
    collapse_whitespace: bool = True

def normalize_arabic(text, cfg):
    text = '' if text is None else str(text)
    if cfg.remove_urls: text = URL_RE.sub(' ', text)
    if cfg.remove_usernames: text = USER_RE.sub(' ', text)
    if cfg.remove_diacritics: text = ARABIC_DIACRITICS.sub('', text)
    if cfg.remove_tatweel: text = text.replace('ـ','')
    if cfg.normalize_alef: text = re.sub('[إأآٱ]', 'ا', text)
    if cfg.normalize_yeh: text = text.replace('ى','ي')
    if cfg.normalize_ta_marbuta: text = text.replace('ة','ه')
    if cfg.collapse_whitespace: text = WS_RE.sub(' ', text).strip()
    return text

def config_from_dict(d):
    return ArabicPreprocessConfig(**d)
