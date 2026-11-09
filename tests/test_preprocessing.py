from aratox_nlp.preprocessing import ArabicPreprocessConfig,normalize_arabic

def test_arabic_normalization():
    out=normalize_arabic('إلىــ @user https://x.com أَهلاً',ArabicPreprocessConfig())
    assert 'http' not in out and '@user' not in out and 'ـ' not in out and 'إ' not in out

def test_optional_folding_off():
    cfg=ArabicPreprocessConfig(normalize_yeh=False,normalize_ta_marbuta=False)
    assert 'ى' in normalize_arabic('فتى',cfg)
    assert 'ة' in normalize_arabic('مدرسة',cfg)
