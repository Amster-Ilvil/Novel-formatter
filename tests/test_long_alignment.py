from engine.long_document_matcher import build_windows

def test_windows():
    assert build_windows(['a','b'],(1,2))
