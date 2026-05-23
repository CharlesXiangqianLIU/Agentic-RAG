def test_root_logger_has_stdout_handler():
    import config  # noqa: F401
    import logging
    from config import _StdoutHandler
    assert any(isinstance(h, _StdoutHandler) for h in logging.root.handlers)

def test_root_logger_has_file_handler():
    import config  # noqa: F401
    import logging
    from logging.handlers import RotatingFileHandler
    assert any(isinstance(h, RotatingFileHandler) for h in logging.root.handlers)
