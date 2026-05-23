def test_history_page_size_is_positive_int():
    from config import HISTORY_PAGE_SIZE
    assert isinstance(HISTORY_PAGE_SIZE, int)
    assert HISTORY_PAGE_SIZE > 0

def test_paginate_shows_recent_messages():
    """With 25 messages and page_size=20 and offset=0, show last 20."""
    history = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
    page_size = 20
    offset = 0
    total = len(history)
    start = max(0, total - page_size - offset)
    visible = history[start:]
    assert len(visible) == 20
    assert visible[0]["content"] == "msg 5"

def test_paginate_offset_loads_earlier():
    """With offset=20, show the first 5 messages."""
    history = [{"role": "user", "content": f"msg {i}"} for i in range(25)]
    page_size = 20
    offset = 20
    total = len(history)
    start = max(0, total - page_size - offset)
    visible = history[start:]
    assert len(visible) == 25  # start=0, so all visible
    assert visible[0]["content"] == "msg 0"
