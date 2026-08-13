import os

import pytest
from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="requires a live GROQ_API_KEY (free tier) to run"
)


def test_demo_mode_end_to_end():
    at = AppTest.from_file("app/app.py", default_timeout=150)
    at.run()
    assert not at.exception

    at.run()  # demo corpus indexes on the first pass through the sidebar
    assert not at.exception
    assert any("CloudSync" in s.value for s in at.success)

    at.chat_input[0].set_value("What happens if I exceed my storage limit?").run()
    assert not at.exception

    assert len(at.session_state["query_log"]) == 1
    assert len(at.chat_message) == 2
    assert at.chat_message[0].name == "user"
    assert at.chat_message[1].name == "assistant"

    sidebar_counter = next(m for m in at.sidebar.metric if m.label == "Queries this session")
    assert sidebar_counter.value == "1"


def test_suggested_question_chip_click():
    at = AppTest.from_file("app/app.py", default_timeout=150)
    at.run()
    at.run()
    assert not at.exception

    assert len(at.button) == 4  # the suggested-question chips
    at.button[0].click().run()
    assert not at.exception

    assert len(at.chat_message) == 2
    assert len(at.session_state["query_log"]) == 1

    # Chips disappear once a conversation has started.
    assert len(at.button) == 0
