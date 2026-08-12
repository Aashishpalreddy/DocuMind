import os

import pytest
from dotenv import load_dotenv
from streamlit.testing.v1 import AppTest

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"), reason="requires a live GROQ_API_KEY (free tier) to run"
)


def test_demo_mode_end_to_end():
    at = AppTest.from_file("app/app.py", default_timeout=60)
    at.run()
    assert not at.exception

    at.run()  # demo corpus indexes on the first pass through the sidebar
    assert not at.exception
    assert any("CloudSync" in s.value for s in at.success)

    at.text_input[0].set_value("What happens if I exceed my storage limit?")
    at.run()
    assert not at.exception

    assert len(at.session_state["query_log"]) == 1
    sidebar_counter = next(m for m in at.sidebar.metric if m.label == "Queries this session")
    assert sidebar_counter.value == "1"
