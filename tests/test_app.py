"""Smoke test of the Streamlit app (skipped unless a trained pipeline artifact exists)."""

import pytest

from tesla_forecast.config import load_config, project_root

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not load_config().pipeline_path.exists(), reason="run `make train` first")
def test_app_renders_without_exceptions():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(project_root() / "app" / "streamlit_app.py"), default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 6 and any(m.label.startswith("Forecast t+") for m in at.metric)
