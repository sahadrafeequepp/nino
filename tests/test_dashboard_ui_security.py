"""Tests for dashboard UI security, plain-text immunity, and secret masking."""

import re
from pathlib import Path
import pytest


def test_dashboard_uses_text_content_for_facts_and_untrusted_data():
    """Verify dashboard.ts renders facts and untrusted memory using textContent, never innerHTML."""
    target = Path("web/dashboard/dashboard.ts")
    if not target.exists():
        pytest.skip("Web dashboard not present")
    dashboard_ts = target.read_text(encoding="utf-8")

    # In renderFactsTable:
    assert "tdContent.textContent = f.fact" in dashboard_ts
    assert "tdContent.innerHTML" not in dashboard_ts

    # In tool activity feed:
    assert "toolName.textContent = act.tool_name" in dashboard_ts

    # In agent runs feed:
    assert "goal.textContent = `Goal: ${r.goal}`" in dashboard_ts


def test_dashboard_html_offline_dependencies():
    """Ensure dashboard/index.html contains zero external CDN scripts, fonts, or stylesheets."""
    target = Path("web/dashboard/index.html")
    if not target.exists():
        pytest.skip("Web dashboard not present")
    html = target.read_text(encoding="utf-8")

    # Reject external domains in links and scripts
    assert "http://" not in html
    assert "https://" not in html
    assert "cdn." not in html
    assert "googleapis.com" not in html
    assert "cdnjs" not in html
    assert "unpkg.com" not in html
