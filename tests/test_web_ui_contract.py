from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from unittest import TestCase

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids = []

    def handle_starttag(self, tag, attrs) -> None:
        del tag
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"])


class WebUIContractTests(TestCase):
    def test_javascript_id_selectors_exist_once_in_html(self) -> None:
        html = (PROJECT_ROOT / "app/static/index.html").read_text(encoding="utf-8")
        javascript = (PROJECT_ROOT / "app/static/app.js").read_text(encoding="utf-8")
        parser = _IdCollector()
        parser.feed(html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)), "HTML IDs must be unique")
        selectors = set(re.findall(r'querySelector\("#([A-Za-z0-9_-]+)"\)', javascript))
        self.assertFalse(selectors - set(parser.ids), f"Missing HTML IDs: {sorted(selectors - set(parser.ids))}")

    def test_behavior_status_and_event_list_are_present(self) -> None:
        html = (PROJECT_ROOT / "app/static/index.html").read_text(encoding="utf-8")
        for element_id in ("behavior-status", "phone-detector", "smoking-detector", "behavior-events"):
            self.assertIn(f'id="{element_id}"', html)
