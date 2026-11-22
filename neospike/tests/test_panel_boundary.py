"""Every dashboard tab panel, the map at least, sits inside a React error
boundary; a render error shows a short message, the error line and "Try again" instead of a blank tab.
Source-reading tests (the host dashboard has no boundary of its own around plugin tabs)."""
from __future__ import annotations

import re
from pathlib import Path


DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
BOUNDARY_KEYS = ("boundary.title", "boundary.retry")
PANELS = ("InstallView", "MemosView", "MemoryView", "DialogsView", "SettingsView", "MapView")


def _source(path: Path | None = None) -> str:
    return (path or DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8-sig")


def _table(source: str, language: str) -> str:
    """Text of one STRINGS table (en: { ... } up to the next language or the end of STRINGS)."""
    match = re.search(rf"\n {{6}}{language}: \{{\n(.*?)\n {{6}}\}}", source, flags=re.DOTALL)
    assert match, language
    return match.group(1)


def boundary_problems(source: str) -> list[str]:
    """What is missing for the error boundary contract (empty when it holds)."""
    problems = []
    match = re.search(r"class (\w+) extends React\.Component \{(.*?)\n  \}\n", source, flags=re.DOTALL)
    if match is None:
        return ["no class component"]
    name, body = match.group(1), match.group(2)
    for needed in ("static getDerivedStateFromError(", "componentDidCatch(", "this.setState({ error: null })",
                   't("boundary.title")', 't("boundary.retry")', "failureMessage(this.state.error)"):
        if needed not in body:
            problems.append(f"{name} lacks {needed}")
    render_tabs = source[source.index("function paintTabs("):source.index("function dashboardLocale(")]
    for panel in PANELS:
        if not re.search(rf"h\({name}, null, h\({panel}\b", render_tabs):
            problems.append(f"{panel} is not wrapped")
    for language in ("en", "ko"):
        table = _table(source, language)
        problems += [f"{key} missing in {language}" for key in BOUNDARY_KEYS if f'"{key}":' not in table]
    return problems


def test_every_tab_panel_has_an_error_boundary() -> None:
    assert boundary_problems(_source()) == []


def test_contract_fails_without_the_boundary() -> None:
    source = _source()
    unwrapped = source.replace("h(PanelBoundary, null, h(MapView,", "(h(MapView,")
    assert "MapView is not wrapped" in boundary_problems(unwrapped)
    without_class = re.sub(r"class PanelBoundary extends React\.Component", "function PanelBoundary()", source)
    assert boundary_problems(without_class) == ["no class component"]
    for language in ("en", "ko"):
        table = _table(source, language)
        stripped = source.replace(table, table.replace('"boundary.retry":', '"boundary.gone":'))
        assert f"boundary.retry missing in {language}" in boundary_problems(stripped)
