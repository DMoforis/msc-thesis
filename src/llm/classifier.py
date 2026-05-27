"""
classifier.py
-------------
Window-title activity classifier — Ollama LLM with keyword fallback.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Classifies the active application window title into one of nine productivity
categories used by the fusion layer and intervention engine.

Design
------
  1. Primary path:  Ollama (local Llama 3.1 8B, GPU via CUDA).
                    Single-shot prompt engineered for a one-word category reply.
                    2-second timeout enforced via a thread pool future.
  2. Fallback path: Rule-based keyword matching.
                    Activates automatically if Ollama is not running,
                    not installed, or exceeds the timeout.
  3. Cache:         LRU dict keyed on the lowercased title — same title always
                    returns the same category without re-querying the LLM.

Categories
----------
  Academic Work | Software Development | Communication | Social Media |
  Entertainment | Document Editing | Research / Reading | System / Utility | Other

Ollama setup (one-time — see SPEC.md §10)
-----------------------------------------
  ollama pull llama3.1:8b
  pip install ollama

Once Ollama is running, this module switches to LLM mode automatically.
Until then, keyword matching provides full functionality.
"""

import sys
import os
import concurrent.futures
import re
from functools import lru_cache

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import OLLAMA_MODEL, OLLAMA_TIMEOUT

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORIES
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES: list[str] = [
    "Academic Work",
    "Software Development",
    "Communication",
    "Social Media",
    "Entertainment",
    "Document Editing",
    "Research / Reading",
    "System / Utility",
    "Other",
]

_CATEGORY_SET = {c.lower() for c in CATEGORIES}

# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD RULES  (fallback when Ollama is unavailable)
# Rules are evaluated in order; the first match wins.
# Each rule: (category, list_of_keywords_to_match_in_lowercased_title)
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORD_RULES: list[tuple[str, list[str]]] = [
    # ── Highest-specificity rules first ──────────────────────────────────────
    ("Software Development", [
        "vs code", "vscode", "visual studio", "pycharm", "intellij", "eclipse",
        "android studio", "xcode", "jupyter", "notebook", "terminal", "powershell",
        "cmd.exe", "git", "github", "gitlab", "bitbucket", "stackoverflow",
        "stackoverflow.com", "docker", "kubernetes", "npm", "webpack", "postman",
        "insomnia", "dbeaver", "pgadmin", "redis", "mongodb",
    ]),
    # Document Editing before Academic Work so "Microsoft Word — thesis.docx"
    # matches the application name rather than the filename keyword.
    ("Document Editing", [
        "microsoft word", "google docs", "libreoffice writer",
        "microsoft excel", "google sheets", "libreoffice calc",
        "microsoft powerpoint", "google slides", "libreoffice impress",
        "notion", "onenote", "evernote", "obsidian", "logseq",
        "notepad", "notepad++", "sublime text", "typora",
    ]),
    ("Academic Work", [
        "overleaf", "latex", "msc thesis", "thesis", "dissertation", "lecture",
        "moodle", "canvas", "blackboard", "coursera", "edx", "khan academy",
        "university", "academia.edu", "researchgate", "arxiv", "pubmed",
        "google scholar", "scholar.google", "zotero", "mendeley", "endnote",
    ]),
    ("Communication", [
        "gmail", "outlook", "yahoo mail", "thunderbird", "mail",
        "microsoft teams", "slack", "zoom", "google meet", "skype", "webex",
        "discord", "whatsapp", "telegram", "signal", "messenger",
        "inbox", "compose", "reply",
    ]),
    ("Social Media", [
        "facebook", "instagram", "twitter", "x.com", "tiktok", "linkedin",
        "reddit", "snapchat", "pinterest", "tumblr", "mastodon",
    ]),
    ("Entertainment", [
        "netflix", "prime video", "disney+", "hulu", "hbo", "apple tv",
        "spotify", "youtube music", "soundcloud", "twitch",
        "steam", "epic games", "origin", "battle.net", "valorant",
        "minecraft", "league of legends", "fortnite",
        "vlc", "mpv", "media player",
    ]),
    # YouTube goes after Entertainment to avoid mis-classifying music/gaming streams
    ("Social Media", [
        "youtube",
    ]),
    ("Document Editing", [
        "microsoft word", "google docs", "libreoffice writer",
        "microsoft excel", "google sheets", "libreoffice calc",
        "microsoft powerpoint", "google slides", "libreoffice impress",
        "notion", "onenote", "evernote", "obsidian", "logseq",
        "notepad", "notepad++", "sublime text", "typora",
    ]),
    ("Research / Reading", [
        "wikipedia", "wiki", "medium.com", "substack", "blog",
        "bbc news", "cnn", "reuters", "guardian", "nytimes",
        "nature.com", "science.org", "ieee", "acm", "springer",
        "pdf", ".pdf",
    ]),
    ("System / Utility", [
        "task manager", "activity monitor", "resource monitor",
        "control panel", "settings", "system preferences",
        "file explorer", "finder", "7-zip", "winrar", "winzip",
        "calculator", "calendar", "clock", "paint", "snipping tool",
        "regedit", "msconfig", "device manager",
    ]),
]


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class WindowClassifier:
    """
    Classifies an active window title into one of nine activity categories.

    Primary: Ollama (Llama 3.2 8B) with a 2-second timeout.
    Fallback: rule-based keyword matching.
    Cache:    LRU dict — same title always returns the same category.

    Example
    -------
    >>> clf = WindowClassifier()
    >>> clf.classify("DMoforis/msc-thesis — GitHub — Google Chrome")
    'Software Development'
    >>> clf.classify("Netflix")
    'Entertainment'
    """

    def __init__(self, model: str = OLLAMA_MODEL, timeout: float = OLLAMA_TIMEOUT):
        self._model      = model
        self._timeout    = timeout
        self._cache: dict[str, str] = {}
        self._ollama_ok: bool | None = None   # None = not yet probed
        self._executor   = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ollama-clf"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, window_title: str) -> str:
        """
        Return the activity category for the given window title.

        Parameters
        ----------
        window_title : str
            Raw active window title string (e.g. from win32 GetForegroundWindow).

        Returns
        -------
        str
            One of the nine CATEGORIES strings.
        """
        if not window_title or not window_title.strip():
            return "Other"

        key = window_title.strip().lower()

        if key in self._cache:
            return self._cache[key]

        if self._should_try_ollama():
            category = self._classify_ollama(window_title)
        else:
            category = _classify_keywords(window_title)

        self._cache[key] = category
        return category

    @property
    def using_llm(self) -> bool:
        """True if the last successful classification used Ollama."""
        return bool(self._ollama_ok)

    def close(self) -> None:
        """Shut down the internal thread pool cleanly."""
        self._executor.shutdown(wait=False)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _should_try_ollama(self) -> bool:
        """
        Return True if Ollama appears available.
        After the first successful call: always True.
        After the first failure: always False (stops retrying to avoid latency).
        On first call (None): probe and record the result.
        """
        if self._ollama_ok is not None:
            return self._ollama_ok

        # Probe: can we even import the ollama library?
        try:
            import ollama  # noqa: F401
            self._ollama_ok = True
        except ImportError:
            print(
                "[Classifier] ollama library not installed — using keyword fallback.\n"
                "[Classifier]   Install: pip install ollama\n"
                "[Classifier]   Then:    ollama pull llama3.1:8b"
            )
            self._ollama_ok = False

        return self._ollama_ok

    def _classify_ollama(self, title: str) -> str:
        """Call Ollama with a timeout. Fall back to keywords on any failure."""
        prompt = self._build_prompt(title)
        try:
            future   = self._executor.submit(self._ollama_generate, prompt)
            response = future.result(timeout=self._timeout)
            category = self._parse_response(response)
            if category:
                return category
            return _classify_keywords(title)

        except concurrent.futures.TimeoutError:
            # LLM too slow this call — don't penalise future calls
            return _classify_keywords(title)

        except Exception as exc:
            # Ollama crashed, not running, etc. — disable for the session
            print(f"[Classifier] Ollama error: {exc}. Switching to keyword fallback.")
            self._ollama_ok = False
            return _classify_keywords(title)

    def _ollama_generate(self, prompt: str) -> str:
        """Blocking Ollama call — runs in thread pool."""
        import ollama
        response = ollama.generate(
            model=self._model,
            prompt=prompt,
            options={"num_predict": 8, "temperature": 0},
        )
        # API returns either a dict or an object with a .response attribute
        if isinstance(response, dict):
            return response.get("response", "")
        return getattr(response, "response", "")

    @staticmethod
    def _build_prompt(title: str) -> str:
        cats = " | ".join(CATEGORIES)
        return (
            f"You are an activity classifier for a productivity monitor.\n"
            f"Reply with ONLY one category from this exact list:\n"
            f"{cats}\n\n"
            f"Window title: {title}\n"
            f"Category:"
        )

    @staticmethod
    def _parse_response(raw: str) -> str | None:
        """
        Extract the category from the LLM response.
        Returns None if the response cannot be mapped to a known category.
        """
        text = raw.strip()

        # Exact match (case-insensitive)
        for cat in CATEGORIES:
            if text.lower() == cat.lower():
                return cat

        # Partial match — accept if a category name appears inside the response
        for cat in CATEGORIES:
            if cat.lower() in text.lower():
                return cat

        return None


# ─────────────────────────────────────────────────────────────────────────────
# KEYWORD FALLBACK (module-level, also usable standalone)
# ─────────────────────────────────────────────────────────────────────────────

def _classify_keywords(window_title: str) -> str:
    """
    Rule-based classification of a window title.

    Evaluates _KEYWORD_RULES in order; the first rule whose keywords appear
    in the lowercased title wins. Returns 'Other' if no rule matches.
    """
    lower = window_title.lower()
    for category, keywords in _KEYWORD_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return "Other"


# ── Convenience top-level function (used by other modules) ───────────────────

_default_classifier: WindowClassifier | None = None


def classify(window_title: str) -> str:
    """
    Module-level shortcut: classify a title using the shared classifier instance.
    Creates the instance on first call.
    """
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = WindowClassifier()
    return _default_classifier.classify(window_title)


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_TITLES = [
        ("DMoforis/msc-thesis — GitHub — Google Chrome",   "Software Development"),
        ("Overleaf — LaTeX editor",                        "Academic Work"),
        ("Gmail — Inbox",                                  "Communication"),
        ("Netflix — Stranger Things",                      "Entertainment"),
        ("YouTube — Lofi Hip Hop",                         "Social Media"),
        ("Microsoft Word — thesis_draft.docx",             "Document Editing"),
        ("Wikipedia — Remote sensing",                     "Research / Reading"),
        ("Task Manager",                                   "System / Utility"),
        ("Facebook",                                       "Social Media"),
        ("Untitled — Notepad",                             "Document Editing"),
        ("SomeRandomApp",                                  "Other"),
    ]

    clf = WindowClassifier()
    print(f"Classifier mode: {'LLM (Ollama)' if clf.using_llm else 'keyword fallback'}\n")
    print(f"{'Title':<52} {'Predicted':<25} {'Expected'}")
    print("-" * 90)

    all_pass = True
    for title, expected in TEST_TITLES:
        predicted = clf.classify(title)
        ok        = "PASS" if predicted == expected else "FAIL"
        if predicted != expected:
            all_pass = False
        print(f"{title:<52} {predicted:<25} {expected}  {ok}")

    print()
    print("All tests passed." if all_pass else "Some tests failed — check keyword rules.")
    clf.close()
