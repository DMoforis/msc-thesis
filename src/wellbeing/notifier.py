"""
notifier.py
-----------
Windows desktop notification delivery with DB logging.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Library: windows-toasts (pip install windows-toasts)
  Native Windows 10/11 toast notifications via the WinRT API.
  Falls back to console print if windows-toasts is unavailable.

Every notification sent is persisted to the interventions table so the
pilot study can analyse what was shown, when, and how effective it was
(correlated with user_feedback ratings).

Usage
-----
  notifier = WindowsNotifier()
  notifier.send(
      title="Time for a break",
      message="You have been focused for 52 minutes. A short walk would help.",
  )
"""

import os
import sys
from datetime import datetime

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import (
    DB_PATH,
    MIN_MINUTES_BETWEEN_NOTIFS,
    MIN_MINUTES_BETWEEN_FLOW_NOTIFS,
)
from src.utils.db import open_db, ensure_interventions_table


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWS NOTIFIER
# ─────────────────────────────────────────────────────────────────────────────

class WindowsNotifier:
    """
    Sends Windows desktop notifications and logs them to the interventions table.

    Uses windows-toasts for native WinRT toasts on Windows 10/11.
    Falls back to console print if the library is not installed.

    Cooldown is enforced here as a second layer of defence (the aggregator also
    tracks it, but the notifier is the last gate before delivery).

    Parameters
    ----------
    app_name : str
        Display name shown in the notification and Windows Action Centre.
    db_path : str
        Path to the shared SQLite database for intervention logging.
    """

    def __init__(
        self,
        app_name: str = "Stress Monitor",
        db_path:  str = DB_PATH,
    ):
        self._app_name = app_name
        self._db_path  = db_path
        self._last_sent: datetime | None = None
        self._backend, self._toaster, self._Toast = self._init_backend()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_backend(self) -> tuple[str, object | None, type | None]:
        """
        Try to initialise windows-toasts; fall back to console print.
        Returns (backend_name, toaster_instance, Toast_class).

        InteractableWindowsToaster is preferred over WindowsToaster because it
        registers the app under its AUMID and has better delivery guarantees
        when called from a subprocess context (e.g. spawned by run_all.py).
        ToastButton is imported so notification actions can be added later.
        """
        try:
            from windows_toasts import InteractableWindowsToaster, Toast, ToastButton  # noqa: F401
            print(f"[Notifier] Initialising InteractableWindowsToaster(app_name={self._app_name!r})")
            toaster = InteractableWindowsToaster(self._app_name)
            print(f"[Notifier] Toaster ready: {toaster!r}")
            return "windows_toasts", toaster, Toast
        except Exception as exc:
            print(f"[Notifier] windows-toasts unavailable ({exc}), "
                  "using console fallback.")
            return "print", None, None

    # ── Public API ────────────────────────────────────────────────────────────

    def send(
        self,
        title:          str,
        message:        str,
        timeout:        int          = 10,
        trigger_reason: str  | None  = None,
        stress_index:   float | None = None,
        valence:        float | None = None,
        arousal:        float | None = None,
    ) -> bool:
        """
        Display a desktop notification and log it to the interventions table.

        Parameters
        ----------
        title          : Notification headline (keep to ≤ 64 chars)
        message        : Notification body (keep to ≤ 200 chars)
        timeout        : Retained for API compatibility; Windows controls
                         toast duration via system settings
        trigger_reason : Why this notification was triggered (for logs)
        stress_index   : Current stress score for logging
        valence        : Current valence for logging
        arousal        : Current arousal for logging

        Returns
        -------
        True if delivered, False if suppressed by cooldown or on error.
        """
        if not self._cooldown_ok(trigger_reason):
            return False

        delivered = self._deliver(title, message)
        self._log(title, message, trigger_reason, stress_index, valence, arousal, delivered)

        if delivered:
            self._last_sent = datetime.now()

        return delivered

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cooldown_ok(self, trigger_reason: str | None = None) -> bool:
        """
        Return True if enough time has elapsed since the last notification.

        Uses the trigger-type-specific cooldown so that a positive_flow
        notification does not consume the standard 15-minute cooldown window.
        This keeps the notifier in sync with the aggregator's per-type logic.
        """
        if self._last_sent is None:
            return True
        threshold = (
            MIN_MINUTES_BETWEEN_FLOW_NOTIFS
            if trigger_reason == "positive_flow"
            else MIN_MINUTES_BETWEEN_NOTIFS
        )
        elapsed = (datetime.now() - self._last_sent).total_seconds() / 60.0
        return elapsed >= threshold

    def _deliver(self, title: str, message: str) -> bool:
        """Dispatch to the available notification backend."""
        try:
            if self._backend == "windows_toasts":
                return self._deliver_windows_toasts(title, message)
            return self._deliver_print(title, message)
        except Exception as exc:
            print(f"[Notifier] Delivery error ({self._backend}): {exc}")
            return self._deliver_print(title, message)

    def _deliver_windows_toasts(self, title: str, message: str) -> bool:
        """
        Dispatch a toast via InteractableWindowsToaster running in a dedicated
        daemon thread.

        Running show_toast() in its own thread avoids COM apartment conflicts
        that arise when this code is called from a subprocess spawned by
        run_all.py (the calling thread may not have a WinRT message pump).
        thread.join(timeout=3) ensures we don't block the aggregator loop
        and gives the OS enough time to accept the toast.

        After the join we check two outcomes:
          - exc_holder populated → the API raised; fall back to balloon
          - thread still alive   → show_toast() hung; Focus Assist likely
                                   suppressed the popup, fall back to balloon
          - otherwise            → delivered successfully
        """
        import threading
        import traceback

        toast = self._Toast()
        toast.text_fields = [title, message]

        print(f"[Notifier] show_toast() call:")
        print(f"  app_name   : {self._app_name!r}")
        print(f"  toaster    : {self._toaster!r}")
        print(f"  Toast type : {type(toast).__name__}")
        print(f"  text_fields: {toast.text_fields!r}")

        exc_holder: list[Exception | None] = [None]

        def _show(toaster, t):
            try:
                toaster.show_toast(t)
            except Exception as e:
                exc_holder[0] = e

        thread = threading.Thread(target=_show, args=(self._toaster, toast), daemon=True)
        thread.start()
        thread.join(timeout=3)

        if exc_holder[0] is not None:
            exc = exc_holder[0]
            print(f"[Notifier] show_toast() raised {type(exc).__name__}: {exc}")
            print("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            return self._balloon_fallback(title, message)

        if thread.is_alive():
            # Thread did not finish within 3 s — WinRT call is stuck,
            # which typically means Focus Assist silently swallowed the toast.
            print("[Notifier] Toast may have been suppressed by Focus Assist")
            return self._balloon_fallback(title, message)

        print(f"[Notifier] Toast delivered successfully: {title!r}")
        return True

    @staticmethod
    def _balloon_fallback(title: str, message: str) -> bool:
        """
        Last-resort audio cue + console log when the toast popup is suppressed.

        MessageBeep(0) plays the system default alert sound so the user gets
        an audible signal even if Focus Assist blocked the visual popup.
        The message is also printed so it is never silently lost.
        """
        import ctypes
        ctypes.windll.user32.MessageBeep(0)
        print(f"[Notifier] Fallback: {title} — {message}")
        return True

    @staticmethod
    def _deliver_print(title: str, message: str) -> bool:
        """Last-resort fallback: print to console so the message is never lost."""
        border = "=" * 60
        print(f"\n{border}")
        print(f"  WELL-BEING NOTIFICATION")
        print(f"  {title}")
        print(f"  {message}")
        print(f"{border}\n")
        return True

    def _log(
        self,
        title:          str,
        message:        str,
        trigger_reason: str  | None,
        stress_index:   float | None,
        valence:        float | None,
        arousal:        float | None,
        delivered:      bool,
    ) -> None:
        """Persist the notification to the interventions table."""
        try:
            conn = open_db(self._db_path)
            ensure_interventions_table(conn)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            full_message = f"{title}: {message}"
            conn.execute("""
                INSERT INTO interventions
                    (timestamp, trigger_reason, message, stress_index,
                     valence, arousal, delivered)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ts, trigger_reason, full_message,
                  stress_index, valence, arousal, int(delivered)))
            conn.commit()
            conn.close()
        except Exception as exc:
            print(f"[Notifier] DB log error: {exc}")


# ── Convenience module-level function ────────────────────────────────────────

_default_notifier: WindowsNotifier | None = None


def send_notification(
    title:          str,
    message:        str,
    timeout:        int          = 10,
    trigger_reason: str  | None  = None,
    stress_index:   float | None = None,
    valence:        float | None = None,
    arousal:        float | None = None,
) -> bool:
    """Module-level shortcut — creates the shared instance on first call."""
    global _default_notifier
    if _default_notifier is None:
        _default_notifier = WindowsNotifier()
    return _default_notifier.send(
        title, message, timeout, trigger_reason, stress_index, valence, arousal
    )


# ── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    n = WindowsNotifier()
    n.send("Test", "Direct notifier test", trigger_reason="high_stress")
