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
        """
        try:
            from windows_toasts import WindowsToaster, Toast
            toaster = WindowsToaster(self._app_name)
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
        toast = self._Toast()
        toast.text_fields = [title, message]
        self._toaster.show_toast(toast)
        print(f"[Notifier] Sent via windows-toasts: {title!r}")
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
    notifier = WindowsNotifier()
    print(f"Notification backend: {notifier._backend}\n")

    ok = notifier.send(
        title          = "Time for a break",
        message        = "You have been focused for 52 minutes. A short walk would help.",
        timeout        = 8,
        trigger_reason = "high_stress",
        stress_index   = 0.71,
        valence        = -0.38,
        arousal        = 0.55,
    )
    print(f"Delivered: {ok}")

    # Verify it was logged
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT timestamp, trigger_reason, message, delivered "
        "FROM interventions ORDER BY id DESC LIMIT 3"
    ).fetchall()
    print("\nLast interventions in DB:")
    for r in rows:
        print(f"  {r[0]}  [{r[1]}]  delivered={r[3]}  {r[2][:60]}...")
    conn.close()
