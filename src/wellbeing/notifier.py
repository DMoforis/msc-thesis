"""
notifier.py
-----------
Windows desktop notification delivery with DB logging.
MSc Thesis — Dimitris Moforis, University of Piraeus, Dept. of Digital Systems

Library choice: plyer (cross-platform, active maintenance, no OS-specific code)
Fallback: PowerShell New-BurntToastNotification if plyer is not installed.

Every notification sent is persisted to the interventions table so the
pilot study can analyse what was shown, when, and how effective it was
(correlated with user_feedback ratings).

Setup (one-time):
  pip install plyer

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
import subprocess
from datetime import datetime

# ── Path bootstrap ────────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils.config import DB_PATH, MIN_MINUTES_BETWEEN_NOTIFS
from src.utils.db import open_db, ensure_interventions_table


# ─────────────────────────────────────────────────────────────────────────────
# WINDOWS NOTIFIER
# ─────────────────────────────────────────────────────────────────────────────

class WindowsNotifier:
    """
    Sends Windows desktop notifications and logs them to the interventions table.

    Cooldown is enforced here as a second layer of defence (the aggregator also
    tracks it, but the notifier is the last gate before delivery).

    Parameters
    ----------
    app_name : str
        Display name shown in the notification.
    db_path : str
        Path to the shared SQLite database for intervention logging.
    """

    def __init__(
        self,
        app_name: str = "Well-being Assistant",
        db_path:  str = DB_PATH,
    ):
        self._app_name = app_name
        self._db_path  = db_path
        self._last_sent: datetime | None = None
        self._backend: str = self._detect_backend()

    # ── Public API ────────────────────────────────────────────────────────────

    def send(
        self,
        title:         str,
        message:       str,
        timeout:       int         = 10,
        trigger_reason: str | None = None,
        stress_index:  float | None = None,
        valence:       float | None = None,
        arousal:       float | None = None,
    ) -> bool:
        """
        Display a desktop notification and log it to the interventions table.

        Parameters
        ----------
        title          : str   — Notification headline (short, ≤ 64 chars)
        message        : str   — Notification body (≤ 200 chars)
        timeout        : int   — Seconds before auto-dismiss (default 10)
        trigger_reason : str   — Why this notification was triggered (for logs)
        stress_index   : float — Current stress score for logging
        valence        : float — Current valence for logging
        arousal        : float — Current arousal for logging

        Returns
        -------
        bool — True if notification was delivered, False if suppressed by cooldown
               or if delivery failed.
        """
        if not self._cooldown_ok():
            return False

        delivered = self._deliver(title, message, timeout)
        self._log(title, message, trigger_reason, stress_index, valence, arousal, delivered)

        if delivered:
            self._last_sent = datetime.now()

        return delivered

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cooldown_ok(self) -> bool:
        """Return True if enough time has elapsed since the last notification."""
        if self._last_sent is None:
            return True
        elapsed = (datetime.now() - self._last_sent).total_seconds() / 60.0
        return elapsed >= MIN_MINUTES_BETWEEN_NOTIFS

    def _detect_backend(self) -> str:
        """Return 'plyer', 'powershell', or 'print' (last resort)."""
        try:
            import plyer  # noqa: F401
            return "plyer"
        except ImportError:
            pass

        # Check whether BurntToast module is available in PowerShell
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command",
             "Get-Module -ListAvailable BurntToast | Select-Object -First 1"],
            capture_output=True, text=True,
        )
        if "BurntToast" in (result.stdout or ""):
            return "powershell"

        return "print"

    def _deliver(self, title: str, message: str, timeout: int) -> bool:
        """Dispatch to the available notification backend."""
        try:
            if self._backend == "plyer":
                return self._deliver_plyer(title, message, timeout)
            elif self._backend == "powershell":
                return self._deliver_powershell(title, message)
            else:
                return self._deliver_print(title, message)
        except Exception as exc:
            print(f"[Notifier] Delivery error ({self._backend}): {exc}")
            # Try print fallback so at least the message surfaces in the console
            return self._deliver_print(title, message)

    def _deliver_plyer(self, title: str, message: str, timeout: int) -> bool:
        from plyer import notification
        notification.notify(
            title    = title,
            message  = message,
            app_name = self._app_name,
            timeout  = timeout,
        )
        print(f"[Notifier] Sent via plyer: {title!r}")
        return True

    def _deliver_powershell(self, title: str, message: str) -> bool:
        # Escape single quotes inside the strings for PowerShell
        t = title.replace("'", "''")
        m = message.replace("'", "''")
        script = f"New-BurntToastNotification -Text '{t}', '{m}'"
        result = subprocess.run(
            ["powershell", "-NonInteractive", "-Command", script],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"[Notifier] Sent via BurntToast: {title!r}")
            return True
        print(f"[Notifier] BurntToast failed: {result.stderr.strip()}")
        return False

    @staticmethod
    def _deliver_print(title: str, message: str) -> bool:
        """Last-resort: print to console so the message is never silently lost."""
        border = "=" * 60
        print(f"\n{border}")
        print(f"  WELL-BEING NOTIFICATION")
        print(f"  {title}")
        print(f"  {message}")
        print(f"{border}\n")
        return True

    def _log(
        self,
        title:         str,
        message:       str,
        trigger_reason: str | None,
        stress_index:  float | None,
        valence:       float | None,
        arousal:       float | None,
        delivered:     bool,
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
    title:         str,
    message:       str,
    timeout:       int         = 10,
    trigger_reason: str | None = None,
    stress_index:  float | None = None,
    valence:       float | None = None,
    arousal:       float | None = None,
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
    print(f"\nDelivered: {ok}")

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
