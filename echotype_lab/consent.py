"""
echotype_lab.consent
====================
Handles user-consent prompts and consent-record logging.

All audio collection MUST be preceded by an accepted consent prompt.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONSENT_LOG = Path("data") / "consent_log.jsonl"

CONSENT_TEXT = """
╔══════════════════════════════════════════════════════════════════╗
║                  EchoType Lab – Consent Notice                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  This application will record audio from your microphone while   ║
║  you type on YOUR OWN keyboard.  The audio is stored LOCALLY on  ║
║  this machine and is NEVER transmitted anywhere.                 ║
║                                                                  ║
║  Purpose : educational / research demonstration only.            ║
║  Data    : raw audio + MFCC features saved in ./data/            ║
║  Deletion: run  echotype-record --delete-data  at any time.      ║
║                                                                  ║
║  You may withdraw consent and stop the recording at any time     ║
║  by pressing Ctrl-C.                                             ║
║                                                                  ║
║  Do NOT use this tool on keyboards you do not own or without     ║
║  the explicit consent of all parties involved.                   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def request_consent(
    log_path: Path | str = _DEFAULT_CONSENT_LOG,
    *,
    non_interactive: bool = False,
) -> bool:
    """Display the consent notice and ask the user to accept.

    Parameters
    ----------
    log_path:
        File path where accepted consent records are appended (JSONL).
    non_interactive:
        When *True* the function returns *True* immediately without
        prompting (useful in automated test environments).

    Returns
    -------
    bool
        *True* if the user accepted; *False* if they declined.
    """
    print(CONSENT_TEXT)

    if non_interactive:
        logger.info("non-interactive mode – consent assumed")
        _log_consent(log_path, accepted=True)
        return True

    try:
        answer = input("Do you consent to the above? [yes/no]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nConsent not given – exiting.")
        return False

    accepted = answer in {"yes", "y"}
    _log_consent(log_path, accepted=accepted)

    if not accepted:
        print("Consent not given – no data will be collected.")
    return accepted


def _log_consent(log_path: Path | str, *, accepted: bool) -> None:
    """Append a consent record (JSON line) to *log_path*."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "accepted": accepted,
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.debug("Consent record written to %s", log_path)
