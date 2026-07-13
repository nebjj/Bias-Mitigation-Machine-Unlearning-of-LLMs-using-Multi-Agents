"""
logger.py — Append-safe CSV logger for pipeline run data.
"""

import csv
import logging
import os

logger = logging.getLogger(__name__)

LOG_FILE = "logs.csv"

# Fixed column order guarantees CSV alignment even if callers
# pass dicts with keys in different orders.
FIELDNAMES = ["query", "bias", "brs", "action", "modified_query", "evaluation"]


def log_data(data: dict) -> None:
    """
    Append *data* as one row to LOG_FILE.

    - Creates the file with a header on first write.
    - Silently drops unknown keys (extra keys in *data* are ignored).
    - Logs a warning instead of crashing on I/O errors.
    """
    try:
        file_exists = os.path.isfile(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=FIELDNAMES,
                extrasaction="ignore",   # drop unexpected keys safely
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)

    except OSError as exc:
        logger.warning("Failed to write log entry: %s", exc)