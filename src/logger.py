"""
Project Logger
--------------
Provides a centralized logger for the project.
"""

import logging
from pathlib import Path

from src.config import BASE_DIR

# ======================================================
# Logs Directory
# ======================================================

LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "project.log"

# ======================================================
# Logger Configuration
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)