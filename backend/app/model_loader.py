import json
import logging
import threading
from pathlib import Path

import keras


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BACKEND_DIR / "model"
CONFIG_PATH = MODEL_DIR / "config.json"
MODEL_PATH = MODEL_DIR / "dr_model.keras"


class ModelState:
    def __init__(self):
        self.model = None
        self.config = None
        self.lock = threading.Lock()
        self.load_error = None

    def load(self):
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
                self.config = json.load(config_file)

            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

            model = keras.models.load_model(MODEL_PATH)
            with self.lock:
                self.model = model
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            logger.exception("Could not load DR model")


model_state = ModelState()
