# smart_food_bot/src/core/config.py
from __future__ import annotations
from dataclasses import dataclass
import os
import torch
import logging

# 1650 Max-Q Optimization: prefer AMP, keep VRAM low.
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "8"))
ACCUM_STEPS: int = int(os.getenv("ACCUM_STEPS", "1"))
MAX_LEN: int = int(os.getenv("MAX_LEN", "128"))
EPOCHS: int = int(os.getenv("EPOCHS", "3"))
LR: float = float(os.getenv("LR", "3e-5"))
SEED: int = int(os.getenv("SEED", "42"))
MODEL_NAME: str = os.getenv("MODEL_NAME", "vinai/phobert-base")

INTENTS = [
    "search_recipe", "ask_recipe_detail", "refine_search",
    "add_ingredients_to_cart", "ask_price_estimate", "fallback"
]
SLOTS = [
    "O","B-DISH","I-DISH","B-INGREDIENT","I-INGREDIENT",
    "B-QUANTITY","B-UNIT","B-TASTE","B-EXCLUDE"
]

# Mongo settings
MONGO_URI: str = os.getenv("MONGO_URI","mongodb+srv://thieulk23:thieulk23@cluster0.es7pd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
MONGO_DB: str = os.getenv("MONGO_DB","test")
MONGO_RECIPES_COL: str = os.getenv("MONGO_RECIPES_COL", "recipies")
MONGO_PRODUCTS_COL: str = os.getenv("MONGO_PRODUCTS_COL", "products")

@dataclass(frozen=True)
class Paths:
    ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATA_RAW: str = os.path.join(ROOT, "data", "raw", "dataset.json")
    DATA_PROCESSED_DIR: str = os.path.join(ROOT, "data", "processed")
    MODEL_OUT_DIR: str = os.path.join(ROOT, "models", "phobert_joint_nlu")

# Global logging (module-level loggers inherit this)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
)
LOGGER = logging.getLogger("smart_food_bot")
