from __future__ import annotations

import logging
import os
import uvicorn
from fastapi import FastAPI
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
from src.api.routes import router
from src.core.config import Paths, MONGO_URI, MONGO_DB, MONGO_PRODUCTS_COL, MONGO_RECIPES_COL

from src.services.nlu_engine_onnx import NLUEngineONNX
from src.infrastructure.mongo_repositories import MongoRecipeRepository, MongoProductRepository
from src.services.search_engine import HybridSearchEngine
from src.application.usecases import SearchRecipes, EstimatePrice, BuildCart, GetRecipeDetail
from src.application.cart_planner import CartPlanner

from src.infrastructure.session_store import InMemorySessionStore
from src.application.rule_engine import RuleEngine
from src.application.dialogue_manager import DialogueManager

log = logging.getLogger("app")
app = FastAPI(title="Smart Food Bot")

_mongo_client: MongoClient | None = None


@app.on_event("startup")
def on_startup() -> None:
    global _mongo_client

    nlu_engine = NLUEngineONNX(
        model_dir=Paths.MODEL_OUT_DIR,
        onnx_relpath=os.path.join("onnx", "phobert_joint_nlu.onnx"),
        providers=["CPUExecutionProvider"],
    )

    _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    db = _mongo_client[MONGO_DB]
    recipe_repo = MongoRecipeRepository(db[MONGO_RECIPES_COL])
    product_repo = MongoProductRepository(db[MONGO_PRODUCTS_COL])

    search_engine = HybridSearchEngine(recipe_repo)
    search_uc = SearchRecipes(search_engine)
    price_uc = EstimatePrice(product_repo)
    cart_uc = BuildCart(product_repo)

    recipe_detail_uc = GetRecipeDetail(recipe_repo)
    cart_planner = CartPlanner(recipe_repo=recipe_repo, product_repo=product_repo)

    sessions = InMemorySessionStore(ttl_seconds=1800)
    rules = RuleEngine()

    dialogue_manager = DialogueManager(
        sessions=sessions,
        rule_engine=rules,
        nlu=nlu_engine,
        search_uc=search_uc,
        recipe_detail_uc=recipe_detail_uc,
        cart_planner=cart_planner,
    )

    # DI for routes.py
    app.state.dialogue_manager = dialogue_manager
    app.state.cart_planner = cart_planner

    # optional
    app.state.nlu_engine = nlu_engine
    app.state.search_uc = search_uc
    app.state.price_uc = price_uc
    app.state.cart_uc = cart_uc

    app.include_router(router)
    log.info("Startup complete")


@app.on_event("shutdown")
def on_shutdown() -> None:
    global _mongo_client
    if _mongo_client:
        _mongo_client.close()


    if __name__ == "__main__":
        uvicorn.run("main:app", host="0.0.0.0", port=8081, reload=False)
