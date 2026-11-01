"""
MEMOS — FastAPI Application Entry Point
Memory Evolution and Management for Organized Semantic Storage
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, health, memories
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.database.mongo import close_database, get_database
from app.database.neo4j_db import close_neo4j, init_neo4j

# ── Bootstrap ───────────────────────────────────────────────
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# ── FastAPI App ─────────────────────────────────────────────
app = FastAPI(
    title="MEMOS API",
    description="Memory Evolution and Management for Organized Semantic Storage",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(memories.router, prefix="/api", tags=["Memories"])


# ── Lifecycle ───────────────────────────────────────────────
@app.on_event("startup")
async def startup() -> None:
    logger.info("MEMOS starting up (env=%s)...", settings.app_env)
    # Init MongoDB
    try:
        await get_database()
        logger.info("[+] MongoDB connected")
    except Exception as e:
        logger.error("[-] MongoDB connection failed: %s", e)

    # Init ChromaDB (lazy — initialized on first use)
    try:
        from app.database.chroma import get_collection
        col = get_collection()
        logger.info("[+] ChromaDB ready (collection=%s, count=%d)", col.name, col.count())
    except Exception as e:
        logger.error("[-] ChromaDB init failed: %s", e)

    # Init Neo4j (optional)
    neo4j_ok = await init_neo4j()
    if neo4j_ok:
        logger.info("[+] Neo4j connected")
    else:
        logger.info("[*] Neo4j disabled or unavailable: graph features degraded")

    # Pre-warm embedding model
    try:
        from app.llm.embeddings import embed_text
        _ = embed_text("warmup")
        logger.info("[+] Embedding model loaded")
    except Exception as e:
        logger.warning("[*] Embedding model warmup failed: %s", e)

    logger.info("MEMOS ready on port %d", settings.backend_port)


@app.on_event("shutdown")
async def shutdown() -> None:
    logger.info("MEMOS shutting down...")
    await close_database()
    await close_neo4j()
    logger.info("MEMOS shutdown complete.")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "MEMOS API",
        "version": "0.1.0",
        "description": "Memory Evolution and Management for Organized Semantic Storage",
        "docs": "/docs",
        "health": "/health",
    }
