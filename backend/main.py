# backend/main.py
from __future__ import annotations

import importlib
import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

APP_NAME = "CST Hub Backend (Monolith)"
APP_VERSION = "1.0.0"

# -------------------------------------------------------------------
# Create app (must happen BEFORE any app.* calls)
# -------------------------------------------------------------------
app = FastAPI(
    title=APP_NAME,
    description="One FastAPI app exposing 8 project APIs under /api/p1 ... /api/p8",
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -------------------------------------------------------------------
# CORS (Vercel rewrites front this, but you can restrict if desired)
# Set FRONTEND_ORIGIN to your Vercel URL to lock this down.
# -------------------------------------------------------------------
FRONTEND = os.getenv("FRONTEND_ORIGIN")  # e.g. "https://your-app.vercel.app"
allow_origins = ["*"] if not FRONTEND else [FRONTEND]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# Helpers: include routers defensively (some may not exist yet)
# Each router file must define: router = APIRouter(prefix="/api/pX", ...)
# -------------------------------------------------------------------
def try_include(router_module: str, mounted: List[str]) -> None:
    """Attempt to import routers.pX and include its router."""
    try:
        mod = importlib.import_module(router_module)
        router = getattr(mod, "router", None)
        if router is None:
            print(f"[main] {router_module} has no 'router' attribute; skipping.")
            return
        app.include_router(router)
        prefix = getattr(router, "prefix", "<unknown>")
        mounted.append(prefix)
        print(f"[main] Mounted router {router_module} at prefix {prefix}")
    except ModuleNotFoundError:
        print(f"[main] Router module not found: {router_module} (ok to skip)")
    except Exception as e:
        print(f"[main] Failed to mount {router_module}: {e}")

mounted_prefixes: List[str] = []
for mod_name in (
    "routers.p1", "routers.p2", "routers.p3", "routers.p4",
    "routers.p5", "routers.p6", "routers.p7", "routers.p8",
):
    try_include(mod_name, mounted_prefixes)

# -------------------------------------------------------------------
# Basic routes
# -------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "ok": True,
        "service": "cst-hub-monolith",
        "version": APP_VERSION,
        "health": "/api/health",
        "docs": "/docs",
        "mounted": mounted_prefixes,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cst-hub-monolith", "mounted": mounted_prefixes}

# -------------------------------------------------------------------
# Optional: Local run
# Railway will run with: uvicorn main:app --host 0.0.0.0 --port $PORT
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
