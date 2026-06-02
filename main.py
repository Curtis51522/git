from dotenv import load_dotenv
load_dotenv()
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
import os, sys
import httpx

app = FastAPI(title="Bakery AI System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



import asyncio

# Windows: use SelectorEventLoop to avoid Proactor connection-reset noise
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")  # suppress scipy Ctrl+C noise

@app.on_event("startup")
async def startup_freshness():
    """Run freshness update on server start + schedule periodic updates."""
    from api.freshness_service import update_all_freshness
    update_all_freshness()
    
    async def periodic_freshness():
        while True:
            await asyncio.sleep(1800)  # every 30 minutes
            try:
                update_all_freshness()
            except Exception:
                pass

    
    asyncio.create_task(periodic_freshness())


BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "api", "module4_frontend", "static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/ping")
async def ping():
    return {"ok": True}



# Freshness update endpoint
@app.post("/freshness/update")
async def update_freshness():
    from api.freshness_service import update_all_freshness
    result = update_all_freshness()
    return result

@app.get("/freshness/discounts")
async def get_discounts():
    from api.freshness_service import DISCOUNT_MAP, FRESHNESS_COLORS
    return {"discounts": DISCOUNT_MAP, "colors": FRESHNESS_COLORS}

from api.module1_yolo import router as s1_router
from api.module2_forecast import router as s2_router
from api.module3_scheduling import router as s3_router
from api.module4_frontend.bff import router as s4_router
# S5 proxy: forward all /s5/* requests to AI Brain (:8001)
@app.api_route("/s5/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_s5(path: str, request: Request):
    async with httpx.AsyncClient() as client:
        url = f"http://localhost:8001/{path}"
        if request.query_params:
            url += f"?{request.query_params}"
        body = await request.body()
        resp = await client.request(
            method=request.method,
            url=url,
            content=body if body else None,
            headers={k: v for k, v in request.headers.items()
                      if k.lower() not in ("host", "content-length", "transfer-encoding")},
            timeout=30,
        )
        return Response(content=resp.content, status_code=resp.status_code,
                        headers={k: v for k, v in resp.headers.items()
                                 if k.lower() not in ("transfer-encoding", "content-encoding")})

app.include_router(s1_router)
app.include_router(s2_router)
app.include_router(s3_router)
app.include_router(s4_router)

if __name__ == "__main__":
    import hypercorn.asyncio
    config = hypercorn.Config()
    config.bind = ["0.0.0.0:8002"]
    config.keep_alive_timeout = 300
    config.graceful_timeout = 300
    config.read_timeout = 300
    asyncio.run(hypercorn.asyncio.serve(app, config))



