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
from fastapi.staticfiles import StaticFiles
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
    """Run freshness update on server start."""
    from api.freshness_service import update_all_freshness
    update_all_freshness()

    async def periodic_freshness():
        while True:
            await asyncio.sleep(1800)
            try:
                update_all_freshness()
            except Exception:
                pass

    asyncio.create_task(periodic_freshness())


BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "api", "module4_frontend", "static")



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
    async with httpx.AsyncClient(timeout=120) as client:
        url = f"http://127.0.0.1:8001/{path}"
        if request.query_params:
            url += f"?{request.query_params}"
        body = await request.body()
        resp = await client.request(
            method=request.method,
            url=url,
            content=body if body else None,
            headers={k: v for k, v in request.headers.items()
                      if k.lower() not in ("host", "content-length", "transfer-encoding")},
            timeout=120,
        )
        return Response(content=resp.content, status_code=resp.status_code,
                        headers={k: v for k, v in resp.headers.items()
                                 if k.lower() not in ("transfer-encoding", "content-encoding")})

app.include_router(s1_router)
app.include_router(s2_router)
app.include_router(s3_router)
app.include_router(s4_router)

app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")




