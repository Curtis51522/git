from dotenv import load_dotenv
load_dotenv()
import logging
import asyncio
from contextlib import asynccontextmanager, suppress

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import os, sys
import httpx
from api.auth import get_current_user, require_manager, validate_auth_configuration
from api.operation_clock import business_date_is_fixed
from config import settings as cfg

validate_auth_configuration()

# Windows: use SelectorEventLoop to avoid Proactor connection-reset noise
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")


@asynccontextmanager
async def lifespan(_app):
    from api.freshness_service import update_all_freshness

    maintenance_paused = business_date_is_fixed()
    if not maintenance_paused:
        update_all_freshness()

    async def periodic_freshness():
        while True:
            await asyncio.sleep(1800)
            try:
                update_all_freshness()
            except Exception:
                logger.exception("Periodic freshness update failed")

    task = None if maintenance_paused else asyncio.create_task(periodic_freshness())
    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(
    title="Bakery AI System",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "api", "module4_frontend", "static")


@app.middleware("http")
async def prevent_stale_ui_assets(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in {
        "/",
        "/index.html",
        "/finesse-ui.css",
        "/login-cinematic.css",
    }:
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response



@app.get("/ping")
async def ping():
    return {"ok": True}


@app.get("/health")
def health():
    from db.mysql_client import get_db

    db = get_db()
    cursor = None
    try:
        cursor = db.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            db.close()
    return {"status": "ok", "database": "ok"}


@app.get("/s5-health")
async def s5_health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{cfg.S5_BASE_URL}/health")
    except httpx.HTTPError:
        response = None

    if response is None or response.status_code != 200:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "s5": "unavailable"},
        )
    return {"status": "ok", "s5": "ok"}



# Freshness update endpoint
@app.post("/freshness/update", dependencies=[Depends(require_manager)])
async def update_freshness():
    from api.freshness_service import update_all_freshness
    result = update_all_freshness()
    return result

@app.get("/freshness/discounts", dependencies=[Depends(get_current_user)])
async def get_discounts():
    from api.freshness_service import DISCOUNT_MAP, FRESHNESS_COLORS
    return {"discounts": DISCOUNT_MAP, "colors": FRESHNESS_COLORS}

from api.module1_yolo import router as s1_router
from api.module2_forecast import router as s2_router
from api.module3_scheduling import router as s3_router
from api.module4_frontend.bff import router as s4_router
# S5 proxy: forward all /s5/* requests to AI Brain (:8001)
@app.api_route(
    "/s5/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_s5(
    path: str,
    request: Request,
    user=Depends(get_current_user),
):
    is_staff_discount_request = (
        request.method == "POST" and path.strip("/") == "discounts"
    )
    if user.get("role") != "manager" and not is_staff_discount_request:
        raise HTTPException(403, "Manager only")

    hop_by_hop_headers = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    request_excluded_headers = hop_by_hop_headers | {"host", "content-length"}
    response_excluded_headers = hop_by_hop_headers | {
        "content-encoding",
        "content-length",
    }

    url = f"{cfg.S5_BASE_URL}/{path.lstrip('/')}"
    if request.query_params:
        url += f"?{request.query_params}"
    body = await request.body()
    try:
        async with httpx.AsyncClient(timeout=210) as client:
            resp = await client.request(
                method=request.method,
                url=url,
                content=body if body else None,
                headers={
                    key: value
                    for key, value in request.headers.items()
                    if key.lower() not in request_excluded_headers
                },
                timeout=210,
            )
    except httpx.RequestError as exc:
        raise HTTPException(503, "S5 service is unavailable") from exc

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers={
            key: value
            for key, value in resp.headers.items()
            if key.lower() not in response_excluded_headers
        },
    )

app.include_router(s1_router)
app.include_router(s2_router)
app.include_router(s3_router)
app.include_router(s4_router)

app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="warning")




