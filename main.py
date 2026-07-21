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
from api.operation_clock import operation_time_scope, parse_operation_time

validate_auth_configuration()

# Windows: use SelectorEventLoop to avoid Proactor connection-reset noise
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")


@asynccontextmanager
async def lifespan(_app):
    from api.freshness_service import update_all_freshness

    replay_enabled = os.getenv("BAKERY_OPERATION_REPLAY") == "1"
    if not replay_enabled:
        update_all_freshness()

    async def periodic_freshness():
        while True:
            await asyncio.sleep(1800)
            try:
                update_all_freshness()
            except Exception:
                logger.exception("Periodic freshness update failed")

    task = None if replay_enabled else asyncio.create_task(periodic_freshness())
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


@app.middleware("http")
async def apply_operation_time(request: Request, call_next):
    selected_value = request.headers.get("X-Operation-At")
    if not selected_value:
        return await call_next(request)

    try:
        user = await get_current_user(request)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    if user.get("role") != "manager":
        return JSONResponse(status_code=403, content={"detail": "Manager only"})

    try:
        selected = parse_operation_time(selected_value)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    with operation_time_scope(selected):
        return await call_next(request)


BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "api", "module4_frontend", "static")



@app.get("/ping")
async def ping():
    return {"ok": True}



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

    async with httpx.AsyncClient(timeout=210) as client:
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
            timeout=210,
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




