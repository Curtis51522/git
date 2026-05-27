from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse, FileResponse
import os, traceback

app = FastAPI(title="Bakery AI System", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



import asyncio

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

    # B1 Proactive Monitor
    from api.module5_agent.alert_store import init_alerts_table
    init_alerts_table()

    async def start_b1_monitor():
        await asyncio.sleep(10)  # let other services init first
        from api.module5_agent.monitor import start_monitor
        await start_monitor(interval_sec=30*60)  # every 30 min
    
    asyncio.create_task(start_b1_monitor())

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "api", "module4_frontend", "static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/test_js.html", response_class=HTMLResponse)
async def test_js():
    with open(os.path.join(STATIC, "test_js.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/login_test.html", response_class=HTMLResponse)
async def login_test():
    with open(os.path.join(STATIC, "login_test.html"), encoding="utf-8") as f:
        return f.read()

@app.get("/app.js")
async def app_js():
    with open(os.path.join(STATIC, "app.js"), encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type="text/javascript", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

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
from api.module5_agent.router import router as s5_router

app.include_router(s1_router)
app.include_router(s2_router)
app.include_router(s3_router)
app.include_router(s4_router)
app.include_router(s5_router)

if __name__ == "__main__":
    import hypercorn.asyncio, asyncio
    config = hypercorn.Config()
    config.bind = ["0.0.0.0:8000"]
    config.keep_alive_timeout = 300
    config.graceful_timeout = 300
    config.read_timeout = 300
    asyncio.run(hypercorn.asyncio.serve(app, config))



