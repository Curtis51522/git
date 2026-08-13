FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && apt-get clean

COPY requirements-runtime.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.13.0 torchvision==0.28.0 \
    && python -m pip install -r requirements-runtime.txt

COPY . .

RUN useradd --create-home --uid 10001 bakery \
    && chown -R bakery:bakery /app
USER bakery

EXPOSE 8001 8002

CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8002"]
