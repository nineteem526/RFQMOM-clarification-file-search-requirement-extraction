FROM python:3.11-slim-bookworm AS build

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git libgl1 libglib2.0-0 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY environment-setup/required-dependencies.in \
     environment-setup/upstream-constraints.txt \
     environment-setup/appservice-runtime-requirements.txt \
     environment-setup/ocr-linux-requirements.txt \
     /build/environment-setup/

RUN mkdir -p /opt/runtime/kotaemon \
    && git -C /opt/runtime/kotaemon init \
    && git -C /opt/runtime/kotaemon remote add origin https://github.com/Cinnamon/kotaemon.git \
    && git -C /opt/runtime/kotaemon fetch --depth 1 origin 9ad3e4e49aa35b8acddd235918a5d9753c1cfdf9 \
    && git -C /opt/runtime/kotaemon checkout --detach FETCH_HEAD \
    && python -m venv /opt/runtime/.venv \
    && /opt/runtime/.venv/bin/python -m pip install --no-cache-dir \
        -c /build/environment-setup/upstream-constraints.txt \
        -r /build/environment-setup/required-dependencies.in \
    && /opt/runtime/.venv/bin/python -m pip install --no-cache-dir \
        -c /build/environment-setup/upstream-constraints.txt \
        /opt/runtime/kotaemon/libs/kotaemon \
        /opt/runtime/kotaemon/libs/ktem \
    && /opt/runtime/.venv/bin/python -m pip install --no-cache-dir \
        -c /build/environment-setup/upstream-constraints.txt \
        -r /build/environment-setup/appservice-runtime-requirements.txt \
    && /opt/runtime/.venv/bin/python -m pip check \
    && python -m venv /opt/rfq-ocr/.venv \
    && /opt/rfq-ocr/.venv/bin/python -m pip install --no-cache-dir \
        -r /build/environment-setup/ocr-linux-requirements.txt

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/runtime/.venv/bin:${PATH}" \
    RFQ_RUNTIME_HOME=/home/rfq-data \
    RFQ_RUNTIME_VENV=/opt/runtime/.venv \
    RFQ_NLTK_DATA=/opt/rfq-nltk \
    RFQ_TIKTOKEN_DATA=/opt/rfq-tiktoken \
    RFQ_EMBEDDING_DIR=/opt/rfq-model \
    RFQ_OCR_ROOT=/opt/rfq-ocr \
    RFQ_AUTHORIZED_SOURCE=/home/rfq-source \
    RFQ_APP_SERVICE=true \
    RFQ_SINGLE_PROJECT_MODE=true \
    RFQ_PROJECT_ID=rfq-poc \
    WEBSITES_PORT=8000 \
    GRADIO_ANALYTICS_ENABLED=False \
    ANONYMIZED_TELEMETRY=False \
    HAYSTACK_TELEMETRY_ENABLED=False \
    OTEL_SDK_DISABLED=true \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /opt/runtime /opt/runtime
COPY --from=build /opt/rfq-ocr /opt/rfq-ocr
COPY *.py project_ui.css project_ui.js entrypoint.sh /app/
COPY environment-setup/embedding-download-manifest.json \
     environment-setup/tiktoken-cache-manifest.json /app/environment-setup/
COPY .appservice-assets/embedding/ /opt/rfq-model/
COPY .appservice-assets/nltk/ /opt/rfq-nltk/
COPY .appservice-assets/tiktoken/ /opt/rfq-tiktoken/
COPY --from=build /opt/runtime/.venv/lib/python3.11/site-packages/llama_index/core/_static/nltk_cache/corpora/stopwords/ /opt/rfq-nltk/corpora/stopwords/
COPY environment-setup/ocr-model-manifest.json /opt/rfq-ocr/model-manifest.json

RUN chmod 755 /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
