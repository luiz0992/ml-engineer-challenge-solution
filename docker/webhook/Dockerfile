FROM python:3.12-slim-bookworm
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --chown=app:app server.py ./server.py
USER app
EXPOSE 8089
ENV WEBHOOK_LOG=/var/tmp/alerts.jsonl
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8089/health')"
CMD ["python", "server.py"]
