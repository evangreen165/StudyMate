#!/bin/sh
set -e
echo "entrypoint: checking volumes..."
if [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
    echo "chroma_db empty -> seeding from image"
    cp -r /app/_seed/chroma_db/* /app/chroma_db/ 2>/dev/null || true
fi
if [ -z "$(ls -A /app/data/datasets 2>/dev/null)" ]; then
    echo "data/datasets empty -> seeding from image"
    cp -r /app/_seed/datasets/* /app/data/datasets/ 2>/dev/null || true
fi
echo "entrypoint: starting services"
exec "$@"