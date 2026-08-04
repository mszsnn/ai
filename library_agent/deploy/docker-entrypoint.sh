#!/bin/sh
set -eu

mkdir -p "$VECTOR_DB_PATH" "$UPLOAD_DIR"

# The repository's Chroma collection is a seed only. Once the named /data
# volume contains chroma.sqlite3, never overwrite it on container restart.
if [ ! -f "$VECTOR_DB_PATH/chroma.sqlite3" ] && [ -d "/app/library_agent/rag_engine/db_vector_data" ]; then
  cp -a /app/library_agent/rag_engine/db_vector_data/. "$VECTOR_DB_PATH/"
fi

exec "$@"
