#!/usr/bin/env bash
set -euo pipefail

BUCKET="${GCS_DATA_BUCKET:-restaurant-success-data-gen-lang-client-0301771744}"
SRC="gs://${BUCKET}/data/processed"
DEST="data/processed"

if ! command -v gsutil >/dev/null 2>&1; then
  echo "gsutil is required (install the Google Cloud SDK)." >&2
  exit 1
fi

mkdir -p "$DEST"

echo "Syncing ${SRC} -> ${DEST} ..."
gsutil -m rsync "$SRC" "$DEST"
echo "Done."
