#!/usr/bin/env bash
# Pings the JupyterHub user-activity endpoint every INTERVAL seconds so the
# pod's idle culler does not reap the user server while a long batch is
# running. Self-terminates as soon as `BATCH COMPLETE` appears in STATUS.txt
# (or after MAX_SECONDS hard cap, whichever comes first).
#
# Run via: setsid -f nohup bash scripts/jupyterhub_keepalive.sh < /dev/null > /dev/null 2>&1

set -uo pipefail

INTERVAL="${INTERVAL:-180}"            # 3 min
MAX_SECONDS="${MAX_SECONDS:-86400}"    # 24 h hard cap
STATUS_FILE="logs/jakob_full_corpus/STATUS.txt"
LOG_FILE="logs/jakob_full_corpus/_heartbeat.log"

: "${JUPYTERHUB_API_URL:?JUPYTERHUB_API_URL not set}"
: "${JUPYTERHUB_API_TOKEN:?JUPYTERHUB_API_TOKEN not set}"
: "${JUPYTERHUB_USER:?JUPYTERHUB_USER not set}"

mkdir -p "$(dirname "$LOG_FILE")"
START_TS=$(date +%s)
echo "$(date -Is) heartbeat START pid=$$ interval=${INTERVAL}s user=$JUPYTERHUB_USER" >> "$LOG_FILE"

while true; do
  NOW_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  HTTP_CODE=$(curl -sS -o /dev/null -w "%{http_code}" \
    -X POST "$JUPYTERHUB_API_URL/users/$JUPYTERHUB_USER/activity" \
    -H "Authorization: token $JUPYTERHUB_API_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"servers\":{\"\":{\"last_activity\":\"$NOW_ISO\"}},\"last_activity\":\"$NOW_ISO\"}" \
    2>/dev/null || echo "ERR")
  echo "$(date -Is) ping http=$HTTP_CODE" >> "$LOG_FILE"

  if [[ -f "$STATUS_FILE" ]] && grep -q "BATCH COMPLETE" "$STATUS_FILE"; then
    echo "$(date -Is) heartbeat STOP (BATCH COMPLETE detected)" >> "$LOG_FILE"
    exit 0
  fi

  NOW_TS=$(date +%s)
  if (( NOW_TS - START_TS >= MAX_SECONDS )); then
    echo "$(date -Is) heartbeat STOP (MAX_SECONDS=$MAX_SECONDS hard cap)" >> "$LOG_FILE"
    exit 0
  fi

  sleep "$INTERVAL"
done
