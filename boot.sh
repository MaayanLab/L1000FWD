#!/bin/sh

if [ -z "${ENTER_POINT}" ]; then
  export ENTER_POINT="/L1000FWD"
fi

python app.py &
PID=$!

sleep 5
while true; do
  URL="http://localhost:5000${ENTER_POINT}/"
  PROBE="$(curl --silent --connect-timeout 1 --write-out '%{response_code}' -o /dev/null ${URL})"
  echo "GET ${URL} returned ${PROBE}"
  if [ "${PROBE}" -eq "200" ]; then
    touch ready
  elif [ "${PROBE}" -eq "000" ]; then
    echo ""
  else
    exit ${PROBE}
  fi
  sleep 60
done
