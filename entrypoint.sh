#!/bin/bash

resolve_mesos_netloc() {
  MESOS_DNS_URI=$1
  HOST=$2
  curl -s "${MESOS_DNS_URI}/v1/services/_${HOST}._tcp.marathon.mesos" \
    | jq '.[]' \
    | jq -r '"\(.ip):\(.port)"'
}
export -f resolve_mesos_netloc

resolve_mesos_host() {
  MESOS_DNS_URI=$1
  HOST=$2
  curl -s "${MESOS_DNS_URI}/v1/services/_${HOST}._tcp.marathon.mesos" \
    | jq '.[]' \
    | jq -r '"\(.ip)"'
}
export -f resolve_mesos_host

# Resolve environment variables
export MONGOURI=$(bash -c "echo ${MONGOURI}")
export RURL=$(bash -c "echo ${RURL}")
export MYSQLURI=$(bash -c "echo ${MYSQLURI}")

env | egrep '^(MONGOURI|RURL|MYSQLURI)='

exec "$@"
