#!/usr/bin/env bash
set -e

echo "$(date) ... creating additional Tezaurs DB role"
psql -v ON_ERROR_STOP=1 --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" <<-EOSQL
create role tezaurs_public;
create extension fuzzystrmatch;
EOSQL
echo "$(date) ... role created successfully"

echo "$(date) ... importing Tezaurs dump"
time psql -U "$POSTGRES_USER" "$POSTGRES_DB" < "/dumps/$DUMP_FNAME"
echo "$(date) ... Tezaurs dump imported successfully"
