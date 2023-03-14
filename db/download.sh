#!/usr/bin/env bash
set -e

cd "$(dirname "$(readlink -f "$0")")"

echo "$(date) ... downloading Tezaurs dump to directory ./dumps"

mkdir -p "dumps"

time curl --request GET -sL \
     --url 'https://wordnet.ailab.lv/data/tezaurs_current-public.pgsql.gz'\
     --output './dumps/tezaurs_current-public.pgsql.gz'

cd "dumps"
gzip -d "tezaurs_current-public.pgsql.gz"

echo "$(date) ... dump downloaded successfully"
