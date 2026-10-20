#!/usr/bin/env bash
# Load Cinelume CSV gzip files into MySQL. Adapted from the source repository's
# mysql/carregar.sh (same LOAD DATA statement, same row-count check), with one
# change: the table list comes from $TABELAS, so the multi-hour usage logs can
# be left out. Fails if any table loads a different number of rows than its CSV.
set -euo pipefail
HOST=${MYSQL_HOST:-127.0.0.1}
DIR=${CSV_DIR:-/dados/csv}
FK=${FK_CHECKS:-1}
MYSQL=(mysql -h "$HOST" -uroot --local-infile=1 cinelume)

for t in $TABELAS; do
  arq="$DIR/$t.csv.gz"
  [ -f "$arq" ] || { echo "faltando: $arq" >&2; exit 1; }
  cab=$(set +o pipefail; gzip -dc "$arq" 2>/dev/null | head -1 | tr -d '"\r')
  vars=""; sets=""
  IFS=',' read -ra cols <<< "$cab"
  for c in "${cols[@]}"; do
    vars+="@$c,"
    sets+="\`$c\` = NULLIF(@$c, ''),"
  done
  vars=${vars%,}; sets=${sets%,}
  fifo=$(mktemp -u /tmp/carga_XXXX); mkfifo "$fifo"
  gzip -dc "$arq" > "$fifo" &
  inicio=$(date +%s)
  "${MYSQL[@]}" -e "SET SESSION foreign_key_checks=0; TRUNCATE TABLE \`$t\`;
    SET SESSION foreign_key_checks=$FK;
    LOAD DATA LOCAL INFILE '$fifo' INTO TABLE \`$t\` CHARACTER SET utf8mb4
      FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' ESCAPED BY ''
      LINES TERMINATED BY '\n' IGNORE 1 LINES ($vars) SET $sets;"
  wait
  rm -f "$fifo"
  n=$("${MYSQL[@]}" -N -e "SELECT COUNT(*) FROM \`$t\`")
  esperado=$(( $(gzip -dc "$arq" | wc -l) - 1 ))
  echo "$(date +%T) $t: $n linhas ($(( $(date +%s) - inicio ))s)"
  if [ "$n" != "$esperado" ]; then
    echo "ERRO: $t tem $esperado linhas no CSV e $n no banco" >&2; exit 2
  fi
done
echo "carga concluída"
