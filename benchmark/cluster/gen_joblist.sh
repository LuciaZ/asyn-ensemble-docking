#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for NT in amp centurion turing; do
  OUT="$SCRIPT_DIR/joblist_${NT}.tsv"
  echo -e "tool\tncpu\tpin\trep" > "$OUT"
  for rep in 1 2; do
    for ncpu in 1 4 8; do
      for pin in on off; do
        echo -e "vinagpu\t$ncpu\t$pin\t$rep" >> "$OUT"
      done
    done
    for ncpu in 1 4 8 16 32; do
      echo -e "vina\t$ncpu\toff\t$rep" >> "$OUT"
    done
  done
  echo "$NT: $(($(wc -l < "$OUT")-1)) jobs -> $OUT"
done
