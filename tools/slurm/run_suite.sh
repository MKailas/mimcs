#!/bin/bash
# Run the mimcs test suite on a Slurm cluster as an array of chunks, one process per chunk.
#
# The suite does not fit in one process (a whole-suite run aborted inside an XLA compile once
# enough programs had accumulated), so it is split the way CLAUDE.md prescribes locally: chunks of
# at most MAX_TESTS tests, each memory-heavy `test_pt*.py` file in a chunk of its own, built from
# `pytest --collect-only` so a new test file can never be silently skipped. Each chunk is one array
# task with its own log and exit code; a dependent job writes `summary.txt` once all have ended.
#
# Run it on a login node, from anywhere:
#
#     tools/slurm/run_suite.sh --account A --partition small --python PATH
#
# Everything lands in one run directory (default `suite_runs/<timestamp>` under the repository):
#
#     chunks/chunk_NNN.txt   the test files of each chunk
#     logs/chunk_NNN.log     pytest's output for that chunk;  logs/chunk_NNN.rc  its exit code
#     slurm/                 Slurm's own stdout/stderr per task (crashes outside pytest land here)
#     summary.txt            per-chunk exit codes, the failed tests, and the totals
#
# Exit code 137 (or a missing .rc) is the out-of-memory signature: the chunk never reached its
# summary, so its failures are *unknown*, not zero. The summary says so rather than counting it clean.
#
# Options (each also settable as the environment variable in brackets):
#   --account A      Slurm account                                   [SUITE_ACCOUNT]  (required)
#   --partition P    Slurm partition                                 [SUITE_PARTITION] (default small)
#   --python PATH    interpreter with jax and pytest                 [SUITE_PYTHON]   (default python)
#   --time T         wall time per chunk                             [SUITE_TIME]     (default 01:00:00)
#   --mem M          memory per chunk                                [SUITE_MEM]      (default 8G)
#   --cpus N         CPUs per chunk                                  [SUITE_CPUS]     (default 4)
#   --max-tests N    tests per chunk                                 [SUITE_MAX_TESTS] (default 150)
#   --out DIR        run directory                                   [SUITE_OUT]
#   --pytest-args S  extra pytest arguments, e.g. "-x" or "-k nuts"  [SUITE_PYTEST_ARGS]
#   --dry-run        build the chunks and print the sbatch commands without submitting

set -u

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACCOUNT=${SUITE_ACCOUNT:-}
PARTITION=${SUITE_PARTITION:-small}
PYTHON=${SUITE_PYTHON:-python}
TIME=${SUITE_TIME:-01:00:00}
MEM=${SUITE_MEM:-8G}
CPUS=${SUITE_CPUS:-4}
MAX_TESTS=${SUITE_MAX_TESTS:-150}
OUT=${SUITE_OUT:-$REPO/suite_runs/$(date +%Y%m%d_%H%M%S)}
PYTEST_ARGS=${SUITE_PYTEST_ARGS:-}
DRY_RUN=0

while [ $# -gt 0 ]; do
  case $1 in
    --account) ACCOUNT=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --mem) MEM=$2; shift 2 ;;
    --cpus) CPUS=$2; shift 2 ;;
    --max-tests) MAX_TESTS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --pytest-args) PYTEST_ARGS=$2; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

if [ -z "$ACCOUNT" ] && [ $DRY_RUN -eq 0 ]; then
  echo "--account is required (or set SUITE_ACCOUNT)" >&2; exit 2
fi
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd) || { echo "cannot create $OUT" >&2; exit 1; }
mkdir -p "$OUT/chunks" "$OUT/logs" "$OUT/slurm"

# --- collect and chunk ------------------------------------------------------------------------ #

cd "$REPO" || exit 1
"$PYTHON" -m pytest --collect-only -q -p no:cacheprovider tests/ > "$OUT/collect.txt" 2>&1
collect_rc=$?
grep "::" "$OUT/collect.txt" | cut -d: -f1 | sort | uniq -c | sort -k2 > "$OUT/counts.txt"
if [ $collect_rc -ne 0 ] || [ ! -s "$OUT/counts.txt" ]; then
  echo "collection failed (exit $collect_rc); see $OUT/collect.txt" >&2; exit 1
fi

n_chunks=0; chunk=(); n=0
flush() {
  [ ${#chunk[@]} -eq 0 ] && return
  n_chunks=$((n_chunks + 1))
  printf '%s\n' "${chunk[@]}" > "$OUT/chunks/chunk_$(printf %03d $n_chunks).txt"
  chunk=(); n=0
}
while read -r cnt f; do
  if [[ $(basename "$f") == test_pt* ]]; then flush; chunk=("$f"); flush; continue; fi
  if [ $((n + cnt)) -gt "$MAX_TESTS" ]; then flush; fi
  chunk+=("$f"); n=$((n + cnt))
done < "$OUT/counts.txt"
flush
total=$(awk '{s += $1} END {print s}' "$OUT/counts.txt")
echo "collected $total tests in $(wc -l < "$OUT/counts.txt") files -> $n_chunks chunks in $OUT"

# --- the per-chunk task and the summary ------------------------------------------------------- #

cat > "$OUT/task.sh" <<EOF
#!/bin/bash
# One chunk of the suite: SLURM_ARRAY_TASK_ID selects chunks/chunk_NNN.txt.
id=\$(printf %03d "\${SLURM_ARRAY_TASK_ID:?not an array task}")
cd "$REPO" || exit 1
mapfile -t files < "$OUT/chunks/chunk_\$id.txt"
echo "chunk \$id on \$(hostname): \${files[*]}" > "$OUT/logs/chunk_\$id.log"
"$PYTHON" -m pytest -q -rfE -p no:cacheprovider $PYTEST_ARGS "\${files[@]}" >> "$OUT/logs/chunk_\$id.log" 2>&1
rc=\$?
echo \$rc > "$OUT/logs/chunk_\$id.rc"
exit \$rc
EOF

cat > "$OUT/summarize.sh" <<'EOF'
#!/bin/bash
# Tabulate the chunks: exit code, pytest's last line, the failed tests, and the totals.
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# pytest's final line, with or without the === border (-q drops it unless something failed).
FINAL='^(=+ )?(no tests ran|[0-9]+ (passed|failed|skipped|error|errors|deselected|xfailed|xpassed|warnings?)).* in [0-9.]+s'
{
  echo "mimcs suite run: $OUT"
  echo
  printf '%-10s %-6s %s\n' chunk rc result
  unknown=0
  for c in "$OUT"/chunks/chunk_*.txt; do
    id=$(basename "$c" .txt); id=${id#chunk_}
    rc=$(cat "$OUT/logs/chunk_$id.rc" 2>/dev/null || echo none)
    last=$(grep -E "$FINAL" "$OUT/logs/chunk_$id.log" 2>/dev/null | tail -1)
    if [ "$rc" = none ] || [ "$rc" -gt 1 ] 2>/dev/null; then
      unknown=$((unknown + 1)); last="${last:-no pytest summary} <- did not finish (see slurm/)"
    fi
    printf '%-10s %-6s %s\n' "$id" "$rc" "$last"
  done
  echo
  echo "failed tests:"
  grep -h -E '^(FAILED|ERROR) ' "$OUT"/logs/chunk_*.log 2>/dev/null | sed 's/^/  /' || true
  echo
  # Totals from each chunk's final summary line only: a count printed inside a test's captured
  # output must not be added in.
  finals=$(for f in "$OUT"/logs/chunk_*.log; do
             grep -E "$FINAL" "$f" | tail -1
           done)
  for k in passed failed skipped error; do
    printf '%s %s\n' "$k" "$(grep -oE "[0-9]+ ${k}s?\b" <<< "$finals" \
      | awk '{s += $1} END {print s + 0}')"
  done
  echo "chunks that did not finish: $unknown (their failures are unknown, not zero)"
} > "$OUT/summary.txt"
cat "$OUT/summary.txt"
EOF
chmod +x "$OUT/task.sh" "$OUT/summarize.sh"

# --- submit ----------------------------------------------------------------------------------- #

common=(--account="$ACCOUNT" --partition="$PARTITION")
array_cmd=(sbatch --parsable "${common[@]}" --job-name=mimcs-suite --array=1-"$n_chunks"
           --time="$TIME" --mem="$MEM" --cpus-per-task="$CPUS"
           --output="$OUT/slurm/chunk_%a.out" --error="$OUT/slurm/chunk_%a.err" "$OUT/task.sh")
if [ $DRY_RUN -eq 1 ]; then
  echo "dry run; would submit:"
  echo "  ${array_cmd[*]}"
  echo "  sbatch ${common[*]} --dependency=afterany:<array job> --time=00:05:00 --mem=1G" \
       "--output=$OUT/slurm/summary.out $OUT/summarize.sh"
  exit 0
fi
array_id=$("${array_cmd[@]}") || { echo "array submission failed" >&2; exit 1; }
summary_id=$(sbatch --parsable "${common[@]}" --job-name=mimcs-suite-summary \
             --dependency=afterany:"$array_id" --time=00:05:00 --mem=1G \
             --output="$OUT/slurm/summary.out" "$OUT/summarize.sh") \
  || { echo "summary submission failed (array job $array_id is running)" >&2; exit 1; }
echo "submitted array job $array_id ($n_chunks chunks) and summary job $summary_id"
echo "results: $OUT/summary.txt once job $summary_id has run"
