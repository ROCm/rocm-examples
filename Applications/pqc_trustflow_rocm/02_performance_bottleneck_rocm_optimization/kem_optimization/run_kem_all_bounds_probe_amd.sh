#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export LD_LIBRARY_PATH="/opt/python/lib/python3.12/site-packages/_rocm_sdk_devel/lib:${LD_LIBRARY_PATH:-}"

N_OPS=${N_OPS:-20}
REPEATS=${REPEATS:-2}
DO_CORRECTNESS=${DO_CORRECTNESS:-1}
KYBER_BATCH=${KYBER_BATCH:-32768}
AIGIS_BATCH=${AIGIS_BATCH:-65536}
KEM_KEYGEN_TPB=${KEM_KEYGEN_TPB:-256}
KEM_ENCAPS_TPB=${KEM_ENCAPS_TPB:-128}
KEM_DECAPS_TPB=${KEM_DECAPS_TPB:-128}
OPT_LEVEL=${OPT_LEVEL:-O2}

stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="amd_results/all_bounds_probe_${stamp}"
mkdir -p "${out_dir}"

summary="${out_dir}/all_bounds_probe_raw.csv"
avg_summary="${out_dir}/all_bounds_probe_avg.csv"
best_summary="${out_dir}/all_bounds_probe_best.csv"

echo "target,algorithm_group,tag,repeat,batch,n_ops,opt,kg_tpb,enc_tpb,dec_tpb,keypair_bounds,encaps_bounds,decaps_bounds,keygen_ops_s,encaps_ops_s,decaps_ops_s,status,log" > "${summary}"

extract_metric() {
  local label="$1"
  local log="$2"
  grep -E "  ${label}:" "${log}" \
    | tail -1 \
    | grep -oE '[0-9]+ ops/sec' \
    | tail -1 \
    | awk '{print $1}'
}

group_for_target() {
  local target="$1"
  if [[ "${target}" == kyber* ]]; then
    echo "Kyber"
  else
    echo "Aigis-enc"
  fi
}

batch_for_target() {
  local target="$1"
  if [[ "${target}" == kyber* ]]; then
    echo "${KYBER_BATCH}"
  else
    echo "${AIGIS_BATCH}"
  fi
}

run_candidate() {
  local target="$1"
  local kb="$2"
  local eb="$3"
  local db="$4"
  local tag="bounds${kb}${eb}${db}"
  local group
  local batch
  group="$(group_for_target "${target}")"
  batch="$(batch_for_target "${target}")"

  for rep in $(seq 1 "${REPEATS}"); do
    local log="${out_dir}/${target}_${tag}_r${rep}.log"
    local status="PASS"
    {
      echo "========== ${target} ${tag} repeat=${rep}/${REPEATS} =========="
      echo "timestamp=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "batch=${batch} n_ops=${N_OPS} opt=${OPT_LEVEL}"
      echo "tpb=${KEM_KEYGEN_TPB}/${KEM_ENCAPS_TPB}/${KEM_DECAPS_TPB} bounds=${kb}/${eb}/${db}"
      OPT_LEVEL="${OPT_LEVEL}" \
      KEM_KEYGEN_TPB="${KEM_KEYGEN_TPB}" KEM_ENCAPS_TPB="${KEM_ENCAPS_TPB}" KEM_DECAPS_TPB="${KEM_DECAPS_TPB}" \
      KEM_KEYPAIR_LAUNCH_BOUNDS="${kb}" KEM_ENCAPS_LAUNCH_BOUNDS="${eb}" KEM_DECAPS_LAUNCH_BOUNDS="${db}" \
        bash build_hip.sh "${target}"

      if [[ "${DO_CORRECTNESS}" == "1" ]]; then
        "./${target}_amd" --batch 128 --n-ops 1
      fi

      "./${target}_amd" --batch "${batch}" --n-ops "${N_OPS}" --no-correctness
    } > "${log}" 2>&1 || status="FAIL"

    local kg_ops=""
    local enc_ops=""
    local dec_ops=""
    if [[ "${status}" == "PASS" ]]; then
      kg_ops="$(extract_metric Keygen "${log}" || true)"
      enc_ops="$(extract_metric Encaps "${log}" || true)"
      dec_ops="$(extract_metric Decaps "${log}" || true)"
    fi

    echo "${target},${group},${tag},${rep},${batch},${N_OPS},${OPT_LEVEL},${KEM_KEYGEN_TPB},${KEM_ENCAPS_TPB},${KEM_DECAPS_TPB},${kb},${eb},${db},${kg_ops},${enc_ops},${dec_ops},${status},${log}" | tee -a "${summary}"
  done
}

echo "[all-bounds] out=${out_dir}"
echo "[all-bounds] n_ops=${N_OPS} repeats=${REPEATS} correctness=${DO_CORRECTNESS}"
echo "[all-bounds] Kyber batch=${KYBER_BATCH}, Aigis batch=${AIGIS_BATCH}"

targets=(kyber512 kyber768 kyber1024 aigisenc1 aigisenc2 aigisenc3 aigisenc4)
bounds=(000 001 010 011 100 101 110 111)

for target in "${targets[@]}"; do
  echo
  echo "[target] ${target}"
  for b in "${bounds[@]}"; do
    run_candidate "${target}" "${b:0:1}" "${b:1:1}" "${b:2:1}"
  done
done

python3 - "${summary}" "${avg_summary}" "${best_summary}" <<'PY'
import csv
import sys
from collections import defaultdict

raw_path, avg_path, best_path = sys.argv[1:4]
rows = []
with open(raw_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("status") != "PASS":
            continue
        try:
            row["_keygen"] = float(row["keygen_ops_s"])
            row["_encaps"] = float(row["encaps_ops_s"])
            row["_decaps"] = float(row["decaps_ops_s"])
        except (TypeError, ValueError):
            continue
        rows.append(row)

groups = defaultdict(list)
for row in rows:
    key = (
        row["target"], row["algorithm_group"], row["tag"], row["batch"], row["n_ops"],
        row["opt"], row["kg_tpb"], row["enc_tpb"], row["dec_tpb"],
        row["keypair_bounds"], row["encaps_bounds"], row["decaps_bounds"],
    )
    groups[key].append(row)

avg_rows = []
for key, vals in groups.items():
    target, group, tag, batch, n_ops, opt, kg_tpb, enc_tpb, dec_tpb, kb, eb, db = key
    count = len(vals)
    kg = sum(v["_keygen"] for v in vals) / count
    enc = sum(v["_encaps"] for v in vals) / count
    dec = sum(v["_decaps"] for v in vals) / count
    # Balanced score avoids selecting configs that improve one operation while
    # badly hurting another. Encaps still gets a small extra weight because the
    # current optimization signal is launch-bounds-sensitive encaps.
    score = 0.30 * kg + 0.40 * enc + 0.30 * dec
    avg_rows.append({
        "target": target,
        "algorithm_group": group,
        "tag": tag,
        "batch": batch,
        "n_ops": n_ops,
        "opt": opt,
        "kg_tpb": kg_tpb,
        "enc_tpb": enc_tpb,
        "dec_tpb": dec_tpb,
        "keypair_bounds": kb,
        "encaps_bounds": eb,
        "decaps_bounds": db,
        "repeats": count,
        "keygen_avg_ops_s": round(kg),
        "encaps_avg_ops_s": round(enc),
        "decaps_avg_ops_s": round(dec),
        "balanced_score": round(score),
    })

fieldnames = [
    "target", "algorithm_group", "tag", "batch", "n_ops", "opt",
    "kg_tpb", "enc_tpb", "dec_tpb", "keypair_bounds", "encaps_bounds",
    "decaps_bounds", "repeats", "keygen_avg_ops_s", "encaps_avg_ops_s",
    "decaps_avg_ops_s", "balanced_score",
]

avg_rows.sort(key=lambda r: (r["target"], -int(r["balanced_score"])))
with open(avg_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(avg_rows)

best_by_target = {}
for row in avg_rows:
    target = row["target"]
    if target not in best_by_target or int(row["balanced_score"]) > int(best_by_target[target]["balanced_score"]):
        best_by_target[target] = row

with open(best_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for target in ["kyber512", "kyber768", "kyber1024", "aigisenc1", "aigisenc2", "aigisenc3", "aigisenc4"]:
        if target in best_by_target:
            w.writerow(best_by_target[target])
PY

echo
echo "[done] raw=${summary}"
echo "[done] avg=${avg_summary}"
echo "[done] best=${best_summary}"
echo
echo "[best]"
cat "${best_summary}"
