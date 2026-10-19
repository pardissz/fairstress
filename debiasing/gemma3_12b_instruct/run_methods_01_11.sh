#!/bin/bash
# Runs methods 01-11's run_all.py sequentially from the project root (required:
# each script's relative paths like external_data/... and firmcore_eval/... are
# resolved against cwd, not against the script's own directory).
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

METHODS=(01_prompting 02_self_debiasing 03_cot_counterfactual 04_self_critique_revise \
         05_leace 06_inlp 07_caa 08_cast 09_loreft 10_shifting_perspectives)

for m in "${METHODS[@]}"; do
  echo "=== [$(date)] starting $m ===" | tee -a logs/_driver.log
  python3 "methods/$m/run_all.py" > "logs/${m}.log" 2>&1
  rc=$?
  echo "=== [$(date)] finished $m rc=$rc ===" | tee -a logs/_driver.log
  if [ $rc -ne 0 ]; then
    echo "!!! $m FAILED (rc=$rc), see logs/${m}.log -- continuing to next method" | tee -a logs/_driver.log
  fi
done

# Method 11 (pressure_mixed) has no methods/11_pressure_mixed/run_all.py in the
# original project either -- its driver lives in methods_deep_analysis/.
echo "=== [$(date)] starting 11_pressure_mixed ===" | tee -a logs/_driver.log
python3 "methods_deep_analysis/run_pressure_mixed_full.py" > "logs/11_pressure_mixed.log" 2>&1
rc=$?
echo "=== [$(date)] finished 11_pressure_mixed rc=$rc ===" | tee -a logs/_driver.log
if [ $rc -ne 0 ]; then
  echo "!!! 11_pressure_mixed FAILED (rc=$rc), see logs/11_pressure_mixed.log" | tee -a logs/_driver.log
fi

echo "=== [$(date)] ALL METHODS DONE ===" | tee -a logs/_driver.log
