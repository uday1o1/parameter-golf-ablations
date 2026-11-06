#!/usr/bin/env bash
# Every run behind the results (one GPU). Finished runs are skipped, so the script resumes. The machine_check
# sweep repeats two baseline seeds on a second machine: there, run `bash scripts/run_all.sh prepare machine_check dev`.
# SLOTS = how many training runs share the GPU at a time.
#   bash scripts/run_all.sh                       # all stages
#   bash scripts/run_all.sh ablation dev          # selected stages
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
SLOTS=${SLOTS:-4}
# CPU threads per run: the export step (quantization search, lzma) is CPU-bound, and several runs with
# one thread per core each starve one another on a shared host
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
GPUS=$(python -c "print(','.join(['0'] * $SLOTS))")
NL=external/DeepDataMiningLearning/DeepDataMiningLearning/llm

prepare() { [ -d data/golf/fineweb10B_sp1024 ] || python -m nanolm.prepare --golf-shards 1 --dest data/golf; }
tests() { (cd $NL && python -m pytest nanolm/tests -q -rs) | tee results/tests.txt; }
sweep() { python -m nanolm.sweep --sweep configs/sweeps/$1.yaml --gpus "$GPUS"; }
ablation() { sweep ablation; }      # every technique alone, 5 seeds
lr() { sweep lr; }                  # learning-rate sensitivity
dev() {                             # dev-set bpb of every run (held-out training tokens)
  python scripts/dev_eval.py runs/sweeps/*/*_s*
}
stack() { sweep stack; }            # final stack, leave-one-out, stack learning rates
machine_check() { sweep machine_check; }   # two baseline seeds on a second machine (not in the default stages)
compress() {                        # bits x compressor table and evaluation strides: baseline, stack, int6 stacks
  for d in runs/sweeps/stack/{baseline,stack,stack_plus_mlp3x,stack_plus_l6}_s*; do
    [ -e $d/compress.json ] || python -m nanolm.compress $d --bits 8 6 5 4 --compressors zlib lzma \
      --strides 0 256 128 64 --eval-tokens 2097152 > $d/compress.log 2>&1
  done
}
conv_weights() {                    # trained token_conv kernels: stack, token_conv alone (matrix_lr 0.02), stack + L6
  python scripts/token_conv_weights.py runs/sweeps/{stack/stack,lr/token_conv_mlr0.5,stack/stack_plus_l6}_s{1337,42,1,2,3} \
    > results/token_conv_weights.json
}
bench() { python scripts/bench_all.py results/bench.json; }    # tokens/s of each configuration alone
analyze() {                         # nanolm.analyze per sweep (val_bpb)
  for s in ablation stack; do
    python -m nanolm.analyze runs/sweeps/$s --metric val_bpb --baseline baseline > runs/sweeps/$s/analyze_stdout.txt
  done
  python -m nanolm.analyze runs/sweeps/lr --metric val_bpb > runs/sweeps/lr/analyze_stdout.txt
}
environment() {                     # hardware and software of this machine -> results/environment_$MACHINE.txt
  { echo "# Environment of the runs on machine ${MACHINE:-a}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; echo
    lscpu | grep -E "^(CPU\(s\)|Model name)"; echo
    python -c "import sys, torch; print('python', sys.version.split()[0]); print('torch', torch.__version__, 'cuda', torch.version.cuda)"; echo
    echo "nanolm: $(git -C external/DeepDataMiningLearning rev-parse HEAD) + patches/$(ls patches) (sha256 $(sha256sum patches/*.patch | cut -c1-64))"; echo
    python -m pip list --format=freeze 2>/dev/null | grep -iE "^($(grep -v '^#' requirements.txt | cut -d= -f1 | paste -sd'|' -))=="
  } > results/environment_${MACHINE:-a}.txt
}

mkdir -p results
STAGES=${*:-prepare tests ablation lr dev stack dev compress conv_weights bench analyze environment}
for s in $STAGES; do echo "[run_all] $s"; $s; done
echo "[run_all] done"
