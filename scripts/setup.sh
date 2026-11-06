#!/usr/bin/env bash
# Fetch nanolm at the pinned commit, apply this project's patch (the new model options and their
# tests), and install the Python dependencies.
#   bash scripts/setup.sh            (from the repository root; uses the active Python)
set -euo pipefail
cd "$(dirname "$0")/.."
NANOLM_COMMIT=8418fc1bae85657190f11749483bf2e729b9702c
REPO=external/DeepDataMiningLearning
if [ ! -d $REPO/.git ]; then
  git clone --filter=blob:none https://github.com/lkk688/DeepDataMiningLearning.git $REPO
fi
git -C $REPO checkout -q -f "$NANOLM_COMMIT"          # the pinned commit, without local changes
for p in patches/*.patch; do
  git -C $REPO apply "$PWD/$p"
done
python -m pip install -q -r requirements.txt
