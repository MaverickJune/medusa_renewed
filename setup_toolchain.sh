#!/usr/bin/env bash
###############################################################################
# setup_toolchain.sh
# Installs GCC/G++ + Clang and exports CC / CXX for future shells.
###############################################################################
set -euo pipefail

# 1) Install compiler tool-chain (skip if gcc already present)
if ! command -v gcc >/dev/null 2>&1; then
  echo "[toolchain] Installing build-essential and clang …"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends build-essential clang
  apt-get clean
  rm -rf /var/lib/apt/lists/*
else
  echo "[toolchain] GCC already present – skipping installation."
fi

# 2) Set CC / CXX for current session
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# 3) Persist them for every future shell in this container
cat >/etc/profile.d/00-cc-env.sh <<'EOF'
# Added by setup_toolchain.sh
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
EOF

echo "[toolchain] Done. Open a new shell or run:"
echo "  source /etc/profile.d/00-cc-env.sh"
echo "then execute   ./launch_train_script.sh"
