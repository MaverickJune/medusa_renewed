#!/bin/bash

# 1) Add the official package repository (one-time step)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

# 2) Install the package
apt-get install git-lfs

# 3) Activate Git-LFS hooks for your user
git lfs install