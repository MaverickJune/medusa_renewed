#!/bin/bash

# Typical cruft
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
rm -rf .pytest_cache
