#!/usr/bin/env bash
set -x
# rm -r build/*
# rm -r dist/*
CFLAGS="-O3 -ffast-math" ./build_ffi.py egg_info --egg-base build install