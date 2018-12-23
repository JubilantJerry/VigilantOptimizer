#!/usr/bin/env bash
set -x
rm -r build/*
rm -r dist/*
./build_ffi.py egg_info --egg-base build install