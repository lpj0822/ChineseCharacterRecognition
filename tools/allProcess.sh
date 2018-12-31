#!/usr/bin/env sh
python ../processCaffeData.py
./create_lmdb.sh
./computerMean.sh
./train.sh
