#!/bin/bash

SELECTED_FILES="1022.wav  1182.wav  1363.wav  1767.wav  2254.wav  2532.wav  2672.wav  3227.wav  \
                3474.wav  3926.wav  4083.wav  4268.wav  449.wav   4838.wav  4993.wav  733.wav   \
                105.wav   1192.wav  1469.wav  1783.wav  2307.wav  2540.wav  2676.wav  3261.wav  \
                3745.wav  4029.wav  4173.wav  4277.wav  4588.wav  491.wav   5004.wav  792.wav   \
                1145.wav  1289.wav  1523.wav  2033.wav  2420.wav  2584.wav  2802.wav  3262.wav  \
                3749.wav  4053.wav  4191.wav  4409.wav  4775.wav  4961.wav  505.wav   856.wav   \
                1176.wav  1300.wav  1707.wav  2072.wav  2450.wav  2670.wav  3138.wav  3399.wav  \
                3843.wav  4080.wav  4216.wav  4471.wav  4819.wav  4976.wav  64.wav    879.wav"

rm -rf data/processed/musiccaps/music_data_debug
mkdir -p data/processed/musiccaps/music_data_debug

# Selected files are copied currently for reproducibility of results.
# If you want to select a new random subset of 64 files, use the find command below instead.
bash -c "cd data/processed/musiccaps/music_data_train/; cp $SELECTED_FILES ../music_data_debug/"

# find data/processed/musiccaps/music_data_train/* -prune |
#     shuf -n 64 |
#     xargs -I {} cp "{}" "data/processed/musiccaps/music_data_debug/"
