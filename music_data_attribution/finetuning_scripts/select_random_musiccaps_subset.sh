#!/bin/bash

rm -rf data/processed/musiccaps/music_data_debug
mkdir -p data/processed/musiccaps/music_data_debug
find data/processed/musiccaps/music_data_train/* -prune |
    shuf -n 64 |
    xargs -I {} cp "{}" "data/processed/musiccaps/music_data_debug/"
