rm -rf data/processed/musiccaps/music_data_debug
mkdir data/processed/musiccaps/music_data_debug
ls -1 data/processed/musiccaps/music_data_train/ |
    shuf -n 64 |
    xargs -I {} cp "data/processed/musiccaps/music_data_train/{}" "data/processed/musiccaps/music_data_debug/"
