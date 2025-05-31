#!/bin/bash

FOLDERNAME=nuscenes_mini
SAVE_DIR=~/nuscenes_data/${FOLDERNAME}/data/sets/nuscenes

mkdir -p "$SAVE_DIR"

FILE_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"
FILE_URL1="https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2"
FILE_URL2="https://www.nuscenes.org/data/nuScenes-panoptic-v1.0-mini.tar.gz"

TGZ_PATH="$SAVE_DIR/v1.0-mini.tgz"
TGZ_PATH1="$SAVE_DIR/nuScenes-lidarseg-mini-v1.0.tar.bz2"
TGZ_PATH2="$SAVE_DIR/nuScenes-panoptic-v1.0-mini.tar.gz"

EXTRACT_DIR="$SAVE_DIR/v1.0-mini"
EXTRACT_DIR1="$SAVE_DIR/nuScenes-lidarseg-mini-v1.0"
EXTRACT_DIR2="$SAVE_DIR/nuScenes-panoptic-v1.0-mini"

if [ ! -f "$TGZ_PATH" ]; then
  echo "Downloading v1.0-mini.tgz..."
  wget -O "$TGZ_PATH" "$FILE_URL"
else
  echo "v1.0-mini.tgz already exists."
fi

if [ ! -d "$EXTRACT_DIR" ]; then
  echo "Extracting v1.0-mini.tgz..."
  tar -xvzf "$TGZ_PATH" -C "$SAVE_DIR"
else
  echo "v1.0-mini already extracted."
fi

if [ ! -f "$TGZ_PATH1" ]; then
  echo "Downloading nuScenes-lidarseg-mini..."
  wget -O "$TGZ_PATH1" "$FILE_URL1"
  echo "Extracting nuScenes-lidarseg-mini..."
  tar -xvjf "$TGZ_PATH1" -C "$SAVE_DIR"
else
  echo "nuScenes-lidarseg-mini already exists."
fi

if [ ! -f "$TGZ_PATH2" ]; then
  echo "Downloading nuScenes-panoptic-mini..."
  wget -O "$TGZ_PATH2" "$FILE_URL2"
  echo "Extracting nuScenes-panoptic-mini..."
  tar -xvzf "$TGZ_PATH2" -C "$SAVE_DIR"
else
  echo "nuScenes-panoptic-mini already exists."
fi
