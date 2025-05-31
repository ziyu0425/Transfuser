#!/bin/bash

FOLDERNAME=nuscenes_trainval
SAVE_DIR=~/nuscenes_data/${FOLDERNAME}/nuscenes
mkdir -p "$SAVE_DIR"

# ====== 下载链接（请替换为你从官网获得的真实链接）======
#URL_01="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz"
#URL_02="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz"
#URL_03="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_blobs.tgz"
#URL_04="https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz"
#URL_05="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_blobs.tgz"
#URL_META="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz"
URL_LIDARSEG="https://d36yt3mvayqw5m.cloudfront.net/public/nuscenes-lidarseg-v1.0/nuScenes-lidarseg-all-v1.0.tar.bz2"
# ====== 文件保存路径 ======
#TGZ_PATH_01="$SAVE_DIR/v1.0-trainval01_blobs.tgz"
#TGZ_PATH_02="$SAVE_DIR/v1.0-trainval02_blobs.tgz"
#TGZ_PATH_03="$SAVE_DIR/v1.0-trainval03_blobs.tgz"
#TGZ_PATH_04="$SAVE_DIR/v1.0-trainval04_blobs.tgz"
#TGZ_PATH_05="$SAVE_DIR/v1.0-trainval05_blobs.tgz"
#TGZ_PATH_META="$SAVE_DIR/v1.0-trainval_meta.tgz"
TGZ_PATH_LIDARSEG="$SAVE_DIR/nuScenes-lidarseg-all-v1.0.tar.bz2"
# ====== 下载函数 ======
download_and_extract () {
  FILE_PATH=$1
  FILE_URL=$2
  FILE_NAME=$(basename "$FILE_PATH")

  if [ ! -f "$FILE_PATH" ]; then
    echo "⬇️ Downloading $FILE_NAME..."
    wget -O "$FILE_PATH" "$FILE_URL"
  else
    echo "✅ $FILE_NAME already exists."
  fi

  echo "📦 Extracting $FILE_NAME..."
  tar -xvjf "$FILE_PATH" -C "$SAVE_DIR"
}

# ====== 下载和解压每个文件 ======
#download_and_extract "$TGZ_PATH_01" "$URL_01"
#download_and_extract "$TGZ_PATH_02" "$URL_02"
#download_and_extract "$TGZ_PATH_03" "$URL_03"
#download_and_extract "$TGZ_PATH_META" "$URL_META"
download_and_extract "$TGZ_PATH_LIDARSEG" "$URL_LIDARSEG"
echo "🎉 All trainval data downloaded and extracted successfully."
