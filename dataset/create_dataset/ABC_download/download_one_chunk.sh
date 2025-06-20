URL="$1"
FILE_NAME="$2"

SAVE_DIR="/home/lkh/siga/dataset/ABC/temp/download_7z"
THREADS=16  # 最大16
  
mkdir -p "$SAVE_DIR"

start_time=$(date +%s)

aria2c -c -x $THREADS -s $THREADS -o "$FILE_NAME" -d "$SAVE_DIR" "$URL"

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Downloaded $FILE_NAME in ${elapsed} seconds"