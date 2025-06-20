INDEX="00"

INDEX="$1"

DIR="/home/lkh/siga/dataset/ABC/temp/download_7z"
FILE_PATH="${DIR}/abc_00${INDEX}_step_v00.7z"
TARGET_DIR="/home/lkh/siga/dataset/ABC/temp/step/${INDEX}"

echo "解压 ${FILE_PATH} 到 ${TARGET_DIR} ..."
mkdir -p "$TARGET_DIR"

# 若文件不存在则跳过
if [ ! -f "$FILE_PATH" ]; then
    echo "文件不存在，跳过：$FILE_PATH"
    exit 0
fi

start_time=$(date +%s)

7z x "$FILE_PATH" -o"$TARGET_DIR" -mmt=on

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "解压 $FILE_PATH 用时 ${elapsed} 秒"
echo "--------------------------"