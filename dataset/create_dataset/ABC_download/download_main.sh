URL_FILE="step_links_raw.txt"  # 最后一行txt文件要有换行符

while read -r line; do
    # 解析 URL 和文件名
    URL=$(echo "$line" | awk '{print $1}')
    FILE_NAME=$(echo "$line" | awk '{print $2}')

    echo "Downloading $FILE_NAME from $URL ..."
    
    # 调用下载脚本
    ./download_one_chunk.sh "$URL" "$FILE_NAME"


done < "$URL_FILE"