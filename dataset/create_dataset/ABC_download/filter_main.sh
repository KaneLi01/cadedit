JSON_DIR='/home/lkh/siga/dataset/ABC/temp/filter_log'
UNZIP_DIR='/home/lkh/siga/dataset/ABC/temp/step'

# for i in $(seq -w 20 39); do
#     # 定义JSON文件
#     JSON_PATH="${JSON_DIR}/${i}.jsonl"

#     # 解压数据集
#     ./unzip_one_chunk.sh "$i"

#     # rm -rf "/home/lkh/siga/dataset/ABC/temp/step/${i}"

# done


# JSON_DIR='/home/lkh/siga/dataset/ABC/temp/filter_log'
# UNZIP_DIR='/home/lkh/siga/dataset/ABC/temp/step'
MAX_JOBS=20

for i in $(seq -w 20 39); do
    {
        JSON_PATH="${JSON_DIR}/${i}.jsonl"
        UNZIP_INDEX_DIR="${UNZIP_DIR}/${i}"
        bash /home/lkh/siga/CADIMG/dataset/create_dataset/ABC_download/filter_one_chunk.sh "$UNZIP_INDEX_DIR" "$JSON_PATH"
    } &

    # 控制并发数
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1  # 等待有进程完成
    done
done

wait 
