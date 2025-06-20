# JSON_DIR='/home/lkh/siga/dataset/ABC/temp/filter_log'
# UNZIP_DIR='/home/lkh/siga/dataset/ABC/temp/step'

# for i in $(seq -w 0 99); do
#     # 定义JSON文件
#     JSON_PATH="${JSON_DIR}/${i}.jsonl"

#     # 解压数据集
#     # ./unzip_one_chunk.sh "$i"

#     # 筛选数据集
#     UNZIP_INDEX_DIR="${UNZIP_DIR}/${i}"
#     bash /home/lkh/siga/CADIMG/dataset/create_dataset/ABC_download/filter_one_chunk.sh "$UNZIP_INDEX_DIR" "$JSON_PATH"

#     rm -rf "/home/lkh/siga/dataset/ABC/temp/step/${i}"

# done


JSON_DIR='/home/lkh/siga/dataset/ABC/temp/filter_log'
UNZIP_DIR='/home/lkh/siga/dataset/ABC/temp/step'
MAX_JOBS=10  # 根据CPU核心数适当设置

parallel_jobs=0

for i in $(seq -w 00 09); do
    {
        # 定义JSON文件
        JSON_PATH="${JSON_DIR}/${i}.jsonl"

        # 解压数据集
        # ./unzip_one_chunk.sh "$i"
    
        # 筛选数据集
        UNZIP_INDEX_DIR="${UNZIP_DIR}/${i}"
        bash /home/lkh/siga/CADIMG/dataset/create_dataset/ABC_download/filter_one_chunk.sh "$UNZIP_INDEX_DIR" "$JSON_PATH"

        # 清理临时目录
        # rm -rf "$UNZIP_INDEX_DIR"
    } &

    parallel_jobs=$((parallel_jobs + 1))

    # 控制并发数
    if [ "$parallel_jobs" -ge "$MAX_JOBS" ]; then
        wait  # 等待已有任务完成
        parallel_jobs=0
    fi
done

# 等待所有后台任务结束
wait
