JSON_DIR='/home/lkh/siga/dataset/ABC/temp/filter_log'
UNZIP_DIR='/home/lkh/siga/dataset/ABC/temp/step'

for i in $(seq -w 60 69); do
    # 定义JSON文件
    JSON_PATH="${JSON_DIR}/${i}.jsonl"

    # 解压数据集
    ./unzip_one_chunk.sh "$i"


done