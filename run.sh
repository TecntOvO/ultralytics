#!/bin/bash

# 训练脚本列表
train_scripts=( "train.py" "train.py" )
log_dir="runs/log"

# 确保日志目录存在
mkdir -p $log_dir

# 遍历每个训练脚本并执行
for script in "${train_scripts[@]}"
do
    script_name=$(basename $script)
    log_file="${log_dir}/${script_name%.py}_$(date +%Y%m%d-%H%M).log"

    echo "开始运行脚本: $script" | tee -a $log_file

    # 记录开始时间
    start_time=$(date +%s)
    echo "训练开始时间: $(date)" | tee -a $log_file

    (
        source "venv/Scripts/activate"
        python $script
    ) > $log_file 2>&1

    # 记录结束时间
    end_time=$(date +%s)
    echo "训练结束时间: $(date)" | tee -a $log_file

    # 计算训练时间，保留一位小数
    duration=$((end_time - start_time))
    hours=$(echo "scale=1; $duration / 3600" | bc)

    echo "脚本 $script 训练时间: ${hours}小时" | tee -a $log_file

    # 从日志文件中提取 experiment_name
    experiment_name=$(grep "Experiment name:" $log_file | awk '{print $3}')

    if [ -z "$experiment_name" ]; then
        echo "未能提取 experiment_name，日志文件保留原名称。" | tee -a $log_file
    else
        # 重命名日志文件
        new_log_file="${log_dir}/${experiment_name}_$(date +%Y%m%d-%H%M).log"
        mv $log_file $new_log_file
        echo "日志文件已重命名为: $new_log_file"
        # 删除原来的日志文件
        rm -f $log_file
    fi

    # 检查脚本是否成功运行
    if [ $? -ne 0 ]; then
        echo "脚本 $script 运行失败，日志已保存到 $new_log_file" | tee -a $new_log_file
    else
        echo "脚本 $script 运行成功，日志已保存到 $new_log_file" | tee -a $new_log_file
    fi

    # 输出一个空行以便日志更易读
    echo ""
done

echo "所有脚本运行完毕"


