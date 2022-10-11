#!/bin/bash

### 将本次作业计费到导师课题组，tutor_project改为导师创建的课题组名
#SBATCH --comment=qinbiao

### 给你这个作业起个名字，方便识别不同的作业
#SBATCH --job-name=pmtlmwq1008test

### 指定该作业需要多少个节点
### 注意！没有使用多机并行（MPI/NCCL等），下面参数写1！不要多写，多写了也不会加速程序！
#SBATCH --nodes=1

### 指定该作业在哪个队列上执行
### 目前可用的GPU队列有 titan/tesla
#SBATCH --partition=tesla

### 申请一块GPU卡，一块GPU卡默认配置了一定数量的CPU核
### 注意！程序没有使用多卡并行优化的，下面参数写1！不要多写，多写也不会加速程序！
#SBATCH --gpus=1

### 以上参数用来申请所需资源
### 以下命令将在计算节点执行

nvidia-smi

### 执行你的作业
### python main.py -test --data_type sq
python show_att_weight.py
