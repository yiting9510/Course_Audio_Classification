#!/bin/bash

# generate split data
# 原始音檔跟標註放 data_dir 底下的 audio 跟 label
data_dir="/mnt/ext_disk/dhf/data(1)"
# 描述各個 split 檔案內容放在 split_dir
split_dir="/mnt/ext_disk/dhf/data(1)/split"
# seed 影響 split 挑選的結果還有 model 的初始化
seed=42
# 輸出紀錄的資料夾
exp_dir=output
# 生成幾種 訓練,驗證集切法
splits=3
for num_class in 5 ;
# for num_class in 2 5 ;
do
    python split_data.py --data_dir $data_dir --valid_ratio 0.2 --output_dir $split_dir --num_class $num_class --seed $seed --split_num $splits
done

# 實驗名稱
ename="data_augmentation-(44:11:10)-b8-e20-(lr1e-4)" 
# 每次用新的 cache_dir 都要重新計算 cache
# 要嘗試不同的 segment_length, stride 再更改
cache_dir=".cache/30-20"

# 有四個相同尺寸的模型可以選
# facebook/wav2vec2-base-960h
# facebook/hubert-base-ls960
# microsoft/wavlm-base
# microsoft/wavlm-base-plus
model_checkpoint=microsoft/wavlm-base-plus
num_epochs=20
lr=1e-4  # 1e-5, 5e-5
batch_size=8
segment_length=30
stride=20
grad_accum_steps=1

# train and evaluate
for num_class in 5 ;
# for num_class in 2 5 ;
do
    for mod in seq_labling audio_cls;
    do
        exp_name=$mod-$num_class"class"-$ename-$segment_length-$stride
        split_file=$split_dir/$num_class"-class-split.json"
        python train.py --exp_name $exp_name --split_file $split_file --output_dir $exp_dir --num_class $num_class  --mod $mod\
                --cache_dir $cache_dir --seed $seed --splits $splits --model_checkpoint $model_checkpoint --num_epochs $num_epochs\
                --lr $lr --batch_size $batch_size --segment_length $segment_length --stride $stride --grad_accum_steps $grad_accum_steps
        
    done
done