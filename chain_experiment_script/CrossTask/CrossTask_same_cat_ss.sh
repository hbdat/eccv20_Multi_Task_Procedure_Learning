#!/bin/bash
GPU_0=0
GPU_1=1

for K in 7 10 12 15
do
    Folder_name="CrossTask_same_cat_ss_$K"
    echo ${Folder_name}
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 0 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 1 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 2 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 3 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 4 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 5 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 6 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 7 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 8 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 9 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 10 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 11 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 12 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 13 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 14 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 15 ${GPU_1} ${K}&
    wait
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 16 ${GPU_0} ${K}&
    python ./experiments/same_cat/CrossTask/CrossTask_rank_key_same_cat_ss_att_summarization.py ${Folder_name} 17 ${GPU_1} ${K}&
    wait
done
