#!/bin/bash
database=bsd500
src_dir=data
save_dir=data
patch_size=41
stride=$patch_size
step=0
batch_size=128
results_clean=img_clean_patches
results_noisy=img_noisy_patches
epoch=50
lr=0.001
use_gpu=1
sigma=25
phase=train
ip=0.3
checkpoint_dir=checkpoint_impulses
sample_dir=sample_impulses
test_dir=test_impulses
eval_noisy_set=noisy_impulses
eval_clean_set=clean


python main.py --database $database --save_dir $save_dir  --checkpoint_dir $checkpoint_dir --sample_dir $sample_dir --test_dir $test_dir --eval_noisy_set  $eval_noisy_set --eval_clean_set $eval_clean_set --batch_size $batch_size --results_clean $results_clean --results_noisy $results_noisy  --epoch $epoch --lr $lr --use_gpu $use_gpu --ip $ip --phase $phase --patch_size $patch_size
    
    
