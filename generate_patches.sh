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
ip=0.3




python generate_patches_impulses.py  --database $database --src_dir $src_dir --save_dir $save_dir --patch_size $patch_size --stride $stride --step $step --batch_size $batch_size --results_clean $results_clean --results_noisy $results_noisy  --ip $ip
