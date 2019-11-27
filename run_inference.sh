#!/bin/bash

test_file=data/img/pic003___in_40.png
save_dir=data/img/
checkpoint_dir=results/checkpoint_impulses_bsd500_41/
phase=inference

python inference.py --test_file $test_file --save_dir $save_dir  --checkpoint_dir $checkpoint_dir --phase $phase



 
