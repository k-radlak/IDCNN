import argparse
from glob import glob

import tensorflow as tf

from model import denoiser
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--test_file',  help='test file name')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint_impulses', help='models are saved here')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')



args = parser.parse_args()



if args.checkpoint_dir:
    checkpoint_dir= args.checkpoint_dir


if args.save_dir:
    results_dir= args.save_dir
else:
    results_dir='.'




def denoiser_inference(denoiser):

    denoiser.inference(args.test_file, ckpt_dir=checkpoint_dir, save_dir=results_dir)



def main(_):
    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            if args.phase == 'inference':
                denoiser_inference(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess)
            if args.phase == 'inference':
                denoiser_inference(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
