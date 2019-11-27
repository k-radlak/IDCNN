import argparse
from glob import glob

import tensorflow as tf

from model import denoiser
from utils import *

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--ip', dest='ip', type=float, default=0.3, help='impulsive noise level')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='checkpoint_impulses', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='sample_impulses', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='test_impulses', help='test sample are saved here')
parser.add_argument('--eval_noisy_set', dest='eval_noisy_set', default='noisy_impulses', help='dataset for eval in training')
parser.add_argument('--eval_clean_set', dest='eval_clean_set', default='clean', help='dataset for eval in training')
parser.add_argument('--test_set_clean', dest='test_set_clean', default='clean', help='dataset for testing')
parser.add_argument('--test_set_noisy', dest='test_set_noisy', default='noisy_impulses', help='dataset for testing')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--database', dest='database', default='bsd500', help='database with images')
parser.add_argument('--results_clean', dest='results_clean', default="./data/img_clean_patches", help='get pic from file')
parser.add_argument('--results_noisy', dest='results_noisy', default="./data/img_noisy_patches", help='get pic from file')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=41, help='patch size')

args = parser.parse_args()



if args.ip:
    ip=float(args.ip)

database_output = args.database +"_" + str(args.pat_size)
results_output = 'results/'

if not os.path.exists(results_output):
    os.mkdir(results_output)

if args.checkpoint_dir:
    checkpoint_dir= results_output+ args.checkpoint_dir+'_'+ database_output

if args.sample_dir:
    sample_dir=results_output + args.sample_dir+'_'+ database_output

if args.test_dir:
    test_dir= results_output + args.test_dir+'_'+ database_output

logs_dir =  results_output + 'logs_'+ database_output






def denoiser_train(denoiser):
    print(args.save_dir+'/'+args.results_clean+'_'+ database_output + '.npy')
    print(args.save_dir+'/'+args.results_noisy+'_'+ database_output + '.npy')
    with load_data(filepath=args.save_dir+'/'+args.results_clean+'_'+ database_output+'.npy') as data_clean:
        with load_data(filepath=args.save_dir+'/'+args.results_noisy+'_'+ database_output+'.npy') as data_noisy:
            # if there is a small memory, please comment this line and uncomment the line99 in model.py
            data_clean = data_clean.astype(np.float32) / 255.0  # normalize the data to 0-1
            data_noisy = data_noisy.astype(np.float32) / 255.0  # normalize the data to 0-1
            eval_noisy_files = sorted(glob('./data/test/{}/*.png'.format(args.eval_noisy_set)))
            eval_clean_files = sorted(glob('./data/test/{}/*.png'.format(args.eval_clean_set)))
            eval_data_noisy = load_images(eval_noisy_files)  # list of array of different size, 4-D, pixel value range is 0-255
            eval_data_clean = load_images(eval_clean_files)  # list of array of different size, 4-D, pixel value range is 0-255
            print("work")

            numBatch = int(data_clean.shape[0] / args.batch_size)
            max_iter_number = 51200
            epoches = args.epoch
            if numBatch * epoches < max_iter_number:
                epoches = round(max_iter_number / numBatch)

            lr = args.lr * np.ones([epoches])
            lr[30:] = lr[0] / 10.0

            denoiser.train(data_clean, data_noisy, eval_data_clean, eval_data_noisy, batch_size=args.batch_size, ckpt_dir=checkpoint_dir, epoch=args.epoch, lr=lr,
                           sample_dir=sample_dir,logs_dir=logs_dir)


def denoiser_test(denoiser):
    test_files_clean = sorted(glob('./data/test/{}/*.png'.format(args.test_set_clean)))
    test_files_noisy = sorted(glob('./data/test/{}/*.png'.format(args.test_set_noisy)))
    print(test_files_clean)
    denoiser.test(test_files_clean,test_files_noisy, ckpt_dir=checkpoint_dir, save_dir=test_dir)

def denoiser_test2(denoiser):
    test_files_noisy = sorted(glob('./data/test/{}/*.png'.format(args.test_set_noisy)))
    test_files_clean = []
    for item in test_files_noisy:
        file = os.path.basename(item)
        basename= os.path.splitext(file)
        test_files_clean.append('./data/test/'+args.test_set_clean+'/'+basename[0][:6] + basename[1])
    denoiser.test_full(test_files_clean,test_files_noisy, ckpt_dir=checkpoint_dir, save_dir=test_dir)

def denoiser_generate_map(denoiser):
    test_files_noisy = sorted(glob('./data/test/{}/*.png'.format(args.test_set_noisy)))
    test_files_clean = sorted(glob('./data/test/{}/*.png'.format(args.test_set_noisy)))
    # for item in test_files_noisy:
    #     file = os.path.basename(item)
    #     test_files_clean.append('./data/test/'+args.test_set_clean+'/'+file)
    denoiser.test_map(test_files_clean,test_files_noisy, ckpt_dir=checkpoint_dir, save_dir=test_dir)

def main(_):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)


    if args.use_gpu:
        # added to control the gpu memory
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, ip=ip)
            if args.phase == 'train':
                denoiser_train(model)
            elif args.phase == 'test':
                denoiser_test(model)
            elif args.phase == 'test_repeatability':
                denoiser_test2(model)
            elif args.phase == 'generate_map':
                denoiser_generate_map(model)
            else:
                print('[!]Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = denoiser(sess, ip=ip)
            if args.phase == 'train':
                denoiser_train(model, lr=lr)
            elif args.phase == 'test':
                denoiser_test(model)
            else:
                print('[!]Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()
