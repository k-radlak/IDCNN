import argparse
import glob
from PIL import Image
import PIL
import random
import os
#import numpy as np
from utils import *
import matplotlib.pyplot as plt


# the pixel value range is '0-255'(uint8 ) of training data

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

parser = argparse.ArgumentParser(description='')
parser.add_argument('--database', dest='database', default='bsd500', help='database with images')
parser.add_argument('--src_dir', dest='src_dir', default='./data/', help='dir of data')
parser.add_argument('--save_dir', dest='save_dir', default='./data', help='dir of patches')
parser.add_argument('--patch_size', dest='pat_size', type=int, default=41, help='patch size')
parser.add_argument('--stride', dest='stride', type=int, default=41, help='stride')
parser.add_argument('--step', dest='step', type=int, default=0, help='step')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='batch size')
parser.add_argument('--results_clean', dest='results_clean', default="./img_clean_patches", help='file with clean patches')
parser.add_argument('--results_noisy', dest='results_noisy', default="./img_noisy_patches", help='file with noisy patches')
parser.add_argument('--ip', dest='ip',type=float, default="0.3", help='noise intensity in the training')
args = parser.parse_args()


if args.ip:
    ip=float(args.ip)

def add_impulsive(img,ip):
    im_h = img.shape[0]
    im_w = img.shape[1]
    number_of_corrupted_pixels=int(round(im_h*im_w*ip))
    oMap=np.zeros(shape=(im_h,im_w))
    oImg=img.copy()

    for i in range(0,number_of_corrupted_pixels):
        while(1):
            x = int(np.floor(random.uniform(0, 1) * im_h))
            y = int(np.floor(random.uniform(0, 1) * im_w ))
            if (oMap[x, y] == 0):
                break

        oImg[x, y, 0] = np.floor((random.uniform(0, 1)  * 256)); # R
        oImg[x, y, 1] = np.floor((random.uniform(0, 1)  * 256)); # G
        oImg[x, y, 2] = np.floor((random.uniform(0, 1)  * 256)); # B

        oMap[x,y]=1

    return oImg



def generate_patches(isDebug=False):
    global DATA_AUG_TIMES
    count = 0

    if args.database:
        database=args.database



    database_output = database +"_" + str(args.pat_size)
    clean_output = os.path.join(args.save_dir, args.results_clean+'_'+ database_output+'.npy')
    noisy_output = os.path.join(args.save_dir, args.results_noisy+'_'+ database_output+'.npy')

    if os.path.isfile(clean_output) and  os.path.isfile(noisy_output):
        print("The training patches are generated")
        return

    if args.src_dir:
        src_dir=args.src_dir
    input_path=src_dir+'/'+database+ '/*.png'
    filepaths = glob.glob(input_path)
    if isDebug:
        filepaths = filepaths[:10]
    print
    "number of training data %d" % len(filepaths)

    scales = [1, 0.9, 0.8, 0.7]

    # calculate the number of patches
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i])#.convert('L')  # convert RGB to gray
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)  # do not change the original img
            im_h, im_w = img_s.size
            for x in range(0 + args.step, (im_h - args.pat_size), args.stride):
                for y in range(0 + args.step, (im_w - args.pat_size), args.stride):
                    count += 1
    origin_patch_num = count * DATA_AUG_TIMES

    if origin_patch_num % args.batch_size != 0:
        numPatches = int((origin_patch_num / args.batch_size + 1) * args.batch_size)
    else:
        numPatches = origin_patch_num
    print("total patches = %d , batch size = %d, total batches = %d" % \
    (numPatches, args.batch_size, numPatches / args.batch_size))

    # data matrix 4-D
    inputs = np.zeros((numPatches, args.pat_size, args.pat_size, 3), dtype="uint8")
    inputs_noisy = np.zeros((numPatches, args.pat_size, args.pat_size, 3), dtype="uint8")

    count = 0
    noise_vector  = np.arange(0.1,0.6,0.1)
    # generate patches
    for i in range(len(filepaths)):
        print("Image:" + str(i))
        img = Image.open(filepaths[i]).convert('RGB')
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
            # print newsize
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 3))  # extend one dimension
            for j in range(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + args.step, im_h - args.pat_size, args.stride):
                    for y in range(0 + args.step, im_w - args.pat_size, args.stride):
                        tst = img_s[0:1,0:1,:]
                        patch = img_s[x:x + args.pat_size, y:y + args.pat_size, :]
                        clean_array=data_augmentation(patch, \
                                                                   random.randint(0, 7))
                        #noisy=np.random.normal( loc = 0, scale=25.0 , size= np.shape(clean_array))
                        if ip==-1:
                            noisy=add_impulsive(clean_array,noise_vector[random.randint(0,len(noise_vector)-1)])
                        else:
                            noisy = add_impulsive(clean_array,ip)
                        inputs[count, :, :, :] = clean_array
                        inputs_noisy[count, :, :, :]=np.clip(noisy, 0, 255).astype('uint8')
                        count += 1
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]
        inputs_noisy[-to_pad:, :, :, :] = inputs_noisy[:to_pad, :, :, :]

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    np.save(clean_output, inputs)
    np.save(noisy_output, inputs_noisy)
    print
    "size of inputs tensor = " + str(inputs.shape)


if __name__ == '__main__':
    generate_patches()
