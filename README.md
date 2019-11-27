# Impulse Detection Convolutional Neural Network (IDCNN)

The code prepared based on the tensorflow implementation of DnCNN from https://github.com/crisb-DUT/DnCNN-tensorflow, which was designed for Gaussian noise removal. The proposed filter is a modyfication of the DnCNN desiged for impulsive noise removal.




## Results
Under preparation

## Environment
Under preparation

## How to run
Under preparation

### Train
* To train the model download BSD500 dataset from https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
* Put images data/bsd500/
* In the next step generate patches that are used in the training with default parameters
```
$ python generate_patches.py
```
  or using shell script in which you can easili modify the parameters to control how the patches are generated
```
$ bash generate_patches.sh
```


* Run training with default parameters 
```
$ python main.py
```
    or using shell script in which you can easili modify the training parameters
```
$ bash generate_patches.sh
```
### Test
```
$ python main.py --phase test

```
### Inference on pretrained model
```
$ python inference.py --test_file data/img/pic003___in_40.png --save_dir .  --checkpoint_dir results/checkpoint_impulses_bsd500_41/ --phase inference
```
  or using shell script
```
bash run_inference.sh
```
 
 










