import time

from utils import *
import numpy as np
from random import randint
from skimage.measure import compare_ssim as ssim
import glob
import matplotlib.pyplot as plt
import re
from inpainting import *



def dncnn(input, is_training=True, output_channels=3):
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, 3, padding='same', activation=tf.nn.relu)
    for layers in range(2, 16 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, 3, padding='same', name='conv%d' % layers, use_bias=False)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
    with tf.variable_scope('block17'):
        output_mask = tf.layers.conv2d(output, 1, 3, padding='same', activation = tf.nn.sigmoid)

        #mask = tf.scalar_mul(0.5,tf.ones_like(output_mask,dtype=tf.float32,name=None))
        #impulse_mask=tf.greater_equal(output_mask,mask)

        #impulse_mask3D = tf.concat([impulse_mask, impulse_mask, impulse_mask],3)
        #output_estimate= tf.layers.conv2d(output, output_channels, 3, padding='same')
        #
        #negation_mask = tf.logical_not(impulse_mask3D)
        #input.set_shape((None, None, None, None))
        #negation_mask.set_shape((None, None, None, None))
        #img_without_impulses=tf.where(negation_mask, input, tf.zeros_like(input))
        #img_without_impulses = tf.boolean_mask(input,np.array(negation_mask))
        #
        #noise_estimate = tf.where(impulse_mask3D, input - output_estimate, tf.zeros_like(output_estimate)) #tf.boolean_mask(output_estimate,impulse_mask)
    return   output_mask #tf.add(img_without_impulses, noise_estimate), impulse_mask, img_without_impulses, noise_estimate


class denoiser(object):
    def __init__(self, sess, input_c_dim=3, ip=25, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.ip = ip
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        is_empty  = tf.equal(tf.size(0), 0)



        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],name='noisy_image')

        self.diff = tf.reduce_sum(tf.abs(self.X - self.Y_), 3) > 0.0
        self.diff_mask = tf.expand_dims(self.diff, 3)
        self.max_Y = tf.reduce_max(self.Y_)
        self.impulse_mask = dncnn(self.X, is_training=self.is_training)


        self.threshold=tf.placeholder(tf.float32, shape=(), name="threshold")
        self.mask = self.impulse_mask > self.threshold #tf.expand_dims(self.diff, 3)
        negative_mask=tf.logical_not(self.mask)
        impulse_mask3D = tf.concat([negative_mask, negative_mask, negative_mask], 3)

        self.Y = tf.where(impulse_mask3D, self.X, tf.zeros_like(self.X))
        self.loss= (1 / batch_size) * tf.nn.l2_loss( tf.to_float(self.diff_mask) - tf.to_float(self.impulse_mask))
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        self.eva_psnr = tf_psnr(self.Y, self.Y_)
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, test_data_clean,test_data_noisy, sample_dir, summary_merged, summary_writer):
        # assert test_data value range is 0-255
        print("[*] Evaluating...")
        psnr_sum = 0
        for idx in range(len(test_data_clean)):
            clean_image = test_data_clean[idx].astype(np.float32) / 255.0
            clean_image_noisy = test_data_noisy[idx].astype(np.float32) / 255.0
            output_clean_image, noisy_image,org,impulse_mask,mask, loss, psnr_summary= self.sess.run(
                [self.Y,self.X,self.Y_, self.impulse_mask,self.mask, self.loss, self.eva_psnr],
                feed_dict={self.Y_: clean_image,self.X: clean_image_noisy,
                           self.is_training: False})
            summary_writer.add_summary(psnr_summary, iter_num)



            groundtruth = np.clip(test_data_clean[idx], 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            outputimage = np.clip( 255 * output_clean_image, 0, 255).astype('uint8')

            max_val = np.max(output_clean_image)

            print("MAx val:  {}".format(max_val) )


            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx + 1, psnr))
            psnr_sum += psnr
            save_images(os.path.join(sample_dir, 'test%d_%d.png' % (idx + 1, iter_num)),groundtruth, noisyimage, outputimage)

            save_images(os.path.join(sample_dir, 'original_mask%d_%d.png' % (idx + 1, iter_num)),np.clip(255 * mask, 0, 255).astype('uint8'))
            save_images(os.path.join(sample_dir, 'mask%d_%d.png' % (idx + 1, iter_num)), np.clip(255 * impulse_mask, 0, 255).astype('uint8'))
            save_images(os.path.join(sample_dir, 'impulses_removed%d_%d.png' % (idx + 1, iter_num)), np.clip(255 * img_without_impulses, 0, 255).astype('uint8'))
            save_images(os.path.join(sample_dir, 'impulses_estimate%d_%d.png' % (idx + 1, iter_num)), np.clip(255 * noise_estimate, 0, 255).astype('uint8'))
        avg_psnr = psnr_sum / len(test_data_clean)

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def train(self, data_clean,data_noisy, eval_data_clean,eval_data_noisy, batch_size, ckpt_dir, epoch, lr, sample_dir,logs_dir, eval_every_epoch=2):
        # assert data range is between 0 and 1
        numBatch = int(data_clean.shape[0] / batch_size)
        max_iter_number = 51200
        max_steps = 1024
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('lr', self.lr)
        tf.summary.image('images', self.Y, 10)

        writer = tf.summary.FileWriter(logs_dir, self.sess.graph)
        merged = tf.summary.merge_all()
        summary_psnr = tf.summary.scalar('eva_psnr', self.eva_psnr)
		
        epoches = epoch		
        if numBatch * epoches < max_iter_number:
            epoches = round(max_iter_number/numBatch)


        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_data_clean,eval_data_noisy, sample_dir=sample_dir, summary_merged=summary_psnr,
                      summary_writer=writer)  # eval_data value range is 0-255
        for epoch in range(start_epoch, epoches):
            p = np.random.permutation(data_clean.shape[0])
            data_clean=data_clean[p,:,:,:]
            data_noisy = data_noisy[p, :, :, :]
            steps = 0
            for batch_id in range(start_step, numBatch):
                if steps >=max_steps or iter_num >= max_iter_number:
                    break
                batch_images_clean = data_clean[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                batch_images_noisy = data_noisy[batch_id * batch_size:(batch_id + 1) * batch_size, :, :, :]
                _, loss, Y,max_Y, summary,XX,YY,impulse_mask,diff_mask = self.sess.run([self.train_op, self.loss, self.Y, self. max_Y, merged,self.X,self.Y_,self.impulse_mask,self.diff_mask],
                                                 feed_dict={self.Y_: batch_images_clean, self.X: batch_images_noisy, self.lr: lr[epoch],
                                                            self.is_training: True,self.threshold:0.5})
                print("Epoch: [%2d/%3d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1,epoches, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)
                if epoch ==1:
                    aa=1
				
                steps += 1
                data_clean2=[]
                data_noisy2=[]	
				
            if np.mod(epoch + 1, eval_every_epoch) == 0:
                self.evaluate(iter_num, eval_data_clean,eval_data_noisy, sample_dir=sample_dir, summary_merged=summary_psnr,
                              summary_writer=writer)  # eval_data value range is 0-255
                self.save(iter_num, ckpt_dir)
        print("[*] Finish training.")

    def save(self, iter_num, ckpt_dir, model_name='IDCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    #def load_checkpoints_list(self,checkpoint_dir):
    #    return sort_nicely(glob.glob(checkpoint_dir+'/*'+'.data-00000-of-00001'))

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            graph=tf.Graph()
            writer = tf.summary.FileWriter('./my_graph', graph)
            writer.close()
            return True, global_step
        else:
            return False, 0

    def load_checkpoint(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()

        if os.path.isfile(checkpoint_dir):

            if "ip" in checkpoint_dir:
                checkpoint_name = checkpoint_dir.split(".")[0:2]
                checkpoint_name='.'.join(checkpoint_name)
                global_step=-1
            else:
                checkpoint_name = checkpoint_dir.split(".")[0]
                global_step = int(checkpoint_name.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, checkpoint_name)
            graph = tf.Graph()
            writer = tf.summary.FileWriter('./my_graph', graph)
            writer.close()
            return True, global_step
        else:
            return False, 0
    def inference(self, test_image_name, ckpt_dir, save_dir,_threshold=0.5):
        """Test DnCNN"""
        # init variables
        tf.initialize_all_variables().run()
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")

        y = np.empty([0,0,0,3], dtype=float, order='C')
        noisy_image = load_images(test_image_name).astype(np.float32) / 255.0
        noisy_image, impulse_mask = self.sess.run([ self.X, self.impulse_mask],
                                                                      feed_dict={self.Y_: y,
                                                                                 self.X: noisy_image,
                                                                                 self.is_training: False,
                                                                                 self.threshold: 0.5})

        filename = os.path.basename(test_image_name)
        mask = impulse_mask < _threshold
        output_clean_image = noisy_image * mask

        reconstructed_img = mean_reconstruction(np.squeeze(output_clean_image), np.logical_not(np.squeeze(mask)), 1)
        outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
        reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype('uint8')

        save_images(os.path.join(save_dir, 'detected_impulses_CNN_' + filename), outputimage)
        save_images(os.path.join(save_dir, 'denoised_CNN_' + filename), reconstructed_img)

    def test(self, test_files_clean,test_files_noisy, ckpt_dir, save_dir, _threshold=0.5):
        """Test IDCNN"""
        # init variables
        tf.initialize_all_variables().run()
        assert len(test_files_clean) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0
        print("[*] " + 'noise level: ' + str(self.ip) + " start testing...")
        for idx in range(len(test_files_clean)):
            filename = os.path.basename(test_files_noisy[idx])
            if os.path.isfile(os.path.join(save_dir, 'denoised_CNN_' + filename)):
                continue;
            clean_image = load_images(test_files_clean[idx]).astype(np.float32) / 255.0
            noisy_image = load_images(test_files_noisy[idx]).astype(np.float32) / 255.0
            output_clean_image, noisy_image,impulse_mask = self.sess.run([self.Y, self.X,self.impulse_mask],
                                                            feed_dict={self.Y_: clean_image,self.X:noisy_image, self.is_training: False,self.threshold:_threshold})

            mask = impulse_mask < _threshold
            groundtruth = np.clip(255 * clean_image, 0, 255).astype('uint8')
            noisyimage = np.clip(255 * noisy_image, 0, 255).astype('uint8')
            output_clean_image = noisy_image*mask
            reconstructed_img = mean_reconstruction(np.squeeze(output_clean_image), np.logical_not(np.squeeze(mask)), 1)
            outputimage = np.clip(255 * output_clean_image, 0, 255).astype('uint8')
            reconstructed_img = np.clip(255 * reconstructed_img, 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, reconstructed_img)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(save_dir, 'detected_impulses_CNN_' + filename), outputimage)
            save_images(os.path.join(save_dir, 'denoised_CNN_' + filename), reconstructed_img)
        avg_psnr = psnr_sum / len(test_files_clean)
        print("--- Average PSNR %.2f ---" % avg_psnr)
