# -*- coding: utf-8 -*-
# by F.H. learnt from https://blog.csdn.net/liuxiao214/article/details/74502975
# 其中z是生成器噪声，d或dis为判别器（discriminator）的缩写，g或gen为生成器（generator）的缩写，c是图像第三维度（颜色）
# 18.8.10
from __future__ import division  # 解决一下整数除法在python2和python3中不同的问题

import os  # 用来加载图片及设置显卡什么的

WINDOWS = True  # 在自己的电脑上就设置为True好了
# WINDOWS = False # srt服务器上

if not WINDOWS:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import numpy as np
from layers import *
from six.moves import xrange
# xrange与range类似，只是返回的是一个“xrange object”对象，而非数组list。要生成很大的数字序列的时候，
# 用xrange会比range性能优很多，因为不需要一上来就开辟一块很大的内存空间，这两个基本上都是在循环的时候用
import time
import scipy.misc


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


class DCGAN:
    def __init__(self, path=r'../common/getchu_faces_64/', input_h=64, input_w=64, batch_size=64, gf_dim=64,
                 df_dim=64, gfc_dim=1024, dfc_dim=1024, z_dim=100, c_dim=3, checkpoint_dir='checkpoint',
                 sample_num=64):
        # load Data
        self.sess = tf.Session()
        self.images = []
        self.image_h, self.image_w, self.batch_size = input_h, input_w, batch_size
        self.gf_dim, self.df_dim, self.gfc_dim, self.dfc_dim, self.c_dim, self.z_dim = \
            gf_dim, df_dim, gfc_dim, dfc_dim, c_dim, z_dim
        self.sample_num = sample_num
        self.checkpoint_dir = checkpoint_dir
        self.imagefiles = os.listdir(path)
        np.random.shuffle(self.imagefiles)
        if WINDOWS:
            self.imagefiles = self.imagefiles[:2000]
        for index, i in enumerate(self.imagefiles):
            data = tf.image.decode_jpeg(tf.gfile.FastGFile(os.path.join(path, i), 'rb').read())
            data = tf.image.convert_image_dtype(data, dtype=tf.uint8)
            self.images.append(data)
            if index % 50 == 0:
                print("Loading image " + str(index) + " of " + str(len(self.imagefiles)))
        print("Processing on images... May take few minutes, please wait")
        self.images = np.asarray(self.sess.run(self.images)) / 255.0  # python2 为整除，所以需要import division
        print("Images ready!")
        # batch normalization：批量标准化处理，不仅仅只对输入层的输入数据x进行标准化，还对每个隐藏层的输入进行标准化。
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.build_model()

    def train(self, config):
        # Adam 也是基于梯度下降的方法，但是每次迭代参数的学习步长都有一个确定的范围，
        # 不会因为很大的梯度导致很大的学习步长，参数的值比较稳定

        # 分别定义判别器和生成器的优化器
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run(session=self.sess)

        self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample = self.images[0:self.sample_num]
        sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            batch_idxs = len(self.imagefiles) // config.batch_size

            for idx in xrange(0, batch_idxs):
                batch = self.images[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, 100) == 1 and counter > 100:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            },
                        )
                        imsave((samples + 1.) / 2, image_manifold_size(samples.shape[0]),
                               'train_{:02d}_{:04d}.png'.format(epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        print("one pic error!...")

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def half(self, num):
        return int(np.ceil(num / 2))

    def build_model(self):
        image_dims = [self.image_h, self.image_w, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    # 生成判别器
    def discriminator(self, image, reuse=False):
        # 为了节约变量存储空间，通过共享变量作用域(variable_scope)来实现共享变量
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # 直接上来对图片数据就是一个卷积……太简单粗暴了吧
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            # 后面几层归一化一下再卷积
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

            return tf.nn.sigmoid(h4), h4

    # 生成器，直接用它和种子z生成所需图像
    def generator(self, z):
        # 在作用域中共享变量
        with tf.variable_scope("generator") as scope:
            # 获取宽高及不同倍率缩放的结果
            s_h, s_w = self.image_h, self.image_w
            s_h2, s_w2 = self.half(s_h), self.half(s_w)
            s_h4, s_w4 = self.half(s_h2), self.half(s_w2)
            s_h8, s_w8 = self.half(s_h4), self.half(s_w4)
            s_h16, s_w16 = self.half(s_h8), self.half(s_w8)

            # 通过输入的z构建第一层线性模型
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                # 在这里，64*64的1/16为4*4，第三维是64*8=512
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                # CONV 1，第一次小卷积，结果是8*8*256
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                # CONV 2，结果是16*16*128
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                # CONV 3，结果是32*32*64
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                # CONV 4，结果是64*64*3，就是最后的结果图
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    # 共享变量，其他和生成器差不多
    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.image_h, self.image_w
            s_h2, s_w2 = self.half(s_h), self.half(s_w)
            s_h4, s_w4 = self.half(s_h2), self.half(s_w2)
            s_h8, s_w8 = self.half(s_h4), self.half(s_w4)
            s_h16, s_w16 = self.half(s_h8), self.half(s_w8)

            # project `z` and reshape
            h0 = tf.reshape(
                linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)

    def model_dir(self):
        return "{}_{}_{}".format(self.batch_size, self.image_h, self.image_w)

    # 保存训练好的模型。创建检查点文件夹，如果路径不存在，则创建；然后将其保存在这个文件夹下。
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir())

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    # 读取检查点，获取路径，重新存储检查点，并且计数。打印成功读取的提示；如果没有路径，则打印失败的提示。
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir())

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
