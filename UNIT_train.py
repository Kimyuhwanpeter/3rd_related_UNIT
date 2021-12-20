# -*- coding:utf-8 -*-
from UNIT_model import *
from random import shuffle

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import easydict

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "batch_size": 2,
                           
                           "lr": 0.0001,

                           "epochs": 100,
                           
                           "A_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/female_16_39_train.txt",
                           
                           "A_img_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/",
                           
                           "B_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/male_40_63_train.txt",
                           
                           "B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_40_63/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "sample_images": "C:/Users/Yuhwan/Downloads/dd"})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def _func(A_filename, B_filename):

    h = tf.random.uniform([1], 1e-2, 5)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 5)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)
    
    A_image_string = tf.io.read_file(A_filename)
    A_decode_image = tf.image.decode_jpeg(A_image_string, channels=3)
    A_decode_image = tf.image.resize(A_decode_image, [FLAGS.img_size + 5, FLAGS.img_size + 5])
    A_decode_image = A_decode_image[h:h+FLAGS.img_size, w:w+FLAGS.img_size, :]
    A_decode_image = tf.image.convert_image_dtype(A_decode_image, tf.float32) / 127.5 - 1.

    B_image_string = tf.io.read_file(B_filename)
    B_decode_image = tf.image.decode_jpeg(B_image_string, channels=3)
    B_decode_image = tf.image.resize(B_decode_image, [FLAGS.img_size + 5, FLAGS.img_size + 5])
    B_decode_image = B_decode_image[h:h+FLAGS.img_size, w:w+FLAGS.img_size, :]
    B_decode_image = tf.image.convert_image_dtype(B_decode_image, tf.float32) / 127.5 - 1.

    if tf.random.uniform(()) > 0.5:
        A_decode_image = tf.image.flip_left_right(A_decode_image)
        B_decode_image = tf.image.flip_left_right(B_decode_image)

    return A_decode_image, B_decode_image

@tf.function
def cal_loss(generator_A, generator_B,
             sh_encoder, sh_generator,
             encoder_A, encoder_B,
             A_images, B_images,
             discriminator_A, discriminator_B):

    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape(persistent=True) as d_tape:

        # translation
        out = tf.concat([encoder_A(A_images, True), encoder_B(B_images, True)], axis=0)
        shared = sh_encoder(out, True)
        out = sh_generator(shared, True)

        out_A = generator_A(out, True)
        out_B = generator_B(out, True)

        x_Aa, x_Ba = tf.split(out_A, 2, axis=0)
        x_Ab, x_Bb = tf.split(out_B, 2, axis=0)

        # generator a2b
        out = encoder_A(x_Ba, True)
        shared_bab = sh_encoder(out, True)
        out = sh_generator(shared_bab, True)
        x_bab = generator_B(out, True)

        # generator b2a
        out = encoder_B(x_Ab, True)
        shared_aba = sh_encoder(out, True)
        out = sh_generator(shared_aba, True)
        x_aba = generator_A(out, True)

        # real discriminator
        real_A_logit = discriminator_A(A_images, True)
        real_B_logit = discriminator_B(B_images, True)

        # fake discriminator
        fake_A_logit = discriminator_A(x_Ba, True)
        fake_B_logit = discriminator_B(x_Ab, True)

        # Define loss
        G_ad_loss_a = generator_loss(fake_A_logit, smoothing=False, use_lsgan=False)
        G_ad_loss_b = generator_loss(fake_B_logit, smoothing=False, use_lsgan=False)

        D_ad_loss_a = discriminator_loss(real_A_logit, fake_A_logit, smoothing=False, use_lasgan=False)
        D_ad_loss_b = discriminator_loss(real_B_logit, fake_B_logit, smoothing=False, use_lasgan=False)

        enc_loss = KL_divergence(shared)
        enc_bab_loss = KL_divergence(shared_bab)
        enc_aba_loss = KL_divergence(shared_aba)

        l1_loss_a = L1_loss(x_Aa, A_images) # identity
        l1_loss_b = L1_loss(x_Bb, B_images) # identity
        l1_loss_aba = L1_loss(x_aba, A_images) # reconstruction
        l1_loss_bab = L1_loss(x_bab, B_images) # reconstruction

        Generator_A_loss = 10.0 * G_ad_loss_a + \
                           100.0 * l1_loss_a + \
                           100.0 * l1_loss_aba + \
                           0.1 * enc_loss + \
                           0.1 * enc_bab_loss
        Generator_B_loss = 10.0 * G_ad_loss_b + \
                           100.0 * l1_loss_b + \
                           100.0 * l1_loss_bab + \
                           0.1 * enc_loss + \
                           0.1 * enc_aba_loss
        Discriminator_A_loss = 10.0 * D_ad_loss_a
        Discriminator_B_loss = 10.0 * D_ad_loss_b

        Generator_loss = Generator_A_loss + Generator_B_loss
        Discriminator_loss = Discriminator_A_loss + Discriminator_B_loss
        
    g_grads1 = g_tape.gradient(Generator_loss, encoder_A.trainable_variables + encoder_B.trainable_variables)
    g_grads2 = g_tape.gradient(Generator_loss, generator_A.trainable_variables + generator_B.trainable_variables)
    g_grads3 = g_tape.gradient(Generator_loss, sh_encoder.trainable_variables)
    g_grads4 = g_tape.gradient(Generator_loss, sh_generator.trainable_variables)

    d_grad = d_tape.gradient(Discriminator_loss, discriminator_A.trainable_variables + discriminator_B.trainable_variables)

    g_optim.apply_gradients(zip(g_grads1, encoder_A.trainable_variables + encoder_B.trainable_variables))
    g_optim.apply_gradients(zip(g_grads2, generator_A.trainable_variables + generator_B.trainable_variables))
    g_optim.apply_gradients(zip(g_grads3, sh_encoder.trainable_variables))
    g_optim.apply_gradients(zip(g_grads4, sh_generator.trainable_variables))

    d_optim.apply_gradients(zip(d_grad, discriminator_A.trainable_variables + discriminator_B.trainable_variables))

    return Generator_loss, Discriminator_loss

def main():

    generator_A = generator()
    generator_B = generator()
    sh_encoder = share_encoder()
    sh_generator = share_generator()
    encoder_A = encoder()
    encoder_B = encoder()
    discriminator_A = discirminator()
    discriminator_B = discirminator()


    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(generator_A=generator_A, generator_B=generator_B, 
                                   sh_encoder=sh_encoder, sh_generator=sh_generator,
                                   encoder_A=encoder_A, encoder_B=encoder_B,
                                   discriminator_A=discriminator_A, discriminator_B=discriminator_B,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    A_input = np.loadtxt(FLAGS.A_txt_path, dtype="<U200", skiprows=0, usecols=0)
    A_input = [FLAGS.A_img_path + data for data in A_input]

    B_input = np.loadtxt(FLAGS.B_txt_path, dtype="<U200", skiprows=0, usecols=0)
    B_input = [FLAGS.B_img_path + data for data in B_input]

    count = 0
    for epoch in range(FLAGS.epochs):

        shuffle(A_input)
        shuffle(B_input)

        A_input, B_input = np.array(A_input), np.array(B_input)
        
        tr_gener = tf.data.Dataset.from_tensor_slices((A_input, B_input))
        tr_gener = tr_gener.shuffle(len(A_input))
        tr_gener = tr_gener.map(_func)
        tr_gener = tr_gener.batch(FLAGS.batch_size)
        tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)
        
        tr_iter = iter(tr_gener)
        tr_idx = len(A_input) // FLAGS.batch_size
        for step in range(tr_idx):
            A_images, B_images = next(tr_iter)
          
            g_loss, d_loss = cal_loss(generator_A, generator_B,
                                      sh_encoder, sh_generator,
                                      encoder_A, encoder_B,
                                      A_images, B_images,
                                      discriminator_A, discriminator_B)


            if count % 100 == 0:
                print("Epoch: {}, G_loss = {}, D_loss = {} [{}/{}]".format(epoch, g_loss, d_loss, step + 1, tr_idx))

                # generator a2b
                out = encoder_A(A_images, False)
                shared_bab = sh_encoder(out, False)
                out = sh_generator(shared_bab, False)
                fake_B = generator_B(out, False)

                # generator b2a
                out = encoder_B(B_images, False)
                shared_aba = sh_encoder(out, False)
                out = sh_generator(shared_aba, False)
                fake_A = generator_B(out, False)

                plt.imsave(FLAGS.sample_images + "/" + "{}_1_B_fake.png".format(count), fake_B[0].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_1_A_fake.png".format(count), fake_A[0].numpy() * 0.5 + 0.5)

            if count % 1000 == 0:
                model_dir = FLAGS.save_checkpoint
                folder_name = int(count/1000)
                folder_neme_str = '%s/%s' % (model_dir, folder_name)
                if not os.path.isdir(folder_neme_str):
                    print("Make {} folder to save checkpoint".format(folder_name))
                    os.makedirs(folder_neme_str)
                checkpoint = tf.train.Checkpoint(generator_A=generator_A, generator_B=generator_B, 
                                   sh_encoder=sh_encoder, sh_generator=sh_generator,
                                   encoder_A=encoder_A, encoder_B=encoder_B,
                                   discriminator_A=discriminator_A, discriminator_B=discriminator_B,
                                   g_optim=g_optim, d_optim=d_optim)
                checkpoint_dir = folder_neme_str + "/" + "UNIT_model_{}_steps.ckpt".format(count + 1)
                checkpoint.save(checkpoint_dir)

            count += 1
            # 집에가서 이미지뽑아보자

if __name__ == "__main__":
    main()
