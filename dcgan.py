try:
    from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
    from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
except: 
    from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LeakyReLU
    from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
    from keras.models import Sequential, Model
    from keras.optimizers import Adam
    from keras import backend as K
import tensorflow as tf

import os
import argparse
import glob 

from PIL import Image
import matplotlib.pyplot as plt

import sys

import numpy as np

class DCGAN():
    def __init__(self, img_rows=128, img_cols=128, channels=4, latent_dim=3, loss='binary_crossentropy', name='earth'):
        self.name = name

        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = latent_dim
        self.loss = loss

        self.optimizer = Adam(0.0005, 0.6)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the generator
        self.generator = self.build_generator()

        # Build the GAN
        self.build_combined()
        
    def build_combined(self):
        self.discriminator.compile(loss='binary_crossentropy',
                optimizer=self.optimizer,
                metrics=['accuracy'])
        
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)    
    
    def load_weights(self, generator_file=None, discriminator_file=None):

        if generator_file:
            generator = self.build_generator()
            generator.load_weights(generator_file)
            self.generator = generator
            print('generator weights loaded')
    
        if discriminator_file:
            discriminator = self.build_discriminator()
            discriminator.load_weights(discriminator_file)
            self.discriminator = discriminator
            print('discriminator weights loaded')

        if generator_file or discriminator_file: 
            self.build_combined() 
            print('build compaied ')

    def build_generator(self):

        model = Sequential()
        #model.add(Dense(128, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        #model.add(Dropout(0.1))
        
        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        model.add(Dropout(0.3))
        model.add(Reshape((16, 16, 128)))
        #model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
        model.add(UpSampling2D())
        
        model.add(Conv2D(64, kernel_size=5, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        #model.add(Dropout(0.2))
        model.add(UpSampling2D())

        model.add(Conv2D(32, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        
        model.add(Conv2D(32, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same", activation="sigmoid", name="generator_output"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img, name="generator")

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        #model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        discrim = Model(img, validity)

        return discrim

    def train(self, X_train, epochs, batch_size=128, save_interval=100):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if epoch % 10 == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs( "images/{}_{:05d}.png".format(self.name,epoch) )
                # self.combined.save_weights("combined_weights ({}).h5".format(self.name)) # https://github.com/keras-team/keras/issues/10949
                self.generator.save_weights("generator ({}).h5".format(self.name))
                self.discriminator.save_weights("discriminator ({}).h5".format(self.name))

    def save_imgs(self, name=''):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        # replace the first two latent variables with known values
        #for i in range(r):
        #    for j in range(c):
        #        noise[4*i+j][0] = i/(r-1)-0.5
        #        noise[4*i+j][1] = j/(c-1)-0.5

        gen_imgs = self.generator.predict(noise)

        fig, axs = plt.subplots(r, c, figsize=(6.72,6.72))
        plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95, wspace=0.2, hspace=0.2)

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1

        if name:
            fig.savefig(name, facecolor='black' )
        else: 
            fig.savefig('{}.png'.format(self.name), facecolor='black' )

        plt.close()
    

def export_model(saver, model, model_name, input_node_names, output_node_name):
    from tensorflow.python.tools import freeze_graph
    from tensorflow.python.tools import optimize_for_inference_lib
    
    if not os.path.exists('out'):
        os.mkdir('out')

    tf.train.write_graph(K.get_session().graph_def, 'out', model_name + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + model_name + '.chkp')

    freeze_graph.freeze_graph('out/' + model_name + '_graph.pbtxt', None, False,
                              'out/' + model_name + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + model_name + '.bytes', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + model_name + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + model_name + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")



def create_dataset(xSize=128, ySize=128, nSlices=100, resize=0.75, directory='dataset/'):
    jpgs = glob.glob( '{}*.jpg'.format(directory) )
    pngs = glob.glob( '{}*.png'.format(directory) )

    allimages = jpgs + pngs

    x_train = []
    y_train = []

    for i in range(len(allimages)):

        # load image
        img = Image.open(allimages[i])

        if resize != 1:
            img.thumbnail((img.size[0]*resize, img.size[1]*resize), Image.LANCZOS) # resizes image in-place

        img_data = np.array(list(img.getdata())).reshape( (img.size[1],img.size[0],-1) ) 

        for n in range(nSlices):

            # create random slices 
            rx = np.random.randint( img.size[0]-xSize)
            ry = np.random.randint( img.size[1]-ySize)
        
            # pull out portion of ccd
            sub = np.copy(img_data[ry:ry+ySize, rx:rx+xSize]).astype(float)

            #x_train.append( preprocessing.scale(sub) )
            x_train.append( sub[:,:,:3]  ) 
            y_train.append( [rx,ry] )

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    return (x_train, y_train)