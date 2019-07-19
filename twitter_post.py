from dcgan import DCGAN

import argparse
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
 
from skimage.transform import rotate 
from skimage.color import rgb2hsv, hsv2rgb

import twitter

class AnimatedGif:
    def __init__(self, size=(512, 512)):
        self.fig, self.axs = plt.subplots(4,4, figsize=(4,4), facecolor=(1,1,1) )
        plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95, wspace=0.2, hspace=0.2)

        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        self.images = []
 
    def add(self, images):
        imgs = []
        for k in range(16):
            ax = self.axs[int(k/4.),k%4].imshow(images[k])
            self.axs[int(k/4.),k%4].axis('off')
            imgs.append(ax)
        self.images.append(imgs)
 
    def save(self, filename, fps=1):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, fps=fps,
            #progress_callback = lambda i, n: print(f'Saving frame {i} of {n}')
        )

models = ['fluid_256_128', 'space_256_128', 'goodsell_256_128']

def create_image(gen_imgs, name, xsize=4, ysize=4):
    
    fig, axs = plt.subplots(xsize, ysize, figsize=(xsize*2,ysize*2))
    plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95, wspace=0.2, hspace=0.2)

    cnt = 0
    for i in range(ysize):
        for j in range(xsize):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1

    fig.savefig(name, facecolor='white' )
    
    plt.close()


def make_post():
    # make sure to load in the correct sized data
    dcgan = DCGAN(img_rows = 128,
                    img_cols = 128,
                    channels = 3, 
                    latent_dim=512,
                    name='goodsell_512_128')
    
    dcgan.load_weights(generator_file="generator ({}).h5".format(dcgan.name), discriminator_file="discriminator ({}).h5".format(dcgan.name))
    
    # video settings
    fps = 30
    maxTime = 30 # seconds
    frameCount = 0
    time = 0
    nframes = int( maxTime*fps )

    # controls for animation
    seed_start = np.random.normal(1, 1, (16, dcgan.latent_dim))
    latentSpeed = np.random.normal(2, 1, (16, dcgan.latent_dim))
    vary = np.random.normal(1, 1, (16, nframes, dcgan.latent_dim)) 

    # randomize image transformations
    #rhue =  np.random.random()
    #rotation = 360 * np.round(np.random.random((4,))*4)/4 # random rotation
    #flip = np.random.randint(0,4,(4,)) # 0=normal, 1=y-axis, 2=x-axis, 3=transpose 

    # latent parameter animation
    for k in range(16):
        time = 0

        # for each image in animation 
        for i in range(nframes): 
            
            # change the latent variables
            for j in range(dcgan.latent_dim):
                vary[k][i][j] = seed_start[k][j] + np.sin( 2*np.pi*(time/maxTime) * latentSpeed[k][j] ) 

            time += 1./fps

    imgs = []
    for k in range(16):
        imgs.append(
            dcgan.generator.predict(vary[k])
        )
    imgs = np.array(imgs)

    # create animation
    animated_gif = AnimatedGif()
    for i in range(nframes):
        
        animated_gif.add(imgs[:,i])

    animated_gif.save('artificial_art.mp4',fps=fps)

    dude() 

    count = np.loadtxt('count.txt')

    with open('hashtags.txt') as fp:
        hashtags = fp.readlines()
    hashtags = [hashtags[i].strip() for i in range(len(hashtags))]

    message = "Automated Artificial Art v 1.{:.1f} - machine hallucinations from an artificial neural network \n \n#".format(count[0])
    message = message + " #".join( np.random.choice(hashtags,4))

    with open('twitter_api_keys.json') as f: 
        data = json.load(f)
        api = twitter.Api(consumer_key=data["consumer_key"],
                    consumer_secret= data["consumer_secret"],
                    access_token_key=data["access_token"],
                    access_token_secret=data["access_secret"])

        api.PostUpdate(message, 
            media="artificial_art.mp4"
        )
        count += 1
        np.savetxt('count.txt',count)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    help_ = "Sleep time"
    parser.add_argument("-s", "--sleep", help=help_, default=24*60*60, type=int)
    args = parser.parse_args()

    #while(True):
    make_post() 
        #time.sleep(args.sleep)