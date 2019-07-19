from dcgan import DCGAN

import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":

    # make sure to load in the correct sized data
    dcgan = DCGAN(img_rows = 128,
                    img_cols = 128,
                    channels = 3, 
                    latent_dim=256,
                    name='goodsell_256_128')
    
    dcgan.load_weights(generator_file="generator ({})4000.h5".format(dcgan.name), discriminator_file="discriminator ({}).h5".format(dcgan.name))

    # starting point for every image
    seed_start = np.random.normal(0, 1, (16, dcgan.latent_dim))

    # these parameters will change every time step
    latentSpeed = np.random.normal(3, 1, (16, dcgan.latent_dim))
    vary = np.copy(seed_start)

    # video settings
    time = 0
    fps = 30
    maxTime = 60 # seconds
    frameCount = 0
    
    while (time <= maxTime):

        # for each image
        for i in range(len(seed_start)): 
            
            # change the latent variables
            for j in range(dcgan.latent_dim):
                vary[i][j] = seed_start[i][j] + np.sin( 2*np.pi*(time/maxTime) * latentSpeed[i][j] ) 

        gen_imgs = dcgan.generator.predict(vary)

        create_image(gen_imgs, "images/goodsell_{0:05d}.png".format(frameCount)  )

        frameCount += 1
        time += 1./fps

    # ffmpeg -framerate 30 -i "galaxy_%05d.png" -i "Khruangbin_Friday Morning.mp3" -map 0:v:0 -map 1:a:0 -shortest -c:v libx264 -pix_fmt yuv420p -strict -2 galaxy.mp4 
    # ffmpeg -framerate 30 -i "nebula_%05d.png" -i "planet_caravan.mp3" -map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p nebula.mp4
    # ffmpeg -framerate 30 -i "fluidart_%05d.png" -c:v libx264 -pix_fmt yuv420p fluidart.mp4