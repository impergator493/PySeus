#from skimage.util.dtype import img_as_float
#from pyseus.core import PySeus
#from pyseus import load
#from pyseus import formats
#from skimage import data, color
#from skimage.restoration import denoise_tv_chambolle 
import matplotlib.pyplot as plt
#import matplotlib.colors as col
import scipy.io
import cv2
import numpy as np
from skimage.util import random_noise

import numpy as np

from pyseus.processing.tv_denoising_old import TV_Denoise
from pyseus.processing.tgv_denoising import TGV_Denoise


img = scipy.io.loadmat('./tests/cameraman.mat')['pic']

noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']
noisy = (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy))

a = TV_Denoise()


b = TGV_Denoise()
denoised_TGV_32 = b.tgv2_denoising_gen(0,noisy,(12, 2, 1, 1000), (1,1))

lambda_n = 4
iter = 100
# rather for gaussian noise
denoised_L2_4 = a.tv_denoising_L2(noisy, lambda_n, iter)

lambda_n = 16
# rather for gaussian noise
denoised_L2_16 = a.tv_denoising_L2(noisy, lambda_n, iter)

lambda_n = 32
# rather for gaussian noise
denoised_L2_32 = a.tv_denoising_L2(noisy, lambda_n, iter)

lambda_n = 64
# rather for gaussian noise
denoised_L2_64 = a.tv_denoising_L2(noisy, lambda_n, iter)



fig = plt.figure(figsize=(16,10))
fig.suptitle(r'Comparison of different regularization parameters $\lambda$', fontsize=20)
plt.subplot(231)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.title('original', fontsize=15)
plt.subplot(232)
plt.imshow(noisy, cmap=plt.cm.gray)
plt.axis('off')
plt.title('noisy', fontsize=15)
plt.subplot(233)
plt.imshow(denoised_L2_4, cmap=plt.cm.gray)
plt.axis('off')
plt.title(r'TV $\lambda$=4 ', fontsize=15)
plt.subplot(234)
plt.imshow(denoised_L2_16, cmap=plt.cm.gray)
plt.axis('off')
plt.title(r'TV $\lambda$=16 ', fontsize=15)
plt.subplot(235)
plt.imshow(denoised_L2_32, cmap=plt.cm.gray)
plt.axis('off')
plt.title(r'TV $\lambda$=32 ', fontsize=15)
plt.subplot(236)
plt.imshow(denoised_TGV_32, cmap=plt.cm.gray)
plt.axis('off')
plt.title(r'TV $\lambda$=64 ', fontsize=15)

plt.subplots_adjust(wspace=0.02, hspace=0.1, top=0.9, bottom=0, left=0,
                    right=1)

#pic_directory = "D:\Mario\Studium\Studieren\Masterarbeit\\03_Daten"
#plt.savefig(pic_directory + "\denoise_tv_L2_iter=" + str(iter) + "_lambda=" + str(lambda_n) +".png" )

#plt.get_current_fig_manager().window.showMaximized()
plt.show()

