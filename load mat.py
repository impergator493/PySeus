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

import h5py

# Full paths do not work some times, windows issue or something else?
# Sometimes Vs Code needs a restart to proper load the pictures again
# There is a difference between / and \ separator, for \ if there is a 0 after it, before the string there must be a r -> r'.\asdf..'

#load('./tests/samples/sample.h5')


#data = scipy.io.loadmat('../../03_Daten/Noise_generation/Mario_Master/brainphantom.mat')

with h5py.File('../../03_Daten/Noise_generation/Mario_Master/brainphantom.mat', 'r') as f:
    print(f.get('#refs#'))
    print(f.get('VObj'))
    
    refs=dict(f['#refs#'])
    obj = dict(f['VObj'])

    print(refs.keys())
    print(obj.keys())

    print(refs['a'].keys())
    print(obj.keys())


plt.figure(figsize=(16,10))
plt.subplot(231)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.title('original', fontsize=20)
plt.subplot(232)
plt.imshow(noisy, cmap=plt.cm.gray)
plt.axis('off')
plt.title('noisy', fontsize=20)
plt.subplot(233)
plt.imshow(denoised_L1, cmap=plt.cm.gray)
plt.axis('off')
plt.title('TV denoising L1', fontsize=20)
plt.subplot(234)
plt.imshow(denoised_L2, cmap=plt.cm.gray)
plt.axis('off')
plt.title('TV denoising L2', fontsize=20)
plt.subplot(235)
plt.imshow(denoised_TGV, cmap=plt.cm.gray)
plt.axis('off')
plt.title('TGV denoising ', fontsize=20)

plt.subplots_adjust(wspace=0.02, hspace=0.1, top=0.9, bottom=0, left=0,
                    right=1)

#pic_directory = "D:\Mario\Studium\Studieren\Masterarbeit\\03_Daten"
#plt.savefig(pic_directory + "\denoise_tv_L2_iter=" + str(iter) + "_lambda=" + str(lambda_n) +".png" )

#plt.get_current_fig_manager().window.showMaximized()
plt.show()

