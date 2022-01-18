from skimage.util.dtype import img_as_float
from pyseus.core import PySeus
from pyseus import load
from pyseus import formats
from skimage import data, color
from skimage.restoration import denoise_tv_chambolle 
import matplotlib.pyplot as plt
import matplotlib.colors as col
import scipy.io

import numpy as np

from pyseus.processing.tv import TV


# Full paths do not work some times, windows issue or something else?
# Sometimes Vs Code needs a restart to proper load the pictures again
# There is a difference between / and \ separator, for \ if there is a 0 after it, before the string there must be a r -> r'.\asdf..'

#load('./tests/samples/sample.h5')



#img = img_as_float(color.rgb2gray(data.camera()))

# noisy image from matlab, to compare the results
img = scipy.io.loadmat('./tests/cameraman.mat')['pic']

#img = (img - np.min(img)) / (np.max(img) - np.min(img))

#noise = 0.05 * np.random.randn(img.shape[0],img.shape[1])
print("---------")
print("Extrem Values original itself")
print(img.max())
print(img.min())
print("---------")


#noisy = img + noise

noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']
noisy = (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy))


# All values have to be normalized mapped to 0 to 1, normally from a uint data format of uint8 (0=0, 255=1)

print("---------")
print("Extrem Values Noisy")
print(noisy.max())
print(noisy.min())
print("---------")


lambda_n = 8
iter = 100

# rather for saltn pepper noise
a = TV()
denoised_L1 = a.tv_denoising_L1(noisy, lambda_n, iter)
denoised_L1 = (denoised_L1 - np.min(denoised_L1)) / (np.max(denoised_L1) - np.min(denoised_L1))

#lambda_n = 16
#iter = 100

# rather for gaussian noise
denoised_L2 = a.tv_denoising_L2(noisy, lambda_n, iter)
denoised_L2 = (denoised_L2 - np.min(denoised_L2)) / (np.max(denoised_L2) - np.min(denoised_L2))

#lambda_n = 16
#iter = 100
alpha = 0.03

denoised_huberROF = a.tv_denoising_huberROF(noisy, lambda_n, iter, alpha )

print("---------")
print("Extrem Values Denoised L1")
print(denoised_L1.max())
print(denoised_L1.min())
print("---------")

print("---------")
print("Extrem Values Denoised L2")
print(denoised_L2.max())
print(denoised_L2.min())
print("---------")

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
plt.imshow(denoised_huberROF, cmap=plt.cm.gray)
plt.axis('off')
plt.title('TV denoising HuberROF', fontsize=20)

plt.subplots_adjust(wspace=0.02, hspace=0.1, top=0.9, bottom=0, left=0,
                    right=1)

#pic_directory = "D:\Mario\Studium\Studieren\Masterarbeit\\03_Daten"
#plt.savefig(pic_directory + "\denoise_tv_L2_iter=" + str(iter) + "_lambda=" + str(lambda_n) +".png" )

plt.get_current_fig_manager().window.showMaximized()
plt.show()

