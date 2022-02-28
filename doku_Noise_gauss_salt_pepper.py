import matplotlib.pyplot as plt
#import matplotlib.colors as col
import scipy.io
import cv2
import numpy as np
from skimage.util import random_noise


mean = 0
var = 0.01
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (224, 224))

img_raw = np.zeros((224,224))+0.5

sp_noise = random_noise(img_raw, mode='s&p',amount=0.3)
gauss_noise = random_noise(img_raw, mode='gaussian',mean=mean, var=var)

fig = plt.figure()
fig.suptitle("Noise histogram: S&P - Gaussian")
plt.subplot(1,2,1)
plt.hist(sp_noise.ravel())
plt.xlabel("Pixel intensity")
plt.ylabel("Number of pixels")
plt.subplot(1,2,2)
plt.hist(gauss_noise.ravel(), bins=256)
plt.xlabel("Pixel intensity")
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(sp_noise,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(gauss_noise,cmap='gray')
plt.show()