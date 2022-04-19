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
rician = np.random.normal(mean, sigma, (224, 224)) + 1j*np.random.normal(mean, sigma, (224, 224))

img_raw = np.zeros((224,224))+0.5

sp_noise = random_noise(img_raw, mode='s&p',amount=0.3)
#gauss_noise = random_noise(img_raw, mode='gaussian',mean=mean, var=var)
gauss_noise = img_raw + gaussian
rician_noise = abs(img_raw + rician)

fig = plt.figure()
fig.suptitle("Noise histogram low SNR: S&P - Gaussian - Rician")
plt.subplot(1,3,1)
plt.hist((np.random.choice([0,1], size=(224, 224), p=[0.50, 0.50])).ravel())
plt.xlabel("Pixel intensity")
plt.subplot(1,3,2)
plt.hist(gaussian.ravel(), bins=256)
plt.xlabel("Pixel intensity")
plt.subplot(1,3,3)
plt.hist(abs(rician.ravel()), bins=256)
plt.xlabel("Pixel intensity")
plt.show()

fig = plt.figure()
fig.suptitle("Noise histogram high SNR: S&P - Gaussian - Rician")
plt.subplot(1,3,1)
plt.hist(sp_noise.ravel())
plt.xlabel("Pixel intensity")
plt.ylabel("Number of pixels")
plt.subplot(1,3,2)
plt.hist(gauss_noise.ravel(), bins=256)
plt.xlabel("Pixel intensity")
plt.subplot(1,3,3)
plt.hist(rician_noise.ravel(), bins=256)
plt.xlabel("Pixel intensity")
plt.show()

fig = plt.figure()
fig.suptitle("Homogenous image with added noise: S&P - Gaussian - Rician")
plt.subplot(1,3,1)
plt.imshow(sp_noise,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(gauss_noise,cmap='gray')
plt.subplot(1,3,3)
plt.imshow(rician_noise,cmap='gray')
plt.show()