import numpy as np
from pyseus.processing.tgv import TGV

import scipy.io
import matplotlib.pyplot as plt


obj = TGV()
noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']

noisy = noisy[np.newaxis,...]

denoised_2D = obj.tgv2_denoising(noisy, 20, 2, 1 ,20)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(denoised_2D[0,:,:], cmap='gray')


noisy_3D = np.ones((1,256,256))
noisy_3D[0,:,:] = noisy

denoised_3D = obj.tgv2_denoising(noisy_3D, 0.001, 2, 1, 20)

plt.subplot(1,2,2)
plt.imshow(denoised_3D[0,:,:], cmap='gray')
plt.show()
