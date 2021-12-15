import numpy as np
from pyseus.denoising.tgv_3D import TGV_3D
from pyseus.denoising.tgv import TGV
import scipy.io
import matplotlib.pyplot as plt


obj_2D = TGV()
noisy = scipy.io.loadmat('./tests/cameraman_noise.mat')['im']

denoised_2D = obj_2D.tgv2_denoising(noisy, 10,200,20)

plt.figure()
plt.imshow(denoised_2D, cmap='gray')
plt.show()

obj = TGV_3D()

noisy = np.ones((4,5,6))

denoised = obj.tgv2_3D_denoising(noisy,0.01,0.04,10)
print(denoised)
