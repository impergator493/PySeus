import numpy as np
from pyseus.denoising.tgv_3D import TGV_3D

obj = TGV_3D()

noisy = np.ones((4,5,6))

denoised = obj.tgv2_denoising(noisy,0.01,0.04,10)
print(denoised)
