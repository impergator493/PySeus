import numpy as np
from pyseus.denoising.tgv_reconstruction import TGV_Reco
import h5py    
import numpy as np  
import matplotlib.pyplot as plt  
import scipy.io


f1 = h5py.File("..\\..\\03_Daten\\fourier_data_reconstruction\\prob01.h5")    
print('\n')
print("Keys: ", list(f1.keys()))
print("Attributes: ", dict(f1.attrs))
# Inhalt der Attribute


real_dat = f1['real_dat'][9,:,64:65,:,:]
imag_dat = f1['imag_dat'][9,:,64:65,:,:]

img = real_dat + imag_dat*(1j)

print(img.shape, type(img[0,0]))

coils = f1['Coils'][:, 64:65,:,:]

# sparse matrix, for having a sparse k-space to demonstrate
sp_mat = np.random.choice([0,1], size=(224,224), p=[0.0, 1.0])

img_spat = np.fft.ifft2(img*sp_mat)


print("Max and Min Value k-space: ", abs(img).min(), abs(img).max())
print("Max and Min Value Spatial domain: ", abs(img_spat).min(), abs(img_spat).max())

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(abs(coils[0,0]))
# plt.subplot(1,3,2)
# plt.imshow(abs(np.imag(coils[0,0])))
# plt.subplot(1,3,3)
# plt.imshow(np.imag(coils[0,0]))
# plt.show()

# plt.figure()
# plt.imshow(np.log(abs(np.fft.fftshift(img[0,0]*sp_mat))), cmap = 'gray')

# plt.figure()
# plt.imshow((abs(img_spat[0,0]/coils[0,0])), vmin =5e-10,vmax =3.0e-8, cmap ='gray')
##cmap ='gray')
##vmin =0.3e-10,vmax =1.0e-8, cmap ='gray')
#plt.show()

#print(img_spat.min(), img_spat.max())


########


obj = TGV_Reco()

denoised_reco = obj.tgv2_reconstruction_gen(1,img, (coils, 0.005, 0.002,10))

np.save('denoise_u_veclist', denoised_reco)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(abs(denoised_reco), cmap='gray')
plt.show()


# noisy_3D = np.ones((1,256,256))
# noisy_3D[0,:,:] = noisy

# denoised_3D = obj.tgv2_denoising(noisy_3D,0.1,200,20)

# plt.subplot(1,2,2)
# plt.imshow(denoised_3D[0,:,:], cmap='gray')
# plt.show()
