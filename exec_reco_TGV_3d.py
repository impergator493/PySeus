import numpy as np
from pyseus.processing.tgv_reconstruction import TGV_Reco
import h5py    
import numpy as np  
import matplotlib.pyplot as plt  
import scipy.io

f1 = h5py.File("..\\..\\03_Daten\\fourier_data_reconstruction\\prob01.h5")    
print('\n')
print("Keys: ", list(f1.keys()))
print("Attributes: ", dict(f1.attrs))
# Inhalt der Attribute


real_dat = f1['real_dat'][0,:,30:31,:,:]
imag_dat = f1['imag_dat'][0,:,30:31,:,:]
raw_data = real_dat + imag_dat*(1j)
coils = f1['Coils'][:,30:31,:,:]
#coils = np.ones_like(raw_data)

np.save('data_3D', raw_data)
np.save('coil_3D', coils)


data_size = raw_data.shape[-3:]

img_sos_raw = coils.conjugate() * np.fft.ifft2(raw_data)
img_sos_raw = (np.sum(img_sos_raw**2, axis=0))**0.5

# sparse matrix, for having a sparse k-space to demonstrate
sp_mat_un = np.random.choice([0,1], size=data_size, p=[0.5, 0.5])
img_sp_un = np.fft.ifft2(raw_data*sp_mat_un)
img_sp_sos = (np.sum(img_sp_un**2,axis=0))**0.5

mat = scipy.io.loadmat("..\\..\\03_Daten\\brain.mat")
data_raw = mat['im']
pdf_un = mat['pdf_unif']
pdf_var = mat['pdf_vardens']
mask_un = mat['mask_unif']
mask_var = mat['mask_vardens']

# take mask_var densitiy and fit it to current Raw data x and y size
y1 = mask_var.shape[0]//2-data_size[-2]//2
y2 = mask_var.shape[0]//2+data_size[-2]//2
x1 = mask_var.shape[1]//2-data_size[-1]//2
x2 = mask_var.shape[1]//2+data_size[-1]//2

#fully sampled with just ones
sp_mask_bin2 = np.ones_like(raw_data)
#sp_mask_bin2[:,:] = mask_var[y1:y2,x1:x2]

# Generate mask with variable density with gaussian distribution

std = 0.4

sp_un = np.random.uniform(low=0.0, high=1, size=data_size)
x = np.linspace(-1, 1, data_size[-1])
y = np.linspace(-1, 1, data_size[-2])
if data_size[-3] == 1:
    z = 0
elif data_size[-3] == 2:
    z = [0,0]
else: z = np.linspace(-1, 1, data_size[-3])
zv, yv, xv = np.meshgrid(z, y, x, indexing='ij')

R = (xv**2 + yv**2 + zv**2)**0.5
Z = (1. / (std * np.sqrt(2 * np.pi))) * np.exp(-.5*(R**2 / std))
sp_mask = sp_un * Z

sp_mask_bin = sp_mask > 0.2
sp_mask_bin[R<0.15] = 1


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

denoised_reco = obj.tgv2_reconstruction_gen(0,raw_data, coils, sp_mask_bin2, 10, 2, 1,10)

#np.save('denoise_u_veclist', denoised_reco)

plt.figure()
plt.subplot(1,3,1)
plt.title("SOS IFFT Full Raw data")
plt.imshow(abs(img_sos_raw[0]), cmap='gray')
plt.subplot(1,3,2)
plt.title("Denoised Reco")
plt.imshow(abs(denoised_reco), cmap='gray')
plt.subplot(1,3,3)
plt.title("SOS IFFT Sparse mask Uniform")
plt.imshow(abs(img_sp_sos[0]), cmap='gray')
plt.show()

# plt.figure()
# plt.subplot(1,3,1)
# plt.title("SOS IFFT Full Raw data")
# plt.imshow(abs(img_sos_raw[3]), cmap='gray')
# plt.subplot(1,3,2)
# plt.title("Denoised Reco")
# plt.imshow(abs(denoised_reco[3]), cmap='gray')
# plt.subplot(1,3,3)
# plt.title("SOS IFFT Sparse mask Uniform")
# plt.imshow(abs(img_sp_sos[3]), cmap='gray')
# plt.show()





# plt für 2D several slices
# plt.figure()
# for i in range(1,7):
#     plt.subplot(1,6,i)
#     plt.imshow(abs(denoised_reco[i-1,:,:]), cmap='gray')
# plt.show()

