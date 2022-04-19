import matplotlib
import numpy as np
from pyseus.processing.tgv_reconstruction import TGV_Reco
from pyseus.processing.tv_reconstruction import TV_Reco
import h5py    
import numpy as np  
import matplotlib.pyplot as plt  
import scipy.io

# f1 = h5py.File("..\\..\\03_Daten\\fourier_data_reconstruction\\prob01.h5")
#mit dem folgenden file gehts sicher
#f1 = h5py.File("..\\..\\03_Daten\\fourier_data_reconstruction\\VFA_test_cart.h5")      
f1 = h5py.File("..\\..\\03_Daten\\Noise_generation\\Mario_Master\\VFA_img_noise=0.1_alpha=9_fft_under60_noise.h5")      
print('\n')
print("Keys: ", list(f1.keys()))
print("Attributes: ", dict(f1.attrs))
# Inhalt der Attribute

# für VFA_test_cart
#real_dat = f1['real_dat'][0,:,100:101,:,:]
#imag_dat = f1['imag_dat'][0,:,100:101,:,:]
# für eigene datensets
real_dat = f1['real_dat'][0]
imag_dat = f1['imag_dat'][0]
raw_data = real_dat + imag_dat*(1j)
coils = f1['Coils'][:,100:101,:,:]
#coils = np.ones_like(raw_data)

#np.save('data_3D', raw_data)
#np.save('coil_3D', coils)


data_size = raw_data.shape[-3:]

img_sos_raw = coils.conjugate() * np.fft.ifft2(raw_data)
img_sos_raw = (np.sum(abs(img_sos_raw)**2, axis=0))**0.5

# sparse matrix, for having a sparse k-space to demonstrate
sp_mask_un = np.random.choice([0,1], size=data_size, p=[0.40, 0.60])
img_sp_un = np.fft.ifft2(raw_data*sp_mask_un)
img_sp_un = (np.sum(abs(img_sp_un)**2,axis=0))**0.5

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

#fully sampled with just ones or with var dens from lustig
sp_mask_lus = np.ones(raw_data.shape[-3:])
sp_mask_lus[:,:] = mask_var[y1:y2,x1:x2]
pdf_var_lus = pdf_var[y1:y2,x1:x2]

img_sp_lus = np.fft.ifft2(raw_data*sp_mask_lus)
img_sp_lus = (np.sum(abs(img_sp_lus)**2,axis=0))**0.5

# Generate mask with variable density with gaussian distribution

std = 0.12

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
#Z = PDF
Z = (1. / (std * np.sqrt(2 * np.pi))) * np.exp(-.5*(R**2 / std))
Z[0] = Z[0]/max(Z.ravel())*1 + 0.4
Z[0][Z[0]>1] = 1
# with current setting, 30% are ones
sp_mask_own = (Z >= sp_un)
# comparison: pdf_var from lusti has 33%, so its perfect

# sparse matrix, for having a sparse k-space to demonstrate
img_sp_own = np.fft.ifft2(raw_data*sp_mask_own/Z)
img_sp_own = (np.sum(img_sp_own**2,axis=0))**0.5

#number of 1 in sp_mask
#sp_mask = sp_un * Z
#sp_mask_bin = sp_mask > 0.2
#sp_mask_bin[R<0.15] = 1


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

obj = TV_Reco()
#obj = TGV_Reco()

#for tv: first argument in method
#obj.tv_l2_reconstruction_gen
denoised_reco_own = obj.tv_reconstruction_gen(obj.tv_l2_reconstruction, 0, raw_data, coils, (10, 3), (1.0, 1.0))



#TGV
#denoised_reco_own = obj.tgv2_reconstruction_gen(0, raw_data, coils, sp_mask_own, (10, 2,1, 100), (1.0, 1.0))
#denoised_reco_lus = obj.tgv2_reconstruction_gen(0, raw_data, coils, sp_mask_lus, (10, 2,1, 100), (1.0, 1.0))
#denoised_reco_un = obj.tgv2_reconstruction_gen(0, raw_data, coils, sp_mask_un, (10, 2,1, 100), (1.0, 1.0))

print("own mask: ", str(np.count_nonzero(sp_mask_own)/len(np.ravel(sp_mask_own))))
print("lustig mask: ", str(np.count_nonzero(sp_mask_lus)/len(np.ravel(sp_mask_lus))))
print("unified mask: ", str(np.count_nonzero(sp_mask_un)/len(np.ravel(sp_mask_un))))


#np.save('denoise_u_veclist', denoised_reco)

plt.figure()
plt.subplot(3,3,1)
plt.title("Denoised Reco Own")
plt.imshow(abs(denoised_reco_own), cmap='gray')
plt.subplot(3,3,2)
plt.title("Denoised Reco Lus")
plt.imshow(abs(denoised_reco_lus), cmap='gray')
plt.subplot(3,3,3)
plt.title("Denoised Reco Uniform")
plt.imshow(abs(denoised_reco_un), cmap='gray')
plt.subplot(3,3,4)
plt.title("SOS IFFT Own Sparse mask")
plt.imshow(abs(img_sp_own[0]), cmap='gray')
plt.subplot(3,3,5)
plt.title("SOS IFFT Sparse mask Lustig")
plt.imshow(abs(img_sp_lus[0]), cmap='gray')
plt.subplot(3,3,6)
plt.title("SOS IFFT Sparse mask Uniform")
plt.imshow(abs(img_sp_un[0]), cmap='gray')
plt.subplot(3,3,7)
plt.title("SOS IFFT Full Raw data")
plt.imshow(abs(img_sos_raw[0]), cmap='gray')


plt.figure()
plt.subplot(2,3,1)
plt.imshow(sp_mask_own[0], cmap='gray')
plt.subplot(2,3,2)
plt.imshow(sp_mask_lus[0], cmap='gray')
plt.subplot(2,3,3)
plt.imshow(sp_mask_un[0], cmap='gray')
plt.subplot(2,3,4)
plt.imshow(Z[0], cmap='gray')
plt.subplot(2,3,5)
plt.imshow(pdf_var_lus, cmap='gray')
plt.subplot(2,3,6)
plt.imshow(0.5*np.ones(pdf_var_lus.shape), cmap='gray')
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

