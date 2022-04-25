import matplotlib
import numpy as np
from pyseus.processing.tgv_reconstruction import TGV_Reco
from pyseus.processing.tv_reconstruction import TV_Reco
from pyseus.processing.tv_denoising import TV_Denoise
from pyseus.processing.tgv_denoising import TGV_Denoise
import h5py    
import numpy as np  
import matplotlib.pyplot as plt  
import scipy.io
import os 





parent_path = "..\\..\\03_Daten\\Noise_generation\\Mario_Master"
# file_paths_denoising = ["image_noise=0.05_gaussian.h5", "image_noise=0.05_rician.h5", 
#                 "image_noise=0.1_gaussian.h5", "image_noise=0.1_rician.h5",
#                 "image_noise=0.2_gaussian.h5", "image_noise=0.2_rician.h5"]

file_paths_denoising = ["image_noise=0.1_gaussian.h5", "image_noise=0.1_rician.h5",
"image_noise=0.1_gaussian.h5", "image_noise=0.1_rician.h5",
"image_noise=0.1_gaussian.h5", "image_noise=0.1_rician.h5",
"image_noise=0.2_gaussian.h5", "image_noise=0.2_rician.h5",
"image_noise=0.2_gaussian.h5", "image_noise=0.2_rician.h5",
"image_noise=0.2_gaussian.h5", "image_noise=0.2_rician.h5"]


#file_paths_denoising = []

denoise_params = [(150, 2, 1, 10000), (150, 2, 1, 10000),
                    (300, 2, 1, 10000), (300, 2, 1, 10000),
                    (400, 2, 1, 10000), (400, 2, 1, 10000),
                    (80, 2, 1, 10000), (80, 2, 1, 10000),
                    (200, 2, 1, 10000), (200, 2, 1, 10000),
                    (400, 2, 1, 10000), (400, 2, 1, 10000)]


# file_paths_reconstruction = ["kspace_noise=0.05.h5", "kspace_noise=0.05_perc_ones_kspace=0.2.h5", "kspace_noise=0.05_perc_ones_kspace=0.6.h5", 
#                 "kspace_noise=0.1.h5", "kspace_noise=0.1_perc_ones_kspace=0.2.h5", "kspace_noise=0.1_perc_ones_kspace=0.6.h5",
#                 "kspace_noise=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.6.h5"]

file_paths_reconstruction = ["kspace_noise=0.05.h5", "kspace_noise=0.05_perc_ones_kspace=0.2.h5", "kspace_noise=0.05_perc_ones_kspace=0.6.h5", 
                "kspace_noise=0.1.h5", "kspace_noise=0.1_perc_ones_kspace=0.2.h5", "kspace_noise=0.1_perc_ones_kspace=0.6.h5",
                "kspace_noise=0.2.h5", "kspace_noise=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.6.h5"]
    


# reco_params = [(1000, 2, 1, 10000), (1000, 2, 1, 10000), (1000, 2, 1, 10000),
#                     (1000, 2, 1, 10000), (1000, 2, 1, 10000), (1000, 2, 1, 10000),
#                     (1000, 2, 1, 10000), (1000, 2, 1, 10000), (1000, 2, 1, 10000)]

reco_params = [(2000, 2, 1, 10000), (3000, 2, 1, 10000), (2000, 2, 1, 10000),
                    (2000, 2, 1, 10000), (3000, 2, 1, 10000), (2000, 2, 1, 10000),
                    (2000, 2, 1, 10000), (500, 2, 1, 10000),  (1500, 2, 1, 10000), (800, 2, 1, 10000)]


# print('\n')
# print("Keys: ", list(f1.keys()))
# print("Attributes: ", dict(f1.attrs))
# Inhalt der Attribute



print(len(file_paths_denoising))
print(len(file_paths_reconstruction))
# For denoising
for i in range(len(file_paths_denoising)): 

    print("TGV Denoising")
    print(file_paths_denoising[i])
    f1 = h5py.File(os.path.join(parent_path, file_paths_denoising[i]))
    # für VFA_test_cart
    img_data = f1['img_dat'][0,108,:,:]


    #obj = TV_Denoise()
    obj = TGV_Denoise()

    #for tv: first argument in method
    #obj.tv_l2_reconstruction_gen
    denoised_data = obj.tgv2_denoising_gen(0, img_data, denoise_params[i], (1.0, 1.0))

    denoised_path = "denoised_TGV_" + file_paths_denoising[i] + "_" + str(denoise_params[i]) + ".npy"
    save_path = os.path.join(parent_path, denoised_path)

    with open(save_path,'wb') as file:
        np.save(file,denoised_data)
    


# For reco
for i in range(len(file_paths_reconstruction)): 


    print("TGV Reconstruction")
    print(file_paths_reconstruction[i])
    f1 = h5py.File(os.path.join(parent_path, file_paths_reconstruction[i]))
    real_dat = f1['real_dat'][0,:,108:109,:,:]
    imag_dat = f1['imag_dat'][0,:,108:109,:,:]
    raw_data = real_dat + imag_dat*(1j)
    coils = f1['Coils'][:,108:109,:,:]

    #obj = TV_Reco()
    obj = TGV_Reco()

    #for tv: first argument in method
    #obj.tv_l2_reconstruction_gen
    reco_data = obj.tgv2_reconstruction_gen(0, raw_data, coils, reco_params[i], (1.0, 1.0))

    reco_path = "reconstructed_TGV_" + file_paths_reconstruction[i] + "_" + str(reco_params[i]) + ".npy"
    save_path = os.path.join(parent_path, reco_path)

    with open(save_path,'wb') as file:
        np.save(file,reco_data)
    

# plt.figure()
# plt.subplot(3,3,1)
# plt.title("Denoised Reco Own")
# plt.imshow(abs(denoised_reco_own), cmap='gray')
# plt.subplot(3,3,2)
# plt.title("Denoised Reco Lus")
# plt.imshow(abs(denoised_reco_lus), cmap='gray')
# plt.subplot(3,3,3)
# plt.title("Denoised Reco Uniform")
# plt.imshow(abs(denoised_reco_un), cmap='gray')
# plt.subplot(3,3,4)
# plt.title("SOS IFFT Own Sparse mask")
# plt.imshow(abs(img_sp_own[0]), cmap='gray')
# plt.subplot(3,3,5)
# plt.title("SOS IFFT Sparse mask Lustig")
# plt.imshow(abs(img_sp_lus[0]), cmap='gray')
# plt.subplot(3,3,6)
# plt.title("SOS IFFT Sparse mask Uniform")
# plt.imshow(abs(img_sp_un[0]), cmap='gray')
# plt.subplot(3,3,7)
# plt.title("SOS IFFT Full Raw data")
# plt.imshow(abs(img_sos_raw[0]), cmap='gray')


# plt.figure()
# plt.subplot(2,3,1)
# plt.imshow(sp_mask_own[0], cmap='gray')
# plt.subplot(2,3,2)
# plt.imshow(sp_mask_lus[0], cmap='gray')
# plt.subplot(2,3,3)
# plt.imshow(sp_mask_un[0], cmap='gray')
# plt.subplot(2,3,4)
# plt.imshow(Z[0], cmap='gray')
# plt.subplot(2,3,5)
# plt.imshow(pdf_var_lus, cmap='gray')
# plt.subplot(2,3,6)
# plt.imshow(0.5*np.ones(pdf_var_lus.shape), cmap='gray')
# plt.show()




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

