#%% load basics

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
import os, glob
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr

from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'qt')



parent_path = "..\\..\\03_Daten\\Noise_generation\\Mario_Master\\"
# file_paths_denoising = ["image_noise=0.05_gaussian.h5", "image_noise=0.05_rician.h5", 
#                 "image_noise=0.1_gaussian.h5", "image_noise=0.1_rician.h5",
#                 "image_noise=0.2_gaussian.h5", "image_noise=0.2_rician.h5"]

# denoise_params = [(400, 10000), (400, 10000),
#                     (200, 10000), (200, 10000),
#                     (100, 10000), (100, 10000)]


# file_paths_reconstruction = ["kspace_noise=0.05.h5", "kspace_noise=0.05_perc_ones_kspace=0.2.h5", "kspace_noise=0.05_perc_ones_kspace=0.6.h5", 
#                 "kspace_noise=0.1.h5", "kspace_noise=0.1_perc_ones_kspace=0.2.h5", "kspace_noise=0.1_perc_ones_kspace=0.6.h5",
#                 "kspace_noise=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.6.h5"]        

# reco_params = [(400, 10000), (400, 10000), (400, 10000),
#                     (200, 10000), (200, 10000), (200, 10000),
#                     (100, 10000), (100, 10000), (100, 10000)]

# print('\n')
# print("Keys: ", list(f1.keys()))
# print("Attributes: ", dict(f1.attrs))
# Inhalt der Attribute


#print(os.listdir())
os.chdir(parent_path)
size_letters = 10




ref_img = np.ndarray.astype(h5py.File("image_noise=0.05_gaussian.h5")['ref_images'][0][108], dtype=np.float64)
ref_img_005 = np.ndarray.astype(h5py.File("image_noise=0.05_gaussian.h5")['img_dat'][0][108], dtype=np.float64)
ref_img_010 = np.ndarray.astype(h5py.File("image_noise=0.1_gaussian.h5")['img_dat'][0][108], dtype=np.float64)
ref_img_020 = np.ndarray.astype(h5py.File("image_noise=0.2_gaussian.h5")['img_dat'][0][108], dtype=np.float64)

mask = ref_img>0
ref_img_sc = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())

os.chdir("doc_final_results\\")

#%% Raw noisy image model generation 
fig, axs = plt.subplots(2,2)
#fig.suptitle("Image data with gaussian noise of \n 5, 10 and 20% standard deviation")
plt.subplot(2,2,1)
plt.title("Without noise", fontsize=size_letters)
plt.imshow(ref_img, cmap='gray')
plt.axis('off')
plt.subplot(2,2,2)
plt.title("5% Gaussian noise", fontsize=size_letters)
plt.imshow(ref_img_005, cmap='gray')
plt.axis('off')
plt.subplot(2,2,3)
plt.title("10% Gaussian noise", fontsize=size_letters)
plt.imshow(ref_img_010, cmap='gray')
plt.axis('off')
plt.subplot(2,2,4)
plt.title("20% Gaussian noise", fontsize=size_letters)
plt.imshow(ref_img_020, cmap='gray')
plt.axis('off')
plt.subplot_tool()

#%% Denoising TV comparison - gaussian noise 0.05, 0.1 0.2 400/150/100 lambda
tv_gauss_005 = np.load("denoised_TV_image_noise=0.05_gaussian.h5_(400, 10000).npy")
tv_gauss_010 = np.load("denoised_TV_image_noise=0.1_gaussian.h5_(150, 10000).npy")
tv_gauss_020 = np.load("denoised_TV_image_noise=0.2_gaussian.h5_(100, 10000).npy")

proc_tv_005 = (tv_gauss_005 - tv_gauss_005.min())/(tv_gauss_005.max() - tv_gauss_005.min())
ssim_tv_005 = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_tv_005*mask), 3))
nrmse_tv_005 = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_tv_005*mask), 3))

proc_tv_010 = (tv_gauss_010 - tv_gauss_010.min())/(tv_gauss_010.max() - tv_gauss_010.min())
ssim_tv_010 = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_tv_010*mask), 3))
nrmse_tv_010 = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_tv_010*mask), 3))

proc_tv_020 = (tv_gauss_020 - tv_gauss_020.min())/(tv_gauss_020.max() - tv_gauss_020.min())
ssim_tv_020 = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_tv_020*mask), 3))
nrmse_tv_020 = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_tv_020*mask), 3))



fig, axs = plt.subplots(3,1)
#fig.suptitle("Image data with gaussian noise of \n 5, 10 and 20% standard deviation")
plt.subplot(3,1,1)
plt.title("5% noise" + "\n" + ssim_tv_005 + "\n" + nrmse_tv_005, fontsize=size_letters)
plt.imshow(tv_gauss_005, cmap='gray')
plt.axis('off')
plt.subplot(3,1,2)
plt.title("10% noise" + "\n" + ssim_tv_010 + "\n" + nrmse_tv_010, fontsize=size_letters)
plt.imshow(tv_gauss_010, cmap='gray')
plt.axis('off')
plt.subplot(3,1,3)
plt.title("20% noise" + "\n" + ssim_tv_020 + "\n" + nrmse_tv_020, fontsize=size_letters)
plt.imshow(tv_gauss_020, cmap='gray')
plt.axis('off')
plt.subplot_tool()

#%% Denoising gaussian TV - TGV 0.05 lambda 400
tv_gauss_005 = np.load("denoised_TV_image_noise=0.05_gaussian.h5_(400, 10000).npy")
tgv_gauss_005 = np.load("denoised_TGV_image_noise=0.05_gaussian.h5_(400, 2, 1, 10000).npy")

proc_tv = (tv_gauss_005 - tv_gauss_005.min())/(tv_gauss_005.max() - tv_gauss_005.min())
ssim_tv = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_tv*mask), 3))
nrmse_tv = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_tv*mask), 3))

proc_tgv = (tgv_gauss_005 - tgv_gauss_005.min())/(tgv_gauss_005.max() - tgv_gauss_005.min())
ssim_tgv = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_tgv*mask), 3))
nrmse_tgv = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_tgv*mask), 3))

fig, axs = plt.subplots(1,2)
#fig.suptitle("Image data with gaussian noise of \n 5, 10 and 20% standard deviation")
plt.subplot(1,2,1)
plt.title("TV" + "\n" + ssim_tv + "\n" + nrmse_tv, fontsize=size_letters)
plt.imshow(tv_gauss_005, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title("TGV" + "\n" + ssim_tgv + "\n" + nrmse_tgv, fontsize=size_letters)
plt.imshow(tgv_gauss_005, cmap='gray')
plt.axis('off')
plt.subplot_tool()

#%% Denoising Gaussian - Rician 0.1 lambda 150 TV 
tv_gauss = np.load("denoised_TV_image_noise=0.1_gaussian.h5_(150, 10000).npy")
tv_rician = np.load("denoised_TV_image_noise=0.1_rician.h5_(150, 10000).npy")

proc_gauss = (tv_gauss - tv_gauss.min())/(tv_gauss.max() - tv_gauss.min())
ssim_gauss = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_gauss*mask), 3))
nrmse_gauss = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_gauss*mask), 3))

proc_rician = (tv_rician - tv_rician.min())/(tv_rician.max() - tv_rician.min())
ssim_rician = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_rician*mask), 3))
nrmse_rician = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_rician*mask), 3))

fig, axs = plt.subplots(1,2)
#fig.suptitle("Image data with gaussian noise of \n 5, 10 and 20% standard deviation")
plt.subplot(1,2,1)
plt.title("Gaussian" + "\n" + ssim_gauss + "\n" + nrmse_gauss, fontsize=size_letters)
plt.imshow(tv_gauss, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title("Rician" + "\n" + ssim_rician + "\n" + nrmse_rician, fontsize=size_letters)
plt.imshow(tv_rician, cmap='gray')
plt.axis('off')
plt.subplot_tool()


#%% Reconstruction TV - TGV 0.05  lambda 400/2000
rec_tv_gauss_005 = abs(np.load("reconstructed_TV_kspace_noise=0.05.h5_(400, 10000).npy"))
rec_tgv_gauss_005 = abs(np.load("reconstructed_TGV_kspace_noise=0.05.h5_(2000, 2, 1, 10000).npy"))

proc_rec_tv = (rec_tv_gauss_005 - rec_tv_gauss_005.min())/(rec_tv_gauss_005.max() - rec_tv_gauss_005.min())
ssim_rec_tv = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_rec_tv*mask), 3))
nrmse_rec_tv = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_rec_tv*mask), 3))

proc_rec_tgv = (rec_tgv_gauss_005 - rec_tgv_gauss_005.min())/(rec_tgv_gauss_005.max() - rec_tgv_gauss_005.min())
ssim_rec_tgv = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_rec_tgv*mask), 3))
nrmse_rec_tgv = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_rec_tgv*mask), 3))

fig, axs = plt.subplots(1,2)
#fig.suptitle("Image data with gaussian noise of \n 5, 10 and 20% standard deviation")
plt.subplot(1,2,1)
plt.title("TV" + "\n" + ssim_rec_tv + "\n" + nrmse_rec_tv, fontsize=size_letters)
plt.imshow(rec_tv_gauss_005, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title("TGV" + "\n" + ssim_rec_tgv + "\n" + nrmse_rec_tgv, fontsize=size_letters)
plt.imshow(rec_tgv_gauss_005, cmap='gray')
plt.axis('off')
plt.subplot_tool()


#%% Reconstruction undersampling comparison - 0.1 noise, 200/300/600 lamba 100% 60% 20% kspace daten
rec_us100 = abs(np.load("reconstructed_TV_kspace_noise=0.1.h5_(200, 10000).npy"))
rec_us60 = abs(np.load("reconstructed_TV_kspace_noise=0.1_perc_ones_kspace=0.6.h5_(300, 10000).npy"))
rec_us20 = abs(np.load("reconstructed_TV_kspace_noise=0.1_perc_ones_kspace=0.2.h5_(600, 10000).npy"))

proc_us100 = (rec_us100 - rec_us100.min())/(rec_us100.max() - rec_us100.min())
ssim_us100 = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_us100*mask), 3))
nrmse_us100 = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_us100*mask), 3))

proc_us60 = (rec_us60 - rec_us60.min())/(rec_us60.max() - rec_us60.min())
ssim_us60 = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_us60*mask), 3))
nrmse_us60 = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_us60*mask), 3))

proc_us20 = (rec_us20 - rec_us20.min())/(rec_us20.max() - rec_us20.min())
ssim_us20 = "SSIM: " + str(round(ssim(ref_img_sc*mask, proc_us20*mask), 3))
nrmse_us20 = "NRMSE: " + str(round(nrmse(ref_img_sc*mask, proc_us20*mask), 3))


fig, axs = plt.subplots(3,1)
plt.subplot(3,1,1)
plt.title("No undersampling" + "\n" + ssim_us100 + "\n" + nrmse_us100, fontsize=size_letters)
plt.imshow(rec_us100, cmap='gray')
plt.axis('off')
plt.subplot(3,1,2)
plt.title("60% of full kspace" + "\n" + ssim_us60 + "\n" + nrmse_us60, fontsize=size_letters)
plt.imshow(rec_us60, cmap='gray')
plt.axis('off')
plt.subplot(3,1,3)
plt.title("20% of full kspace" + "\n" + ssim_us20 + "\n" + nrmse_us20, fontsize=size_letters)
plt.imshow(rec_us20, cmap='gray')
plt.axis('off')
plt.subplot_tool()



#%% Old code


# Display all the results.
files = []
for i, file in enumerate(os.listdir(".")):
    if ".npy" in file and file.startswith("denoised"):
        files.append(file)

fig, axs = plt.subplots(3,4, figsize= (60,40))
fig.suptitle("Denoising")

for i, file in enumerate(files):
    proc_img = np.load(file)
    proc_img_sc = (proc_img - proc_img.min())/(proc_img.max() - proc_img.min())
    ssim_t = "SSIM: " + str(ssim(ref_img_sc*mask, proc_img_sc*mask))
    mse_t = "MSE: " + str(mse(ref_img_sc*mask, proc_img_sc*mask))
    nrmse_t = "NRMSE: " + str(nrmse(ref_img_sc*mask, proc_img_sc*mask))
    psnr_t = "PSNR: " + str(psnr(ref_img_sc*mask, proc_img_sc*mask))

    # print(ssim_t)
    # print(mse_t)
    # print(nrmse_t)
    # print(psnr_t)
    plt.subplot(3,4, i+1)
    #plt.title(str(file).replace('denoised_', '').replace('.npy', '') + "\n" + ssim_t + "\n" + mse_t + "\n" + nrmse_t + "\n" + psnr_t , fontsize=size_letters)
    #plt.subplots_adjust(left=0.03, right=0.97, bottom= 0.03, top= 0.9, wspace=0.2, hspace = 0.5)
    plt.title(str(file).replace('denoised_', '').replace('.npy', '').replace('_image','').replace(', 10000','') + "\n" + ssim_t + "\n" + nrmse_t, fontsize=size_letters)
    plt.imshow(proc_img, cmap='gray')
    plt.axis('off')



# Reconstruction
files = []
for i, file in enumerate(os.listdir(".")):
    if ".npy" in file and file.startswith("reconstructed"):
        files.append(file)


fig, axs = plt.subplots(3,6, figsize= (100, 30))
fig.suptitle("Reconstructed")

for i, file in enumerate(files):
    proc_img = abs(np.load(file))
    proc_img_sc = (proc_img - proc_img.min())/(proc_img.max() - proc_img.min())
    ssim_t = "SSIM: " + str(ssim(ref_img_sc*mask, proc_img_sc*mask))
    mse_t = "MSE: " + str(mse(ref_img_sc*mask, proc_img_sc*mask))
    nrmse_t = "NRMSE: " + str(nrmse(ref_img_sc*mask, proc_img_sc*mask))
    psnr_t = "PSNR: " + str(psnr(ref_img_sc*mask, proc_img_sc*mask))

    # print(ssim_t)
    # print(mse_t)
    # print(nrmse_t)
    # print(psnr_t)
    
    plt.subplot(3,6, i+1)
    #plt.subplots_adjust(left=0.03, right=0.97, bottom= 0.03, top= 0.9, wspace=0.2, hspace = 0.2)
    plt.title(str(file).replace('reconstructed_', '').replace('.npy','').replace('_kspace','').replace('_ones','').replace('2, 1,','').replace(' 10000','') + "\n" + ssim_t + "\n" + nrmse_t, fontsize=size_letters)
    #plt.title(str(file).replace('reconstructed_', '').replace('.npy','').replace('_kspace','').replace('_ones','') + "\n" + ssim_t + "\n" + mse_t + "\n" + nrmse_t + "\n" + psnr_t , fontsize=size_letters)
    plt.imshow(abs(np.load(file)), cmap='gray')
    plt.axis('off')


#plt.tight_layout()
#plt.subplot_tool()
plt.show()




# %%
