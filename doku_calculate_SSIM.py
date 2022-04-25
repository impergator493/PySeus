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



parent_path = "..\\..\\03_Daten\\Noise_generation\\Mario_Master"
file_paths_denoising = ["image_noise=0.05_gaussian.h5", "image_noise=0.05_rician.h5", 
                "image_noise=0.1_gaussian.h5", "image_noise=0.1_rician.h5",
                "image_noise=0.2_gaussian.h5", "image_noise=0.2_rician.h5"]

denoise_params = [(400, 10000), (400, 10000),
                    (200, 10000), (200, 10000),
                    (100, 10000), (100, 10000)]


file_paths_reconstruction = ["kspace_noise=0.05.h5", "kspace_noise=0.05_perc_ones_kspace=0.2.h5", "kspace_noise=0.05_perc_ones_kspace=0.6.h5", 
                "kspace_noise=0.1.h5", "kspace_noise=0.1_perc_ones_kspace=0.2.h5", "kspace_noise=0.1_perc_ones_kspace=0.6.h5",
                "kspace_noise=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.2.h5", "kspace_noise=0.2_perc_ones_kspace=0.6.h5"]        

reco_params = [(400, 10000), (400, 10000), (400, 10000),
                    (200, 10000), (200, 10000), (200, 10000),
                    (100, 10000), (100, 10000), (100, 10000)]

# print('\n')
# print("Keys: ", list(f1.keys()))
# print("Attributes: ", dict(f1.attrs))
# Inhalt der Attribute

# os.listdir()
os.chdir(parent_path)
size_letters = 7

ref_img = np.ndarray.astype(h5py.File("image_noise=0.05_gaussian.h5")['ref_images'][0][108], dtype=np.float64)
ref_img_gaussian = np.ndarray.astype(h5py.File("image_noise=0.2_gaussian.h5")['img_dat'][0][108], dtype=np.float64)
ref_img_rician = np.ndarray.astype(h5py.File("image_noise=0.2_rician.h5")['img_dat'][0][108], dtype=np.float64)

fig, axs = plt.subplots(1,3)
fig.suptitle("Comparison original, gaussian, rician images")
plt.subplot(1,3,1)
plt.imshow(ref_img, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(ref_img_gaussian, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(ref_img_rician, cmap='gray')

mask = ref_img>0
ref_img_sc = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())


# Display all the results.
files = []
for i, file in enumerate(os.listdir(".")):
    if ".npy" in file and file.startswith("denoised"):
        files.append(file)

fig, axs = plt.subplots(2,6, figsize= (40,15))
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
    
    plt.subplot(2,6, i+1)
    plt.title(str(file).replace('denoised_', '').replace('.npy', '') + "\n" + ssim_t + "\n" + mse_t + "\n" + nrmse_t + "\n" + psnr_t , fontsize=size_letters)
    plt.imshow(proc_img, cmap='gray')

# Reconstruction
files = []
for i, file in enumerate(os.listdir(".")):
    if ".npy" in file and file.startswith("reconstructed"):
        files.append(file)


fig, axs = plt.subplots(2,15, figsize= (100, 30))
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
    
    plt.subplot(2,15, i+1)
    plt.subplots_adjust(left=0.03, right=0.97, bottom= 0.03, top= 0.9, wspace=0.6, hspace = 0.42)
    #plt.title(str(file).replace('reconstructed_', '').replace('.npy','').replace('kspace_','') + "\n" + ssim_t, fontsize=size_letters)
    plt.title(str(file).replace('reconstructed_', '').replace('.npy','').replace('_kspace','').replace('_ones','') + "\n" + ssim_t + "\n" + mse_t + "\n" + nrmse_t + "\n" + psnr_t , fontsize=size_letters)
    plt.imshow(abs(np.load(file)), cmap='gray')


#plt.tight_layout()
plt.subplot_tool()
plt.show()



