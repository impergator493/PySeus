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


ref_img = np.array
ref_img = np.ndarray.astype(h5py.File("image_noise=0.05_gaussian.h5")['ref_images'][0][108], dtype=np.float64)
plt.imshow(ref_img, cmap='gray')
# mask with values greater 0 for Calculating SSIM
#plt.imshow(ref_img[108]>0, cmap='gray')
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
    print(ssim(ref_img_sc*mask, proc_img_sc*mask))
    print(mse(ref_img_sc*mask, proc_img_sc*mask))
    print(file)

    plt.subplot(2,6, i+1)
    plt.title(file)
    plt.imshow(proc_img, cmap='gray')

# Reconstruction
files = []
for i, file in enumerate(os.listdir(".")):
    if ".npy" in file and file.startswith("reconstructed"):
        files.append(file)


fig, axs = plt.subplots(5,6, figsize= (60,30))
fig.suptitle("Reconstructed")

for i, file in enumerate(files):
    proc_img = abs(np.load(file))
    proc_img_sc = (proc_img - proc_img.min())/(proc_img.max() - proc_img.min())
    print(ssim(ref_img_sc*mask, proc_img_sc*mask))
    print(mse(ref_img_sc*mask, proc_img_sc*mask))
    print(file)
    
    plt.subplot(5,6, i+1)
    plt.title(file)
    plt.imshow(abs(np.load(file)), cmap='gray')




