# Assignment from Image Processing, taken from there

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.recfunctions import rec_append_fields
import scipy
import scipy.sparse as sp

# @TODO: womöglich als methode in TV klasse inkludieren?
class TGV_Reco(): 


    def __init__(self):
        pass


    def _make_nabla(self,L, M, N):
        row = np.arange(0, L * M * N)
        dat = np.ones(L* M * N)
        col = np.arange(0, M * N * L).reshape(L, M, N)
        col_xp = np.concatenate([col[:, :, 1:], col[:, :, -1:]], axis = 2)
        col_yp = np.concatenate([col[:, 1:, :], col[:, -1:, :]], axis = 1)
        col_zp = np.concatenate([col[1:, :, :], col[-1:, :, :]], axis = 0)

        # flatten vector contains all the indices for the positions where -1 and 1 should be placed to calculate
        # gradient for every pixel in all 3 dimensions.
        
        # for every pixel (pixel amount L*M*N) that should be calculated, a sparse vector (length L*M*N) is generated 
        # which just contains the 1 and -1 on the 
        # specific place and is 0 otherwhise. Thats why its a (L*M*N, L*M*N) matrix

        nabla_x = scipy.sparse.coo_matrix((dat, (row, col_xp.flatten())), shape=(L * M * N, L * M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N))

        nabla_y = scipy.sparse.coo_matrix((dat, (row, col_yp.flatten())), shape=(L * M * N, L * M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N))
        
        nabla_z = scipy.sparse.coo_matrix((dat, (row, col_zp.flatten())), shape=(L * M * N, L * M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N))



        nabla = scipy.sparse.vstack([nabla_x, nabla_y, nabla_z])

        return nabla, nabla_x, nabla_y, nabla_z


    def prox_G(self, x, uk, tau, gamma):
        """
        Used for calculation of the dataterm projection

        compute pi with pi = \bar x + tau * W_i
        @param u: MN
        @param tau: scalar
        @param Wis: MN x K
        @param f: MN x K
        """
      
        prox = (uk*tau/gamma + x)/(1 + tau/gamma)

        # bei nur 2 einträgen (f, pis) macht median dasselbe wie mean und nimmt den mittelwert
        # egal wieviele einträge, über eine achse nimmt er nur den median und diese
        # dimension verschwindet dann sogar, d.h. aus shape (M*N,2) wird dann (M*N,)
        # Rückgabe hat also K dimension wieder weniger
        # pis = u[...] + tau
        # prox = np.median((uk,pis), axis=0)

        return prox


    def make_K(self, L, M, N):
        """
        @param M:
        @param N:
        @return: the K operator as described in Equation (5)
        """
        nabla, nabla_x, nabla_y, nabla_z = self._make_nabla(L, M, N)
        neg_I = sp.identity(L*M*N) * -1

        K = sp.bmat([[nabla_x, neg_I, None, None], [nabla_y, None, neg_I, None], [nabla_z, None, None, neg_I], \
            [None, nabla_x, None, None], [None, nabla_y, None, None], [None, nabla_z, None, None], \
            [None, None, nabla_x, None], [None, None, nabla_y, None], [None, None, nabla_z, None],
            [None, None, None, nabla_x], [None, None, None, nabla_y], [None, None, None, nabla_z]])

        return K


    def proj_ball(self, Z, alpha):
        """
        Projection to a ball as described in Equation (6)
        @param Y: either 2xMN or 4xMN
        @param alpha: scalar hyperparameter alpha
        @return: projection result either 2xMN or 4xMN
        """
        norm = np.linalg.norm(Z, axis=0)
        projection = Z / np.maximum(1, 1/alpha * norm)
    
        return projection


 
    #@TODO temporarily absolute value of coils sensitivities just for trying
    def DA(self, u, sens_c):
        """
        input parameter:

        u - current reconstructed sample in spatial domain, size (L,M,N)
        sens_c - coil sensitivities 
        """
        return np.fft.fftn((sens_c * u), axes=(-3,-2,-1))
        

        
    #@TODO temporaril absolute value of coils sensitivities just for trying
    def DAH(self, r, sens_c):

        """
        input parameter:

        R - dual variable of difference of current reconstructed sample u(n) in fourier domain and initial fourier data
        """
        
        r_IFT = sens_c.conjugate() * np.fft.ifftn(r,axes=(-3,-2,-1))

        
        return np.sum( r_IFT, axis=0)


    def proj_L2(self, R, sigma, d_init):
        
        return (R-sigma*d_init)/(1+ sigma)

   # if its a big dataset, a lot of RAM is needed because all the raw data to process will be 
    # stored in the RAM
    def tgv2_reconstruction_gen(self, dataset_type, data_raw, data_coils, alpha0, alpha1, iter):

        params = (alpha0,alpha1,iter)

        if dataset_type == 1:
            
            # prepare artifical 3D dataset(1,M,N) for 2D image (M,N), because to be universal applicable 
            # at least one entry in 3rd dimension is expected
            # Because of Coil data, correct slice has to be select with L=1 already when method is called


            # dat_real, imag are of dimension (C*L*M*N), if 2D L is just length 1
            # make return value 2D array again
            dataset_denoised = self.tgv2_reconstruction(data_raw, data_coils, *params)[0,:,:]

            return dataset_denoised

        elif dataset_type == 2:
            
            dataset_denoised = np.zeros((data_raw.shape[-3:]), dtype=complex)
            slices = data_raw.shape[1]

            # added newaxis, to have a 3D array altough it just contains 2D entries (3rd dim is length=1)
            # for every slice a new 3D array is build with the 0 axis with a length of 1
            for index in range(0, slices):
                
                dataset_denoised[index,:,:] = self.tgv2_reconstruction(data_raw[:,index:index+1,:,:], data_coils[:,index:index+1,:,:], *params)[0,:,:]

            return dataset_denoised


        elif dataset_type == 3:
            # keep dataset just as it is, if its allready 3D -> dataset_noisy = dataset_noisy

            dataset_denoised = self.tgv2_reconstruction(data_raw, data_coils, *params)

            return dataset_denoised


        else:
            raise TypeError("Dataset must be either 2D or 3D and matching the correct dataset type")
        

    def tgv2_reconstruction(self, img_kspace, sens_coils, alpha0, alpha1, iterations):
        """
        @param f: the K observations of shape MxNxK
        @param alpha: tuple containing alpha1 and alpha2
        @param maxit: maximum number of iterations
        @return: tuple of u with shape MxN and v with shape 2xMxN
        """

        # Parameters
        gamma = 10
        beta = 1
        theta = 1
        mu = 0.5

        # d is the variable which contains all the k-space data for the sample for all coils
        # and has dimension Nc*Nz*Ny*Nx
        d = img_kspace

        # C is number of coils, L,M,N are the length of the 3D dimensions
        C, L, M, N = d.shape
        

        # make operators
        k = self.make_K(L,M,N)

        
        # Lipschitz constant of K
        Lip = np.sqrt(12)

        # initialize primal variables - numpy arrays shape (L*M*N, )
        u = np.zeros(L*M*N, dtype=complex)
        v = np.zeros(3*L*M*N, dtype=complex)

        #@TODO change p,q to z1 z2
        # initialize dual variables
        p = np.zeros(3*L*M*N, dtype=complex)
        q = np.zeros(9*L*M*N, dtype=complex)
        r = np.zeros(C*L*M*N, dtype=complex)


        # primal and dual step size
        tau = 1 / Lip
        sigma = 1 / Lip

        # array of array (still shape (xxxx, ))
        u_vec = np.concatenate([u, v])
        p_vec = np.concatenate([p, q])

        # list (not array!) which contains n arrays, each with length (xxxx, )
        

        # original picture of sample reconstructed from initial mri sample data
        # fourier inverse, coil correction and sum (or sum of squares?) over all coil pictures
        uk = np.ravel(self.DAH(d, sens_coils))

        #@TODO das ist nur ein test, um zu sehen wie es grundsätzlich mit der ifft aussieht und uk (summenbild von d reco)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(abs(uk.reshape(M,N)), cmap='gray')
        # plt.subplot(1,2,2)
        # plt.imshow(abs(np.fft.ifft2(d[9,0,:,:])), cmap='gray')
        # plt.show()


        # temp vector for DAH*r 
        DAHr = np.zeros_like(u_vec, dtype=complex)

        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            
            #add result of DAHr only to first L*M*N entries, because they belong to the u_vec , v_vec should not be influenced
            DAHr[0:L*M*N] = np.ravel(self.DAH(r.reshape(C,L,M,N), sens_coils))
            
            x = u_vec - tau * (k.T @ p_vec + DAHr)
            u = self.prox_G(x[0:L*M*N], uk, tau, gamma)
            v = x[L*M*N:12*L*M*N]
            u_vec_old = u_vec
            u_vec = np.concatenate([u, v])

            u_bar = 2*u_vec - u_vec_old

            p_temp = p_vec + sigma*k@(u_bar)
            p = np.ravel(self.proj_ball(p_temp[0:3*L*M*N].reshape(3, L*M*N), alpha0))
            q = np.ravel(self.proj_ball(p_temp[3*L*M*N:12*L*M*N].reshape(9, L*M*N), alpha1))
            p_vec = np.concatenate([p, q])
            

            r_temp = r + np.ravel(sigma*(self.DA(u_bar[0:L*M*N].reshape(L,M,N), sens_coils)-d))
            r = np.ravel(self.proj_L2(r_temp, sigma, np.ravel(d)))


        u = u.reshape(L,M,N)
           
        return u


