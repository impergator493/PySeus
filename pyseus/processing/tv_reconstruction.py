# Assignment from Image Processing, taken from there
# plus changes to TGV-L2 according Knoll stollberger paper
# main algorithmen is from homework, not from knoll paper (problem with u = u -tau*(), in paper there is u + tau*())

import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as la

from ..settings import ProcessSelDataType

# @TODO: womöglich als methode in TV klasse inkludieren?
class TV_Reco(): 


    def __init__(self):
        
        # inverted spacing is used so that h* = 0 is an infinite spacing
        self.h_inv = 1.0
        self.hz_inv = 1.0

         # Lipschitz constant of K, according to papers ok, aber beim probieren ist es eigentlich zu klein.
        #self.lip_inv = np.sqrt((2*(1/self.h_inv)**2)/(16+(1/self.h_inv)**2+np.sqrt(32*(1/self.h_inv)**2+(1/self.h_inv)**4)))
        self.lip_inv = np.sqrt(1/64)

        # dimension for which the fft and ifft should be calculated, standard is 2D
        self.fft_dim = (-2,-1)


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

        nabla_x = (scipy.sparse.coo_matrix((dat, (row, col_xp.flatten())), shape=(L * M * N, L * M * N)) -
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N)))*self.h_inv

        nabla_y = (scipy.sparse.coo_matrix((dat, (row, col_yp.flatten())), shape=(L * M * N, L * M * N)) -
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N)))*self.h_inv
        
        nabla_z = (scipy.sparse.coo_matrix((dat, (row, col_zp.flatten())), shape=(L * M * N, L * M * N)) -
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N)))*self.hz_inv


        nabla = scipy.sparse.vstack([nabla_x, nabla_y, nabla_z])

        return nabla, nabla_x, nabla_y, nabla_z

    # not used anymore bc r has the prox op already
    # def prox_U(self, x, uk, tau, gamma):
    #     """
    #     Used for calculation of the dataterm projection

    #     @param uk: initial reconstructed image from raw data

    #     compute pi with pi = \bar x + tau * W_i
    #     @param u: MN
    #     @param tau: scalar
    #     @param Wis: MN x K
    #     @param f: MN x K
    #     """
      
    #     prox = (uk*tau/gamma + x)/(1 + tau/gamma)

    #     # bei nur 2 einträgen (f, pis) macht median dasselbe wie mean und nimmt den mittelwert
    #     # egal wieviele einträge, über eine achse nimmt er nur den median und diese
    #     # dimension verschwindet dann sogar, d.h. aus shape (M*N,2) wird dann (M*N,)
    #     # Rückgabe hat also K dimension wieder weniger
    #     # pis = u[...] + tau
    #     # prox = np.median((uk,pis), axis=0)

    #     return prox


    def make_K(self, L, M, N):
        """
        @param M:
        @param N:
        @return: the K operator as described in Equation (5)
        """
        nabla, nabla_x, nabla_y, nabla_z = self._make_nabla(L, M, N)

        K = sp.bmat([[nabla_x], [nabla_y], [nabla_z]])


        return K


    def proj_ball(self, Z):
        """
        Projection to a ball as described in Equation (6)
        @param Y: either 2xMN or 4xMN
        @param alpha: scalar hyperparameter alpha
        @return: projection result either 2xMN or 4xMN
        """
        norm = np.linalg.norm(Z, axis=0)
        projection = Z / np.maximum(1, norm)
    
        return projection


 
    #@TODO temporarily absolute value of coils sensitivities just for trying
    def DA(self, u, sens_c, sparse_mask):
        """
        input parameter:

        u - current reconstructed sample in spatial domain, size (L,M,N)
        sens_c - coil sensitivities 
        """
        return sparse_mask * np.fft.fftn((sens_c * u), axes=self.fft_dim)
        

        
    #@TODO temporaril absolute value of coils sensitivities just for trying
    def DAH(self, r, sens_c, sparse_mask):

        """
        input parameter:

        R - dual variable of difference of current reconstructed sample u(n) in fourier domain and initial fourier data
        """
        
        r_IFT = sens_c.conjugate() * np.fft.ifftn(r*sparse_mask,axes=self.fft_dim)

        
        return np.sum( r_IFT, axis=0)


    def prox_R(self, R, sigma, lambd):
        
        
        return (R*lambd)/(lambd+ sigma) # this is from knoll stollberger tgv paper

   # if its a big dataset, a lot of RAM is needed because all the raw data to process will be 
    # stored in the RAM
    def tv_reconstruction_gen(self, func_reco, dataset_type, data_raw, data_coils, sparse_mask, params, spac):

        self.h_inv = spac[0]
        self.hz_inv = spac[1]

        if dataset_type == ProcessSelDataType.SLICE_2D:
            # Because of Coil data, correct slice has to be select with L=1 already when method is called
            # dat_real, imag are of dimension (C*L*M*N), if 2D with 3.Dim L is just length 1
            # make return value 2D array again
            dataset_denoised = func_reco(data_raw, data_coils, sparse_mask, *params)[0,:,:]

            return dataset_denoised

        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_2D:
            
            self.fft_dim = (-2,-1)
              
            dataset_denoised = func_reco(data_raw, data_coils, sparse_mask, *params)

            return dataset_denoised

        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
            
            self.fft_dim = (-3,-2,-1)
            
            dataset_denoised = func_reco(data_raw, data_coils, sparse_mask, *params)

            return dataset_denoised

        else:
            raise TypeError("Dataset must be either 2D or 3D and matching the correct dataset type")
    

    def tv_l2_reconstruction(self, img_kspace, sens_coils, sparse_mask, lambd, iterations):
        """
        @param f: the K observations of shape MxNxK
        @param alpha: tuple containing alpha1 and alpha2
        @param maxit: maximum number of iterations
        @return: tuple of u with shape MxN and v with shape 2xMxN
        """


        # Parameters
        beta = 1
        theta = 1
        mu = 0.5
        delta = 1

        # d is the variable which contains all the k-space data for the sample for all coils
        # and has dimension Nc*Nz*Ny*Nx
        d = img_kspace

        # C is number of coils, L,M,N are the length of the 3D dimensions
        C, L, M, N = d.shape
        

        # make operators
        k = self.make_K(L,M,N)
     
       
        # initialize primal variables - numpy arrays shape (L*M*N, )
        u = np.zeros(L*M*N, dtype=complex)

        #@TODO change p,q to z1 z2
        # initialize dual variables
        p = np.zeros(3*L*M*N, dtype=complex)
        r = np.zeros(C*L*M*N, dtype=complex)


        # primal and dual step size
        tau = self.lip_inv
        sigma = self.lip_inv

        tau_n = tau
        tau_n_old = tau

        # array of array (still shape (xxxx, ))

        # list (not array!) which contains n arrays, each with length (xxxx, )
        

        # temp vector for DAH*r 
        DAHr = np.zeros_like(u, dtype=complex)

        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            
            #add result of DAHr only to first L*M*N entries, because they belong to the u_vec , v_vec should not be influenced
            DAHr[0:L*M*N] = np.ravel(self.DAH(r.reshape(C,L,M,N), sens_coils, sparse_mask))
            
            # prox for u not necessary
            u_old = u
            u = u - tau_n_old * (k.T @ p + DAHr)
            # v = u_vec[L*M*N:12*L*M*N]
            # u_vec = np.concatenate([u, v])

            tau_n = tau_n_old*(1+theta)**0.5
            
            while True:
                theta = tau_n/tau_n_old
                sigma = beta * tau_n

                u_bar = u + theta * (u - u_old)
                
                y_old = np.concatenate([p, r])
                DAHr[0:L*M*N] = np.ravel(self.DAH(r.reshape(C,L,M,N), sens_coils, sparse_mask))
                ky_old = k.T@p + DAHr

                p_temp = p + sigma*k@(u_bar)
                p = np.ravel(self.proj_ball(p_temp[0:3*L*M*N].reshape(3, L*M*N)))
                r_temp = r + np.ravel(sigma*(self.DA(u_bar[0:L*M*N].reshape(L,M,N), sens_coils, sparse_mask)-d))
                r = np.ravel(self.prox_R(r_temp, sigma, lambd))

                y_new = np.concatenate([p, r])
                DAHr[0:L*M*N] = np.ravel(self.DAH(r.reshape(C,L,M,N), sens_coils, sparse_mask))
                ky_new = k.T@p + DAHr

                if (np.sqrt(beta)*tau_n*(np.linalg.norm(ky_new - ky_old))) <= (delta*(np.linalg.norm(y_new - y_old))):
                    break
                else: tau_n = tau_n * mu


        u = u.reshape(L,M,N)
           
        return u


