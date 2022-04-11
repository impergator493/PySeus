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
        #self.lip_inv = np.sqrt(1/64)
        self.lip_inv = 100

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

    def make_K(self, L, M, N):
        """
        @param M:
        @param N:
        @return: the K operator as described in Equation (5)
        """
        nabla, nabla_x, nabla_y, nabla_z = self._make_nabla(L, M, N)

        K = sp.bmat([[nabla_x], [nabla_y], [nabla_z]])


        return K


    def proj_ball(self, Y):
        """
        Projection to a ball as described in Equation (6)
        @param Y: either 2xMN or 4xMN
        @param alpha: scalar hyperparameter alpha
        @return: projection result either 2xMN or 4xMN
        """
        norm = np.linalg.norm(Y, axis=0)
        projection = Y / np.maximum(1, norm)
    
        return projection

    def prox_F(self, r, sigma, lambd):
        
        return (r*lambd)/(lambd+ sigma) # this is from knoll stollberger tgv paper
 
    #@TODO temporarily absolute value of coils sensitivities just for trying
    def op_A(self, u, sens_c, sparse_mask):
        """
        input parameter:

        u - current reconstructed sample in spatial domain, size (L,M,N)
        sens_c - coil sensitivities 
        """
        # if coilmask has just ones, sum of squares is used instead of PFCu as there are no
        # coil sensitivites then

        return sparse_mask * np.fft.fftn((sens_c * u), axes=self.fft_dim, norm='ortho')

         
    #@TODO temporaril absolute value of coils sensitivities just for trying
    def op_A_conj(self, r, sens_c, sparse_mask):

        """
        input parameter:

        R - dual variable of difference of current reconstructed sample u(n) in fourier domain and initial fourier data
        """
        
        r_IFT = sens_c.conjugate() * np.fft.ifftn(r*sparse_mask,axes=self.fft_dim, norm='ortho')

        
        return np.sum( r_IFT, axis=0)


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

        # Check if Coils are adjoint
        sumsqrC = np.sum(sens_coils*np.conj(sens_coils),axis=0)
        sens_coils /= np.sqrt(sumsqrC)


        # Parameters
        beta = 1
        theta = 1
        mu = 0.5
        delta = 0.5

        # d is the variable which contains all the k-space data for the sample for all coils
        # and has dimension Nc*Nz*Ny*Nx
        d = img_kspace/np.linalg.norm(img_kspace)

    

        # C is number of coils, L,M,N are the length of the 3D dimensions
        C, L, M, N = d.shape
        

        # make operators
        k = self.make_K(L,M,N)
     
       
        # this is basically u_old, just defined here for better readability
        u_old = np.zeros(L*M*N, dtype=np.complex128)

        #@TODO change p,q to z1 z2
        # initialize dual variables
        p_old = np.zeros(3*L*M*N, dtype=np.complex128)
        r_old = np.zeros(C*L*M*N, dtype=np.complex128)


        # primal and dual step size
        tau_old = self.lip_inv
        sigma = self.lip_inv

        y_old = np.zeros((3+C)*L*M*N, dtype=np.complex128)
        kTy_old = np.zeros(u_old, dtype=np.complex128)


        # temp vector for A* * r
        Aconj_r = np.zeros_like(u_old, dtype=np.complex128)

        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            
            #add result of DAHr only to first L*M*N entries, because they belong to the u_vec , v_vec should not be influenced
            Aconj_r = np.ravel(self.op_A_conj(r_old.reshape(C,L,M,N), sens_coils, sparse_mask))
            
            # prox for u not necessary
            u_new = u_old - tau_old * (k.T @ p_old + Aconj_r)
            # v = u_vec[L*M*N:12*L*M*N]
            # u_vec = np.concatenate([u, v])

            tau_new = tau_old*(1+theta)**0.5
            print("new tau")
            print("Tau_n:", tau_new)



            while True:
                #import ipdb
                #ipdb.set_trace()
                theta = tau_new/tau_old
                sigma = beta * tau_new
                u_bar = u_new + theta * (u_new - u_old)

                p_temp = p_old + sigma*k@(u_bar)
                p_new = np.ravel(self.proj_ball(p_temp.reshape(3, L*M*N)))
                r_temp = r_old + np.ravel(sigma*(self.op_A(u_bar.reshape(L,M,N), sens_coils, sparse_mask)-d))
                r_new = np.ravel(self.prox_F(r_temp, sigma, lambd))

                Aconj_r = np.ravel(self.op_A_conj(r_new.reshape(C,L,M,N), sens_coils, sparse_mask))
                kTy_new = k.T @ p_new + Aconj_r
                y_new = np.concatenate([p_new, r_new])

                print("calculate norm")
                LS = np.sqrt(beta)*tau_new*(np.linalg.norm(kTy_new - kTy_old))
                RS = delta*(np.linalg.norm(y_new - y_old))
                print("LS is:", LS)
                print("RS is:", RS)
                if  LS <= RS:
                    print("Break Linesearch")
                    break
                else: tau_new = tau_new * mu
                print("reduce tau")
                print("Tau_n:", tau_new)

            # update variables
            u_old = u_new
            p_old = p_new
            r_old = r_new
            y_old = y_new
            kTy_old = kTy_new
            tau_old = tau_new

        u_new = u_new.reshape(L,M,N)
           
        return u_new


