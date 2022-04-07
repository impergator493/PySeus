# Assignment from Image Processing, taken from there
# plus changes to TGV-L2 according Knoll stollberger paper
# main algorithmen is from homework, not from knoll paper (problem with u = u -tau*(), in paper there is u + tau*())

import numpy as np
import scipy
import scipy.sparse as sp

from ..settings import ProcessSelDataType

# @TODO: womöglich als methode in TV klasse inkludieren?
class TGV_Reco(): 


    def __init__(self):
        
        # inverted spacing is used so that h* = 0 is an infinite spacing
        self.h_inv = 1.0
        self.hz_inv = 1.0

         # Lipschitz constant of K, according to papers ok, aber beim probieren ist es eigentlich zu klein.
        #self.lip_inv = np.sqrt((2*(1/self.h_inv)**2)/(16+(1/self.h_inv)**2+np.sqrt(32*(1/self.h_inv)**2+(1/self.h_inv)**4)))
        self.lip_inv = 10
        #self.lip_inv = np.sqrt(1/64)
        #self.lip_inv = np.sqrt(1/12)

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
        neg_I = sp.identity(L*M*N) * -1

        K = sp.bmat([[nabla_x, neg_I, None, None], [nabla_y, None, neg_I, None], [nabla_z, None, None, neg_I], \
            [None, nabla_x, None, None], [None, nabla_y, None, None], [None, nabla_z, None, None], \
            [None, None, nabla_x, None], [None, None, nabla_y, None], [None, None, nabla_z, None],
            [None, None, None, nabla_x], [None, None, None, nabla_y], [None, None, None, nabla_z]])

        return K

    def proj_ball(self, Y, alpha):
        """
        Projection to a ball as described in Equation (6)
        @param Y: either 2xMN or 4xMN
        @param alpha: scalar hyperparameter alpha
        @return: projection result either 2xMN or 4xMN
        """
        norm = np.linalg.norm(Y, axis=0)
        projection = Y / np.maximum(alpha, norm)
    
        return projection

    def prox_F(self, r, sigma, lambd):
        
        return (r*lambd)/(lambd + sigma) # this is from knoll stollberger tgv paper

 
    #@TODO temporarily absolute value of coils sensitivities just for trying
    def op_A(self, u, sens_c, sparse_mask):
        """
        input parameter:

        u - current reconstructed sample in spatial domain, size (L,M,N)
        sens_c - coil sensitivities 
        """
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
    def tgv2_reconstruction_gen(self, dataset_type, data_raw, data_coils, sparse_mask, params, spac):

        self.h_inv = spac[0]
        self.hz_inv = spac[1]

        if dataset_type == ProcessSelDataType.SLICE_2D:
            # Because of Coil data, correct slice has to be select with L=1 already when method is called
            # dat_real, imag are of dimension (C*L*M*N), if 2D with 3.Dim L is just length 1
            # make return value 2D array again
            dataset_denoised = self.tgv2_reconstruction(data_raw, data_coils, sparse_mask, *params)[0,:,:]

            return dataset_denoised

        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_2D:
            
            self.fft_dim = (-2,-1)
              
            dataset_denoised = self.tgv2_reconstruction(data_raw, data_coils, sparse_mask, *params)

            return dataset_denoised

        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
            
            self.fft_dim = (-3,-2,-1)
            
            dataset_denoised = self.tgv2_reconstruction(data_raw, data_coils, sparse_mask, *params)

            return dataset_denoised

        else:
            raise TypeError("Dataset must be either 2D or 3D and matching the correct dataset type")
        

    def tgv2_reconstruction(self, img_kspace, sens_coils, sparse_mask, lambd, alpha0, alpha1, iterations):
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
        delta = 0.98

        # d is the variable which contains all the k-space data for the sample for all coils
        # and has dimension Nc*Nz*Ny*Nx
        d = img_kspace

        # C is number of coils, L,M,N are the length of the 3D dimensions
        C, L, M, N = d.shape
        
        # make operators
        k = self.make_K(L,M,N)
       
        # initialize primal variables - numpy arrays shape (L*M*N, )
        u = np.zeros(L*M*N, dtype=np.complex128)
        v = np.zeros(3*L*M*N, dtype=np.complex128)

        #@TODO change p,q to z1 z2
        # initialize dual variables
        p = np.zeros(3*L*M*N, dtype=np.complex128)
        q = np.zeros(9*L*M*N, dtype=np.complex128)
        r = np.zeros(C*L*M*N, dtype=np.complex128)


        # primal and dual step size
        tau = self.lip_inv
        sigma = self.lip_inv

        tau_old = tau
        x_old = np.concatenate([u, v])
        pq_old = np.concatenate([p, q])
        y_old = np.zeros((3+9+C)*L*M*N, dtype=np.complex128)
        kTy_old = np.zeros(L*M*N, dtype=np.complex128)

            

        # temp vector for DAH*r 
        Aconj_r = np.zeros_like(x_old, dtype=np.complex128)

        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            
            #add result of DAHr only to first L*M*N entries, because they belong to the u_vec , v_vec should not be influenced
            Aconj_r[0:L*M*N] = np.ravel(self.op_A_conj(r.reshape(C,L,M,N), sens_coils, sparse_mask))
            
            # prox for u not necessary
            x_new = x_old - tau_n * (k.T @ pq_old + Aconj_r)
            #u = x_new[0:L*M*N]

            tau_new = tau_old*(1+theta)**0.5
            print("new tau")
            print("Tau_n:", tau_new)
            
            while True:
                theta = tau_new/tau_old
                sigma = beta * tau_new
                x_bar = x_new + theta * (x_new - x_old)

                pq_temp = pq_old + sigma*k@(x_bar)
                p = np.ravel(self.proj_ball(pq_temp[0:3*L*M*N].reshape(3, L*M*N), alpha1))
                q = np.ravel(self.proj_ball(pq_temp[3*L*M*N:12*L*M*N].reshape(9, L*M*N), alpha0))
                pq_old = np.concatenate([p, q])
                r_temp = r + np.ravel(sigma*(self.op_A(x_bar[0:L*M*N].reshape(L,M,N), sens_coils, sparse_mask)-d))
                r = np.ravel(self.prox_F(r_temp, sigma, lambd))

                Aconj_r[0:L*M*N] = np.ravel(self.op_A_conj(r.reshape(C,L,M,N), sens_coils, sparse_mask))
                kTy_new = k.T@pq_old + Aconj_r
                y_new = np.concatenate([p, r])


                y_new = np.concatenate([pq_old, r])
                Aconj_r[0:L*M*N] = np.ravel(self.op_A_conj(r.reshape(C,L,M,N), sens_coils, sparse_mask))
                ky_new = k.T@pq_old + Aconj_r

                print("TGV")
                print("calculate norm")
                LS = np.sqrt(beta)*tau_n*(np.linalg.norm(kTy_new - kTy_old))
                RS = delta*(np.linalg.norm(y_new - y_old))
                print("LS is:", LS, "and of type:", type(LS))
                print("RS is:", RS, "and of type:", type(RS))
                if  LS <= RS:
                    print("Update tau!")
                    break
                else: tau_n = tau_n * mu
                print("reduce tau")
                print("Tau_n:", tau_n)

        u = u.reshape(L,M,N)
           
        return u


