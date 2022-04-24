# Assignment from Image Processing, taken from there

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp

from ..settings import ProcessSelDataType

# @TODO: womöglich als methode in TV klasse inkludieren?
class TGV_Denoise(): 


    def __init__(self):
        
        self.h_inv = 1.0
        self.hz_inv = 1.0

         # Lipschitz constant of K, according to knoll paper TGV reco and denoising, with iso spacing for x and y
        #self.lip_inv = np.sqrt((2*(1/self.h_inv)**2)/(16+(1/self.h_inv)**2+np.sqrt(32*(1/self.h_inv)**2+(1/self.h_inv)**4)))
        self.lip_inv = 10

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

    def prox_G(self, u, f, tau, lambd):
        """
        Used for calculation of the dataterm projection

        compute pi with pi = \bar x + tau * W_i
        @param u: MN
        @param tau: scalar
        @param Wis: MN x K
        @param f: MN x K
        """
      
        # Das ist prox operator für L1 norm von datenterm, nicht für L2, deswegen haben wir
        # sonst immer andere Formel gehabt die weiter unten steht.
        #pis = u[...] + tau

        # bei nur 2 einträgen (f, pis) macht median dasselbe wie mean und nimmt den mittelwert
        # egal wieviele einträge, über eine achse nimmt er nur den median und diese
        # dimension verschwindet dann sogar, d.h. aus shape (M*N,2) wird dann (M*N,)
        # Rückgabe hat also K dimension wieder weniger
        #prox = np.median((f,pis), axis=0)

        # Derweil auf das umgestellt, weils laut Papers so richtig ist.
        prox = (f*tau*lambd + u)/(1 + tau*lambd)

        return prox

    def proj_ball(self, Y, alpha):
        """
        Projection to a ball as described in Equation (6)
        @param Y: either 2xMN or 4xMN
        @param lamb: scalar hyperparameter lambda
        @return: projection result either 2xMN or 4xMN
        """
        norm = np.linalg.norm(Y, axis=0)
        projection = Y / np.maximum(1, norm/alpha)

        return projection


    def tgv2_denoising_gen(self, dataset_type, dataset_noisy, params, spac):


        self.h_inv = spac[0]
        self.hz_inv = spac[1]

        # prepare artifical 3D dataset(1,M,N) for 2D image (M,N), because to be universal applicable at least one entry
        # in 3rd dimension is expected  
        if dataset_type == ProcessSelDataType.SLICE_2D:
            temp = np.zeros((1,*dataset_noisy.shape))
            temp[0,:,:] = dataset_noisy
            dataset_noisy = temp

            # make return value 2D array again
            dataset_denoised = self.tgv2_denoising(dataset_noisy, *params)[0,:,:]

            return dataset_denoised

        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_2D:
            
            # set z gradient spaces to 1/0 = infinity, so that no gradient is z direction is calculated
            # therefor every 2D picture is denoised individually
            
            dataset_denoised = self.tgv2_denoising(dataset_noisy, *params)

            return dataset_denoised


        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
            # keep dataset just as it is, if its allready 3D -> dataset_noisy = dataset_noisy

            # set inverted z spacing to any number different then 0 for a 3D denoising

            dataset_denoised = self.tgv2_denoising(dataset_noisy, *params)

            return dataset_denoised

        else:
            raise TypeError("Dataset must be either 2D or 3D and matching the correct dataset type")



    def tgv2_denoising(self, img_noisy, lambd, alpha0, alpha1, iterations):
        """
        @param f: the K observations of shape MxNxK
        @param alpha: tuple containing alpha1 and alpha2
        @param maxit: maximum number of iterations
        @return: tuple of u with shape MxN and v with shape 2xMxN
        """
        # star argument to take really the value of the variable as argument
        # if 2dim noisy data make it a 3D array, if 3D just let it be
        
        # Parameters
        beta = 1
        theta = 1
        mu = 0.5
        delta = 0.99

        # inverted spacing is used so that h* = 0 is an infinite spacing
        f = img_noisy

        L, M, N = f.shape
        img = img_noisy.reshape(L*M*N)

        # make operators
        k = self.make_K(L,M,N)

        # initialize primal variables
        u_old = np.zeros(L*M*N)
        v_old = np.zeros(3*L*M*N)

        # initialize dual variables
        p_old = np.zeros(3*L*M*N)
        q_old = np.zeros(9*L*M*N)

        # primal and dual step size
        tau_old = self.lip_inv
        sigma = self.lip_inv

        x_vec_old = np.concatenate([u_old, v_old])
        y_vec_old = np.concatenate([p_old, q_old])

        
        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            print("iterations: " + str(it))
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            x_vec_new = x_vec_old - tau_old * k.T @ y_vec_old
            u_new = self.prox_G(x_vec_new[0:L*M*N], img, tau_old, lambd)
            v_new = x_vec_new[L*M*N:12*L*M*N]
            x_vec_new = np.concatenate([u_new, v_new])

            tau_new = tau_old*(1+theta)**0.5
            #print("new tau")
            #print("Tau_n:", tau_new)

            while True:
                theta = tau_new/tau_old
                sigma = beta * tau_new
                x_bar = x_vec_new + theta * (x_vec_new - x_vec_old)

                y_temp = y_vec_old + sigma*k@(x_bar)
                p_new = np.ravel(self.proj_ball(y_temp[0:3*L*M*N].reshape(3, L*M*N), alpha1))
                q_new = np.ravel(self.proj_ball(y_temp[3*L*M*N:12*L*M*N].reshape(9, L*M*N), alpha0))
                y_vec_new = np.concatenate([p_new, q_new])


                #print("Denoise TGV: calculate norm")
                LS = np.sqrt(beta)*tau_new*(np.linalg.norm(k.T@y_vec_new - k.T@y_vec_old))
                RS = delta*(np.linalg.norm(y_vec_new - y_vec_old))
                #print("LS is:", LS, "and of type:", type(LS))
                #print("RS is:", RS, "and of type:", type(RS))
                if  LS <= RS:
                    #print("Break Linesearch")
                    break
                else: tau_new = tau_new * mu
                #print("reduce tau")
                #print("Tau_n:", tau_new)

            # update variables
            x_vec_old = x_vec_new
            y_vec_old = y_vec_new
            tau_old = tau_new

        u_new = u_new.reshape(L,M,N)
        return u_new


