# Assignment from Image Processing, taken from there

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp

# @TODO: womöglich als methode in TV klasse inkludieren?
class TGV_3D(): 


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

        nabla_x = scipy.sparse.coo_matrix((dat, (row, col_xp.flatten())), shape=(L * M * N, L * M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N))

        nabla_y = scipy.sparse.coo_matrix((dat, (row, col_yp.flatten())), shape=(L * M * N, L * M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N))
        
        nabla_z = scipy.sparse.coo_matrix((dat, (row, col_zp.flatten())), shape=(L * M * N, L * M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(L * M * N, L * M * N))



        nabla = scipy.sparse.vstack([nabla_x, nabla_y, nabla_z])

        return nabla, nabla_x, nabla_y, nabla_z


    def compute_Wi(self, W, i):
        """
        Used for calculation of the dataterm projection

        can be used for confidences or set to zero if datapoint is not available
        @param W:
        @param i: index of the observation
        @return:
        """
        Wi = -np.sum(W[:,:, :, :i], axis=-1) + np.sum(W[:,:, :, i:], axis=-1)
        return Wi


    def prox_sum_l1(self, u, f, tau, Wis):
        """
        Used for calculation of the dataterm projection

        compute pi with pi = \bar x + tau * W_i
        @param u: MN
        @param tau: scalar
        @param Wis: MN x K
        @param f: MN x K
        """
        pis = u[..., np.newaxis] + tau * Wis

        var = np.concatenate((f, pis), axis=-1)

        prox = np.median(var, axis=-1)

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


    def proj_ball(self, Y, lamb):
        """
        Projection to a ball as described in Equation (6)
        @param Y: either 2xMN or 4xMN
        @param lamb: scalar hyperparameter lambda
        @return: projection result either 2xMN or 4xMN
        """
        norm = np.linalg.norm(Y, axis=0)
        projection = Y / np.maximum(1, 1/lamb * norm)
    
        return projection



    # def energy(u, v, alpha1, alpha2, f, nabla, nabla_tilde, M, N):
    #     matrix1 = nabla @ u - v
    #     matrix2 = nabla_tilde @ v

    #     energy1 = alpha1 * np.sum(np.linalg.norm(matrix1.reshape(2, M*N), axis=0))
    #     energy2 =  alpha2 * np.sum(np.linalg.norm(matrix2.reshape(4, M*N), axis=0))
    #     energy3 = np.sum(np.linalg.norm(u[:, np.newaxis]-f, axis = 0))
    #     energy_value =  energy1 + energy2 + energy3

    #     return energy_value


    def tgv2_3D_denoising(self,img_noisy, alpha0, alpha1, iterations):
        """
        @param f: the K observations of shape MxNxK
        @param alpha: tuple containing alpha1 and alpha2
        @param maxit: maximum number of iterations
        @return: tuple of u with shape MxN and v with shape 2xMxN
        """
        # K = 1 just for fast experiment, bc it was used for TGV fusion before.
        # Could it be used for TGV Reco, as this is also somehow the same with several pcitures from one and 
        # the same thing?
        f = np.zeros(img_noisy.shape + (1,))
        f[:,:,:,0] = img_noisy

        L, M, N, K = f.shape
        img = img_noisy.reshape(L*M*N, K)

        # make operators
        k = self.make_K(L,M,N)

        # Used for calculation of the dataterm projection
        W = np.ones((L, M, N, K))
        Wis = np.asarray([self.compute_Wi(W, i) for i in range(K)])
        Wis = Wis.transpose(1, 2, 3, 0)
        Wis = Wis.reshape(L * M * N, K)

        # Lipschitz constant of K
        Lip = np.sqrt(12)

        # initialize primal variables
        u = np.zeros(L*M*N)
        v = np.zeros(3*L*M*N)

        # initialize dual variables
        p = np.zeros(3*L*M*N)
        q = np.zeros(9*L*M*N)

        # primal and dual step size
        tau = 1 / Lip
        sigma = 1 / Lip

        u_vec = np.concatenate([u, v])
        p_vec = np.concatenate([p, q])

        u_veclist = [u_vec]
        p_veclist = [p_vec]
        u_list = []
        v_list = []
        p_list = []
        q_list = []
        #energy_list = []

        # Parameters for the energy function
        # nabla, _, _ = _make_nabla(M, N)
        # nabla_tilde = sp.bmat([[nabla, None], [None, nabla]])

        for it in range(0, iterations):
            # TODO calculate iterates as described in Equation (4)
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            x = u_veclist[it] - tau * k.T @ p_veclist[it]
            u_list.append(self.prox_sum_l1(x[0:L*M*N], img, tau, Wis))
            v_list.append(x[L*M*N:12*L*M*N])
            u_veclist.append(np.concatenate([u_list[it], v_list[it]]))

            p_temp = p_veclist[it] + sigma*k@(2*u_veclist[it+1] - u_veclist[it])
            p_list.append(np.ravel(self.proj_ball(p_temp[0:3*L*M*N].reshape(3, L*M*N), alpha0)))
            q_list.append(np.ravel(self.proj_ball(p_temp[3*L*M*N:12*L*M*N].reshape(9, L*M*N), alpha1)))
            p_veclist.append(np.concatenate([p_list[it], q_list[it]]))

            #energy_list.append(energy(u_list[it], v_list[it], alpha1, alpha2, img, nabla, nabla_tilde, M, N))

        u = u_list[iterations-1].reshape(L,M,N)
        img_denoised = u
        #v = v_list[iterations-1].reshape(2, M, N)

        return img_denoised

    # unneccessary, directly take tgv2_pd method from above
    #def tgv2_denoising(self,img, alpha, iterations):
        # Load Observations
        # samples = np.array([np.load('data/observation{}.npy'.format(i)) for i in range(0,9)])
        # f = samples.transpose(1,2,0)


        # Perform TGV-Fusion
        #alpha_list = [(1.0, 0.01), (1.0, 0.1), (1.0, 1.0),  (1.0, 10.0), (1,30), (0.5,0.1), (0.5,0.5), (0.5,1), (0.01,0.1), (0.1, 0.1), (0.1, 1), (0.1, 10), (0.1, 30), (10, 0.01), (10, 0.1), (10, 1), (10, 10), (10,30)]

        #for alpha in alpha_list:
            #res, v, energyval = tgv2_pd(img, alpha=alpha, iterations)  # TODO: set appropriate parameters

            # Calculate accuracy, könnte man mit orginial vergleichen ohne denoised, später
            #print(alpha[0], alpha[1], compute_accX(res, gt))



