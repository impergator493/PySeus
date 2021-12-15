# Assignment from Image Processing, taken from there

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp

# @TODO: womöglich als methode in TV klasse inkludieren?
class TGV(): 


    def __init__(self):
        pass


    def _make_nabla(self,M, N):
        row = np.arange(0, M * N)
        dat = np.ones(M * N)
        col = np.arange(0, M * N).reshape(M, N)
        col_xp = np.hstack([col[:, 1:], col[:, -1:]])
        col_yp = np.vstack([col[1:, :], col[-1:, :]])

        # das sind darstellungen von gradienten operatoren in vektorformen, wsl um schneller berechnen zu können
        # col_xp hat die um 1 verschobenen einträge nach links, col die originalen. diese werden voneinander abgezogen
        # um die differenz bilden zu können.
        nabla_x = scipy.sparse.coo_matrix((dat, (row, col_xp.flatten())), shape=(M * N, M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M * N, M * N))

        nabla_y = scipy.sparse.coo_matrix((dat, (row, col_yp.flatten())), shape=(M * N, M * N)) - \
                scipy.sparse.coo_matrix((dat, (row, col.flatten())), shape=(M * N, M * N))

        nabla = scipy.sparse.vstack([nabla_x, nabla_y])

        return nabla, nabla_x, nabla_y


    def compute_Wi(self, W, i):
        """
        Used for calculation of the dataterm projection

        can be used for confidences or set to zero if datapoint is not available
        @param W:
        @param i: index of the observation
        @return:
        """

        #if there is just one K ist just makes the sum over this K, which is in fact
        # the same as if there would be just a 2D array full of ones with (M,N,)
        Wi = -np.sum(W[:, :, :i], axis=-1) + np.sum(W[:, :, i:], axis=-1)
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
        # if K is 1, no new axis is needed and also
        # Wis is just a vector full of ones with length(M*N,K) = (M*N,1)
        # hier kann also einfach tau mit ones(M*N) multipliziert werden
        # und np.newaxis gelöscht werden
        pis = u[..., np.newaxis] + tau * Wis

        # f, pis haben alle bei K=1 shape(M*N,1)
        # var hat shape(M*N,2)
        var = np.concatenate((f, pis), axis=-1)

        # bei nur 2 einträgen (f, pis) macht median dasselbe wie mean und nimmt den mittelwert
        # egal wieviele einträge, über eine achse nimmt er nur den median und diese
        # dimension verschwindet dann sogar, d.h. aus shape (M*N,2) wird dann (M*N,)
        # Rückgabe hat also K dimension wieder weniger
        prox = np.median(var, axis=-1)

        return prox


    def make_K(self, M, N):
        """
        @param M:
        @param N:
        @return: the K operator as described in Equation (5)
        """
        nabla, nabla_x, nabla_y = self._make_nabla(M, N)
        neg_I = sp.identity(M*N) * -1

        K = sp.bmat([[nabla_x, neg_I, None], [nabla_y, None, neg_I], [None, nabla_x, None], [None, nabla_y, None], [None, None, nabla_x], [None, None, nabla_y]])

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


    def tgv2_denoising(self,img_noisy, alpha0, alpha1, iterations):
        """
        @param f: the K observations of shape MxNxK
        @param alpha: tuple containing alpha1 and alpha2
        @param maxit: maximum number of iterations
        @return: tuple of u with shape MxN and v with shape 2xMxN
        """
        # K = 1 just for fast experiment, bc it was used for TGV fusion before.
        f = np.zeros(img_noisy.shape + (1,))
        f[:,:,0] = img_noisy

        M, N, K = f.shape
        img = img_noisy.reshape(M*N, K)

        # make operators
        k = self.make_K(M,N)

        # Used for calculation of the dataterm projection
        # if K is just 1, there is no need for computation in 
        # function, just keep W without K
        W = np.ones((M, N, K))
        Wis = np.asarray([self.compute_Wi(W, i) for i in range(K)])
        Wis = Wis.transpose(1, 2, 0)
        Wis = Wis.reshape(M * N, K)

        # Lipschitz constant of K
        L = np.sqrt(12)

        # initialize primal variables
        u = np.zeros(M*N)
        v = np.zeros(2*M*N)

        # initialize dual variables
        p = np.zeros(2*M*N)
        q = np.zeros(4*M*N)

        # primal and dual step size
        tau = 1 / L
        sigma = 1 / L

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
            u_list.append(self.prox_sum_l1(x[0:M*N], img, tau, Wis))
            v_list.append(x[M*N:3*M*N])
            u_veclist.append(np.concatenate([u_list[it], v_list[it]]))

            p_temp = p_veclist[it] + sigma*k@(2*u_veclist[it+1] - u_veclist[it])
            p_list.append(np.ravel(self.proj_ball(p_temp[0:2*M*N].reshape(2, M*N), alpha0)))
            q_list.append(np.ravel(self.proj_ball(p_temp[2*M*N:6*M*N].reshape(4, M*N), alpha1)))
            p_veclist.append(np.concatenate([p_list[it], q_list[it]]))

            #energy_list.append(energy(u_list[it], v_list[it], alpha1, alpha2, img, nabla, nabla_tilde, M, N))

        u = u_list[iterations-1].reshape(M,N)
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



