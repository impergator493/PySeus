# Assignment from Image Processing, taken from there

from re import U
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp

from ..settings import ProcessSelDataType

# @TODO: womöglich als methode in TV klasse inkludieren?
class TV_Denoise(): 


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

        K = sp.bmat([[nabla_x], [nabla_y], [nabla_z]])
        
        return K

    def prox_G_L1(self, u, u_0, tau, lambd):
        
        prox = (u - tau * lambd) * (u - u_0 > tau * lambd
                ) + (u + tau * lambd) * (u - u_0 < -tau * lambd
                ) + (u_0) * (abs(u - u_0) <= tau * lambd)

        return prox

    def prox_G_L2(self, u, u_0, tau, lambd):
        
        prox = (u + tau*lambd*u_0) / (1 + tau * lambd)

        return prox


    def proj_ball(self, Y):
        """
        Projection to a ball as described in Equation (6)
        @param Y: either 2xMN or 4xMN
        @param lamb: scalar hyperparameter lambda
        @return: projection result either 2xMN or 4xMN
        """
        norm = np.linalg.norm(Y, axis=0)
        projection = Y / np.maximum(1, norm)
    
        return projection


    def tv_denoising_gen(self, func_denoise, dataset_type, dataset_noisy, params, spac):

        self.h_inv = spac[0]
        self.hz_inv = spac[1]

        # prepare artifical 3D dataset(1,M,N) for 2D image (M,N), because to be universal applicable at least one entry
        # in 3rd dimension is expected  
        if dataset_type == ProcessSelDataType.SLICE_2D:
            dataset_noisy = dataset_noisy[np.newaxis,...]

            # make return value 2D array again
            dataset_denoised = func_denoise(dataset_noisy, *params)[0,:,:]

            return dataset_denoised

        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_2D:
            
            # set z gradient spaces to 1/0 = infinity, so that no gradient is z direction is calculated
            # therefor every 2D picture is denoised individually
            
            dataset_denoised = func_denoise(dataset_noisy, *params)

            return dataset_denoised


        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_3D:
            # keep dataset just as it is, if its allready 3D -> dataset_noisy = dataset_noisy

            # set inverted z spacing to any number different then 0 for a 3D denoising

            dataset_denoised = func_denoise(dataset_noisy, *params)

            return dataset_denoised

        else:
            raise TypeError("Dataset must be either 2D or 3D and matching the correct dataset type")



    def tv_denoising_L1(self,img_noisy, lambd, iterations):
        """
        @param f: the K observations of shape MxNxK
        @param alpha: tuple containing alpha1 and alpha2
        @param maxit: maximum number of iterations
        @return: tuple of u with shape MxN and v with shape 2xMxN
        """

        # star argument to take really the value of the variable as argument
        # if 2dim noisy data make it a 3D array, if 3D just let it be
        
        # inverted spacing is used so that h* = 0 is an infinite spacing
        # Parameters
        beta = 1
        theta = 1
        mu = 0.5
        delta = 0.99


        L, M, N = img_noisy.shape
        img = img_noisy.reshape(L*M*N)

        # make operators
        k = self.make_K(L,M,N)

        # initialize primal variables
        u_old = np.zeros(L*M*N)

        # initialize dual variables
        p_old = np.zeros(3*L*M*N)

        # primal and dual step size
        tau_old = self.lip_inv
        sigma = self.lip_inv
    

        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            u_temp = u_old - tau_old * k.T @ p_old
            u_new = self.prox_G_L1(u_temp, img, tau_old, lambd)


            tau_new = tau_old*(1+theta)**0.5
            #print("new tau")
            #print("Tau_n:", tau_new)
            
            while True:
                theta = tau_new/tau_old
                sigma = beta * tau_new
                u_bar = u_new + theta * (u_new - u_old)

                p_temp = p_old + sigma*k@(u_bar)            
                p_new = np.ravel(self.proj_ball(p_temp[0:3*L*M*N].reshape(3, L*M*N)))
                
                #print("calculate norm")
                LS = np.sqrt(beta)*tau_new*(np.linalg.norm(k.T@p_new - k.T@p_old))
                RS = delta*(np.linalg.norm(p_new - p_old))
                #print("LS is:", LS)
                #print("RS is:", RS)
                if  LS <= RS:
                    #print("Update tau!")
                    break
                else: tau_new = tau_new * mu
                #print("reduce tau")
                #print("Tau_n:", tau_new)

            u_old = u_new
            p_old = p_new
            tau_old = tau_new
            
        u_new = u_new.reshape(L,M,N)
        
        return u_new

    def tv_denoising_huberROF(self,img_noisy, lambd, iterations, alpha):

        # Parameters
        beta = 1
        theta = 1
        mu = 0.5
        delta = 0.99


        L, M, N = img_noisy.shape
        img = img_noisy.reshape(L*M*N)


        # make operators
        k = self.make_K(L,M,N)

        # initialize primal variables
        u_old = np.zeros(L*M*N)

        # initialize dual variables
        p_old = np.zeros(3*L*M*N)

        # primal and dual step size
        tau_old = self.lip_inv
        sigma = self.lip_inv    

        
        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            
            u_temp = u_old - tau_old * k.T @ p_old
            u_new = self.prox_G_L2(u_temp, img, tau_old, lambd)

            tau_new = tau_old*(1+theta)**0.5
            #print("new tau")
            #print("Tau_n:", tau_new)

            while True:
                theta = tau_new/tau_old
                sigma = beta * tau_new
                u_bar = u_new + theta * (u_new - u_old)

                divisor = (1 + sigma * alpha)
                p_temp = p_old + sigma*k@(u_bar)            
                p_new = np.ravel(self.proj_ball(p_temp[0:3*L*M*N].reshape(3, L*M*N)/divisor))
                
                #print("calculate norm")
                LS = np.sqrt(beta)*tau_new*(np.linalg.norm(k.T@p_new - k.T@p_old))
                RS = delta*(np.linalg.norm(p_new - p_old))
                #print("LS is:", LS)
                #print("RS is:", RS)
                if  LS <= RS:
                    #print("Update tau!")
                    break
                else: tau_new = tau_new * mu
                #print("reduce tau")
                #print("Tau_n:", tau_new)

            u_old = u_new
            p_old = p_new
            tau_old = tau_new

        u_new = u_new.reshape(L,M,N)
        
        return u_new

    def tv_denoising_L2(self,img_noisy, lambd, iterations):

        # Parameters
        beta = 1
        theta = 1
        mu = 0.5
        delta = 0.99

        L, M, N = img_noisy.shape
        img = img_noisy.reshape(L*M*N)


        # make operators
        k = self.make_K(L,M,N)

        # initialize primal variables
        u_old = np.zeros(L*M*N)

        # initialize dual variables
        p_old = np.zeros(3*L*M*N)

        # primal and dual step size
        tau_old = self.lip_inv
        sigma = self.lip_inv
    

        
        # @ is matrix multiplication of 2 variables

        for it in range(0, iterations):
            print("iterations: " + str(it))           
            # To calculate the data term projection you can use:
            # prox_sum_l1(x, f, tau, Wis)
            # where x is the parameter of the projection function i.e. u^(n+(1/2))
            u_temp = u_old - tau_old * k.T @ p_old
            u_new = self.prox_G_L2(u_temp, img, tau_old, lambd)

            tau_new = tau_old*(1+theta)**0.5
            #print("new tau")
            #print("Tau_n:", tau_new)

            while True:
                theta = tau_new/tau_old
                sigma = beta * tau_new
                u_bar = u_new + theta * (u_new - u_old)

                p_temp = p_old + sigma*k@(u_bar)            
                p_new = np.ravel(self.proj_ball(p_temp[0:3*L*M*N].reshape(3, L*M*N)))
            
                #print("calculate norm")
                LS = np.sqrt(beta)*tau_new*(np.linalg.norm(k.T@p_new - k.T@p_old))
                RS = delta*(np.linalg.norm(p_new - p_old))
                #print("LS is:", LS)
                #print("RS is:", RS)
                if  LS <= RS:
                    #print("Update tau!")
                    break
                else: tau_new = tau_new * mu
                #print("reduce tau")
                #print("Tau_n:", tau_new)

            u_old = u_new
            p_old = p_new
            tau_old = tau_new

        u_new = u_new.reshape(L,M,N)
        
        return u_new
