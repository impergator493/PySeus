# TV-L1 implementation based on Matlab User method

# define as class methods, then there is no need for initializing
# but predefined values must be defined in another way then

import numpy as np



class TV():

    def __init__(self):
        
        self.tau = 0.01

        self.h2 = 1.0

        self.L2 = 8.0/self.h2

        self.sigma = 1./(self.L2*self.tau)

        self.theta = 1.0


    def tv_denoising_gen(self, func_denoise, dataset_type, dataset_noisy, params):
        """ General Denoising Method for all TV types, 2D and 3D """

        # if 2D per slice is done, for loop is needed, for repeat 2D calculation of 3D dataset
        if dataset_type == 2:
           
            dataset_denoised = np.zeros(dataset_noisy.shape)
            slices = dataset_noisy.shape[0]

            for index in range(0, slices):
                    args = (dataset_noisy[index,:,:], *params)
                    dataset_denoised[index,:,:] = func_denoise(*args)

            return dataset_denoised

        #  2D and 3D dataset is automatically correctly handled by methods according to dataset dimensions 
        elif dataset_type == 1 or dataset_type == 3:
            
            args = (dataset_noisy, *params)
            dataset_denoised = func_denoise(*args)

            return dataset_denoised

        else:
            raise ValueError("Dataset must be either 2D or 3D")
        

            
    def gradient_img(self, matrix):
        
        # 2 dim for x and y gradient values
        grad = np.zeros(((matrix.ndim,) + matrix.shape))

        grad[0,0:-1,:] = (matrix[1:,:] -matrix[0:-1,:])
        grad[1,:,0:-1] = (matrix[:,1:] -matrix[:,0:-1])

        return grad

    def divergence_img(self, matrix_div):
        
        # parameter matrix is 2D normally, so 2D must be given to gradient function

        div = np.zeros_like(matrix_div)

        # Y
        dim0_len = matrix_div.shape[1]
        # X
        dim1_len = matrix_div.shape[2]
        
        # according to other program, first row/column should be taken from old, and last from old also but negative
        # the number specifies along which axis matrix should be concenated
        div[0,:,:] = np.r_['0',matrix_div[0,0:-1,:], np.zeros((1,dim1_len))] -np.r_['0',np.zeros((1,dim1_len)), matrix_div[0,0:-1,:]]
        div[1,:,:] = np.r_['1',matrix_div[1,:,0:-1], np.zeros((dim0_len,1))] -np.r_['1',np.zeros((dim0_len,1)), matrix_div[1,:,0:-1]] 

        return div[0] + div[1]

    def gradient_img_3D(self, dataset):

                    
        grad = np.zeros(((dataset.ndim,) + dataset.shape))

        grad[0,0:-1,:,:] = (dataset[1:,:,:] -dataset[0:-1,:,:])
        grad[1,:,0:-1,:] = (dataset[:,1:,:] -dataset[:,0:-1,:])
        grad[2,:,:,0:-1] = (dataset[:,:,1:] -dataset[:,:,0:-1])

        return grad

    #@TODO Implement 3D taken over primarly from 2D
    def divergence_img_3D(self, matrix_div):
        
        div = np.zeros_like(matrix_div)

        dim0_len = matrix_div.shape[1]
        dim1_len = matrix_div.shape[2]
        dim2_len = matrix_div.shape[3]
        
        # according to other program, first row/column should be taken from old, and last from old also but negative
        div[0,:,:,:] = np.r_['0',matrix_div[0,0:-1,:,:], np.zeros((1,dim1_len,dim2_len))] -np.r_['0',np.zeros((1,dim1_len,dim2_len)), matrix_div[0,0:-1,:,:]]
        div[1,:,:,:] = np.r_['1',matrix_div[1,:,0:-1,:], np.zeros((dim0_len,1,dim2_len))] -np.r_['1',np.zeros((dim0_len,1,dim2_len)), matrix_div[1,:,0:-1,:]]
        div[2,:,:,:] = np.r_['2',matrix_div[2,:,:,0:-1], np.zeros((dim0_len,dim1_len,1))] -np.r_['2',np.zeros((dim0_len,dim1_len,1)), matrix_div[2,:,:,0:-1]] 

        return div[0] + div[1] + div[2]


    def tv_denoising_L1(self,img, lambda_rat, iterations):
        # check dimensions, so that just 2D image is processed at once
        # loop for multiple slices
        # maybe for first try hand over 3D image but just select 2D slice?

        #p_n+1
        #u_n+1 = model_L2 e.g.
        
        
        # automatic selection of correct gradient and divergence function
        grad_func = None
        div_func = None

        if img.ndim == 2:
            grad_func = self.gradient_img
            div_func = self.divergence_img
        elif img.ndim == 3: 
            grad_func = self.gradient_img_3D
            div_func = self.divergence_img_3D

        
        max_val = img.max()

        # for negative values also normalize?
        if max_val > 1.0:
            x_0 = img/max_val
        else:
            x_0 = img


        # first initialized for x_0, y_0, x_bar_0
        y_n = grad_func(x_0)
        x_n = x_0
        x_bar_n = x_n

        for i in range(iterations):


            y_n_half = y_n + self.sigma*grad_func(x_bar_n)
            y_n_half_norm = (y_n_half[0]**2 + y_n_half[1]**2)**0.5
            y_n_half_norm[y_n_half_norm<1] = 1
            y_n  = y_n_half/y_n_half_norm

            x_old = x_n
            x_n_half = x_n + self.tau*div_func(y_n)

            x_n =   (x_n_half - self.tau * lambda_rat) * (x_n_half - x_0 > self.tau * lambda_rat
                    ) + (x_n_half + self.tau * lambda_rat) * (x_n_half - x_0 < -self.tau * lambda_rat
                    ) + (x_0) * (abs(x_n_half - x_0) <= self.tau * lambda_rat)

                       
            x_bar_n = x_n + self.theta*(x_n - x_old)


        img_denoised = x_bar_n

        return img_denoised


    # Algorithm 3: according to the paper this should work with ALG3.
    # Alg1, Alg2 not possible like with L1, L2?
    def tv_denoising_huberROF(self,img, lambda_rat, iterations, alpha):

        # maybe calculate later sigma on oneself?
        # sigma = .....

        # alpha = 0.05

        grad_func = None
        div_func = None

        # automatic selection of correct gradient and divergence function
        if img.ndim == 2:
            grad_func = self.gradient_img
            div_func = self.divergence_img
        elif img.ndim == 3: 
            grad_func = self.gradient_img_3D
            div_func = self.divergence_img_3D

        max_val = img.max()

        # for negative values also normalize?
        if max_val > 1.0:
            x_0 = img/max_val
        else:
            x_0 = img


        # first initialized for x_0, y_0, x_bar_0
        y_n = grad_func(x_0)
        x_n = x_0
        x_bar_n = x_n

        for i in range(iterations):


            divisor = (1 + self.sigma * alpha)
            y_n_half = (y_n + self.sigma*grad_func(x_bar_n))
            y_n_half_norm = ((y_n_half[0]/divisor)**2 + (y_n_half[1]/divisor)**2)**0.5
            y_n_half_norm[y_n_half_norm<1] = 1
            y_n  = (y_n_half/divisor)/y_n_half_norm

            x_old = x_n
            x_n = ((x_n + self.tau*div_func(y_n)) + self.tau*lambda_rat*x_0) / (1 + self.tau * lambda_rat)
            
            x_bar_n = x_n + self.theta*(x_n - x_old)


        img_denoised = x_bar_n

        return img_denoised


    def tv_denoising_huberROF_3D(self,img, lambda_rat, iterations, alpha):

        # maybe calculate later sigma on oneself?
        # sigma = .....

        # alpha = 0.05

        max_val = img.max()

        # for negative values also normalize?
        if max_val > 1.0:
            x_0 = img/max_val
        else:
            x_0 = img


        # first initialized for x_0, y_0, x_bar_0
        y_n = self.gradient_img_3D(x_0)
        x_n = x_0
        x_bar_n = x_n

        for i in range(iterations):


            divisor = (1 + self.sigma * alpha)
            y_n_half = (y_n + self.sigma*self.gradient_img_3D(x_bar_n))
            y_n_half_norm = ((y_n_half[0]/divisor)**2 + (y_n_half[1]/divisor)**2)**0.5
            y_n_half_norm[y_n_half_norm<1] = 1
            y_n  = (y_n_half/divisor)/y_n_half_norm

            x_old = x_n
            x_n = ((x_n + self.tau*self.divergence_img_3D(y_n)) + self.tau*lambda_rat*x_0) / (1 + self.tau * lambda_rat)
            
            x_bar_n = x_n + self.theta*(x_n - x_old)


        img_denoised = x_bar_n

        return img_denoised

    def tv_alg1(self):
        pass

    def tv_alg2(self):
        pass

    def tv_alg3(self):
        pass


