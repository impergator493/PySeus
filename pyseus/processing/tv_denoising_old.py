# TV-L1 implementation based on Matlab User method
# from Convex optimization PD Chambolle Pock Paper

# define as class methods, then there is no need for initializing
# but predefined values must be defined in another way then

import numpy as np

from ..settings import ProcessSelDataType



class TV_Denoise():

    def __init__(self):
        
        self.tau = 0.01

        # inverted isotrop spacing x and y dim
        self.h_inv = 1.0
        # inverted spacing of z dim (slice)
        self.hz_inv = 1.0

        # according to chambolle pock TV denoising paper, invert h back for correct calculation according to original definition
        self.L2 = 8.0/np.square(1/self.h_inv)

        self.sigma = 1./(self.L2*self.tau)

        self.theta = 1.0

        # remove later, when functions are implemented as classes, not neccessary anymore, remove later if it works
        #self.i = None
        #self.iterations = None


    def tv_denoising_gen(self, func_denoise, dataset_type, dataset_noisy, params, spac):
        """ General Denoising Method for all TV types, 2D and 3D """

        self.h_inv = spac[0]
        self.hz_inv = spac[1]


        # if 2D per slice is done, for-loop is needed, for repeating 2D calculation of 3D dataset
        if dataset_type == ProcessSelDataType.WHOLE_SCAN_2D:
           
            dataset_denoised = np.zeros(dataset_noisy.shape)
            slices = dataset_noisy.shape[0]

            for index in range(0, slices):
                    args = (dataset_noisy[index,:,:], *params)
                    dataset_denoised[index,:,:] = func_denoise(*args)

            return dataset_denoised

        #  2D and 3D dataset is automatically correctly handled by methods according to dataset dimensions 
        elif dataset_type == ProcessSelDataType.WHOLE_SCAN_3D or dataset_type == ProcessSelDataType.SLICE_2D:
            
            args = (dataset_noisy, *params)
            dataset_denoised = func_denoise(*args)

            return dataset_denoised

        else:
            raise ValueError("Dataset must be either 2D or 3D")
        

            
    def gradient_img(self, matrix):
        
        # 2 dim for x and y gradient values
        grad = np.zeros(((matrix.ndim,) + matrix.shape))

        grad[0,0:-1,:] = (matrix[1:,:] -matrix[0:-1,:])*self.h_inv
        grad[1,:,0:-1] = (matrix[:,1:] -matrix[:,0:-1])*self.h_inv

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
        div[0,:,:] = (np.r_['0',matrix_div[0,0:-1,:], np.zeros((1,dim1_len))] -np.r_['0',np.zeros((1,dim1_len)), matrix_div[0,0:-1,:]])*self.h_inv
        div[1,:,:] = (np.r_['1',matrix_div[1,:,0:-1], np.zeros((dim0_len,1))] -np.r_['1',np.zeros((dim0_len,1)), matrix_div[1,:,0:-1]])*self.h_inv

        return div[0] + div[1]

    def gradient_img_3D(self, dataset):

                    
        grad = np.zeros(((dataset.ndim,) + dataset.shape))
        # dim 0 should be z: z can have another spacing as x and y
        grad[0,0:-1,:,:] = (dataset[1:,:,:] -dataset[0:-1,:,:])*self.hz_inv
        grad[1,:,0:-1,:] = (dataset[:,1:,:] -dataset[:,0:-1,:])*self.h_inv
        grad[2,:,:,0:-1] = (dataset[:,:,1:] -dataset[:,:,0:-1])*self.h_inv

        return grad

    def divergence_img_3D(self, matrix_div):
        
        div = np.zeros_like(matrix_div)

        dim0_len = matrix_div.shape[1]
        dim1_len = matrix_div.shape[2]
        dim2_len = matrix_div.shape[3]
        
        # according to other program, first row/column should be taken from old, and last from old also but negative
        div[0,:,:,:] = (np.r_['0',matrix_div[0,0:-1,:,:], np.zeros((1,dim1_len,dim2_len))] -np.r_['0',np.zeros((1,dim1_len,dim2_len)), matrix_div[0,0:-1,:,:]])*self.hz_inv
        div[1,:,:,:] = (np.r_['1',matrix_div[1,:,0:-1,:], np.zeros((dim0_len,1,dim2_len))] -np.r_['1',np.zeros((dim0_len,1,dim2_len)), matrix_div[1,:,0:-1,:]])*self.h_inv
        div[2,:,:,:] = (np.r_['2',matrix_div[2,:,:,0:-1], np.zeros((dim0_len,dim1_len,1))] -np.r_['2',np.zeros((dim0_len,dim1_len,1)), matrix_div[2,:,:,0:-1]])*self.h_inv

        return div[0] + div[1] + div[2]


    def tv_denoising_L1(self,img, lambda_rat, iterations):
        """ Denoising with TVfor regularization term and L1 on data term"""
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
        """ Denoising with Huber regularization and L2 on data term"""

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


# L2 denoising
# M: @TODO define as class method?
    def tv_denoising_L2(self,img, lambda_rat, iterations):
        """ Denoising with TV for regularization term and L2 on data term"""


        # check dimensions, so that just 2D image is processed at once
        # loop for multiple slices
        # maybe for first try hand over 3D image but just select 2D slice?

        #p_n+1
        #u_n+1 = model_L2 e.g.
        
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
        # TODO add offset to negative and scale afterwards
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
            x_n = ((x_n - self.tau*(-div_func(y_n))) + self.tau*lambda_rat*x_0) / (1 + self.tau * lambda_rat)
            
            x_bar_n = x_n + self.theta*(x_n - x_old)


        img_denoised = x_bar_n

        return img_denoised
