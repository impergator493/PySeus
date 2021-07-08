# H1 = L2 denoising

# M: @TODO define as class method?
    def tv_denoising_L2(self,img, lambda_rat, iterations):

        # check dimensions, so that just 2D image is processed at once
        # loop for multiple slices
        # maybe for first try hand over 3D image but just select 2D slice?

        #p_n+1
        #u_n+1 = model_L2 e.g.
        
        
        max_val = img.max()

        # for negative values also normalize?
        # TODO add offset to negative and scale afterwards
        if max_val > 1.0:
            x_0 = img/max_val
        else:
            x_0 = img


        # first initialized for x_0, y_0, x_bar_0
        y_n = self.gradient_img(x_0)
        x_n = x_0
        x_bar_n = x_n

        for i in range(iterations):


            y_n_half = y_n + self.sigma*self.gradient_img(x_bar_n)
            y_n_half_norm = (y_n_half[0]**2 + y_n_half[1]**2)**0.5
            y_n_half_norm[y_n_half_norm<1] = 1
            y_n  = y_n_half/y_n_half_norm

            x_old = x_n
            x_n = ((x_n - self.tau*(-self.divergence_img(y_n))) + self.tau*lambda_rat*x_0) / (1 + self.tau * lambda_rat)
            
            x_bar_n = x_n + self.theta*(x_n - x_old)


        img_denoised = x_bar_n

        return img_denoised