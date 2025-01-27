"""
Special Cases for optimization: tries to find update for hessian (DFP, BFGS) , but there is a more theoretical 
    way to think about hessians. We can ex-ante write down the function of the hessian. 
* Maximumm Likelikood 
    we can use the score function as the derivative o fth eloglike at that observation data. 
        so nxk matrix: obs * vars evaluate -> the expectation of the inner product will give us the information matrix
        this new matrix will replac ethe hessian for the initial approximaiton. 
        we can code this derivate ourselves , and update the old value with the new function 
   
"""

"""
* Non Least Squares
    use for non linear model, here we define a residual function 
    there is an algotrith to update the hessian: 
        initialize 
        define jacobian (this is the score analogy)
        rewrite the gradient of the function using the jacobian 
        to get the actual hessian: inner product of jacobians + a cross terms of derivatives
        to estimate the Hessian, we dorp the second term to guarantee positive definite matrix. (no a mathematic arguments, just convenient)
        define the search step that it will be use to update the old value 

"""