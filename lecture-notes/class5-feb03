# Class 5
**Date:** 2025-02-03  
**Source:** [Lecture]  

---
## BINARY CHOICE
### Linear probability model (LPM)
We use OLS , but there are some well known problems using this framework :
- constant marginal effects , not precise 
- the prediction doesn;t need to alighn to 0-1 , so bound it is not precise. And induce heteroskadicity
- The condition of X being uncorrelated to epsilon cannot be satisfied when X has a wide support.

There is a counterfactual where the results show even contrary sign of the marginal effects , (Lewbel Dong and yang)

### Logit / Probit
- expost we rationalize the model into different probabilities distributions CDFs .
- The marginal fucntions now depend on X, no constant anymore 

#### Index models
start with a large number of regressors 
but now marginal effects depend of the data, so you have to be specific about the answer becasue the effect it is goign to depend of who you are treating

#### How to compare across models
Different non-nested way to estimate, but according to Cameron and Trivedi they can be related. 

#### Interactions
no proper to do d-i-d when you have binary outcome because the interactions cannot have interpretations. 

### Probit Model 
to bring the probit to the data we use a Log Likelihood: 
Minimize ln L(beta)
we choose beta to minimize deviations .
the score funciton will be the radient of this lnL respect to beta 
and finally get the hessian . 
we can compute this procedure using Newton-Raphson .
The information matrix is the covariance matrix and can use BHHH to aproximate it. 

Note it's important to have an idea of the Data generating process (DGP) in order to see if a LPM, Logit or Probit would work better. 

### What about endogeneity 
- LPM
- we can specify the distribution of error for first and second stage: this would be a full solution method
- control function estimation
- special regressor methods

#### Solution 1 : MLE 
we have equarion of interest, and what determines $X^e$.. We need to fully specify the distribution of $ε$, e . Still an identification problem because the hessian of this problem should be invertible. With biprobit you can specify both equations and they will estimate the joint probability.

We need to parametrize EVERYTHING : $G$, $F_{e,\epsilon | Z}$

We cannot have an excluded variable, everything have to be included.

#### Solution 2: Control Functions
First stage relationship has to be invertible in the error term. 
get estiamted residuals from first stage and then estimate the initital regression with these residuals, SO it cleans out the correlation between $\epsilon$ and $e$

Stata version is ivprobit , we plug in the first step residuals as an additional variable in the original equation. 


This looks like 2SLS but there are stronger assumptions , one about $\epsion$ and $e$ 

#### Solution 3: Special Regressor