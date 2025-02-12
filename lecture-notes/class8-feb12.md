# Class 8
**Date:** 2025-02-12  
**Source:** [Lecture]  

---
## Multinomial logit
### Demand and Supply in Market Equilibrium
- we have seen that the control fucntion has to be monotone, so would be easy to see in binomial than multinomial
- the special case needed a large support, over the multinomial space
### Endogeneity 
- Instead of starting the logit model , when we have endogenity we run 2SLS 
- so in this case with multinomial we run GMM IV 
- let's find delta and 2sls , but also we need to find the parameter $\theta$ 
 Rewrite the choice problem as:
  $u_{ij} = \delta_j + \mu_{ij})(\theta) + \epsilon_{ij} $

  where $\delta_j = x_j \beta + \alpha p_j + \xi_j$
### Finding $\delta$
 $s_{j}^{\text{data}} = s_j^{\text{model}}(\theta, \delta)$

### Procedure
- guess theta that aren't delta, 
- find the shares/ choice Probabilities
- let's find out deltas that make  $s_{j}^{\text{data}} = s_j^{\text{model}}(\theta, \delta)$
- We have a vector of deltas so in theory we cna run a simpre iv r for $\delta_j = x_j \beta + \alpha p_j + \xi_j$
 an guess a beta and alpha, and infere the residual 
- we instead form a GMM to recover $\theta , \beta , \alpha$ . 

  $$\min_{\theta,\beta} m(\theta,\beta,\alpha)' \, W \, m(\theta,\beta,\alpha)' $$ 
  where $m(\theta,\beta,\alpha) = (\xi' Z)$

    with instuments we can leverage the GMM to estimate the parameters. 

- the way how we find delta, is that we pick a theta that give us the delta. An then once we have delta, we can use the residual to characterize the residual $\xi_j$ 
- a critic to the solution of the endogeneity problem is what moments we should choose, there can be other  moments that can identified better the random coefficient for example. 
 #### Notes in BLP
 - Function interation for berry invertion is a common approach. 
 - if the function is convex , it may make easier the calc
 - what if we don't force  $s_{j}^{\text{data}} = s_j^{\text{model}}(\theta, \delta)$ we can collect product type deltas in a more flexible way.
 - we saw the case with choice shares, but we can do the same with  individual level data but when aggregating to shares you are loosing info from the indiv idual data. 
 #### Probit with BLP
 - choice prob takes a more complex form , and the berry inversion holds because we are still working with a choice model with substitutes. 

 #### Some unresolved issues
 - because we have so many parameters, we have IV s

 $\mu_j = \sigma v_i X_j , v \sim N(0,1)$ we estiamte sigma, the unobserved random coeff is $v_i$ . if we have a sigma=0, individual will substitute depending on X_i, but if it's high it will make that individuals substitutes based on highest X. 
 
 But we can also have
$\mu_j = \eta y_i X_j , y \sim $ income from a particualr distribution 



