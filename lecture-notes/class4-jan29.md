# Class 4
**Date:** 2025-01-29  
**Source:** [Lecture]  

---

## Numerical Integration 
we have seen differents ways of approximation . 
wehn we are evaluating Gaussian Quadrature : when we wanted to match 2n moments of weight funtion g(x) , and to figure out the nodes and weights, we need to solve a system of nonlinear equations : 6 unknows, and 6 equaitons. so we get 3 weights and 3 nodes we can find the solution. 
Beware of the number of dimensions you are working, because it can overflow the number of calculaitons.

Note: that usually when are estimating demand, we use monte Carlo estiamtion. But another alternaitve is to use quadrature because is more computational efficiently. 

## Examples
### Gravity in International Trade
**Anderson & Van Wincoop (2003 AER ) - Gravity with Gravitas**

Main regression
$$ln(x_{ij}) = α_1 + α_2 log(y) + α_3 log(y_j) + α_4 log(d_{ij}) α_5 δ_{ij} +ε_{ij}$$
derive this gravity equation using trade costs and CES   preferences , to describe how much quality are between 2 countries. 
Trade flow equation: are funciton of normalize income, times, income normalize by general equilibrium price resistant terms :
  $$x_{ij} = (y_i * y_j/y_w) * (t_{ij}/P_i*P_j)^{(1-σ)}$$
we can not pickup the parameter sigma and calculate the flow because prices eq are not observed in the deta but are derived themselves in the model . 
The solution woudl be guess the parameters first, and the solve the model to find these prices . we finally can estiamre the trade flow.
guess sigma. guess theta , and solve the equations for P. Note that the estimated flows closer to the real values will define the correct values for the aprameters.

In the empirical application, the aiuthors want to find theta.
Summarizing:
start with an initial guess for sigma(and other parameters.), calculate the initial trade costs , then solve for all P and update them until they converge. then we predicgt the trade flows for this model. then we define the objective funtion as a nonlinear least square that comapres estiamted and real trade flows, until they converge. 

in this setup we are trying to match trade flows to the data, so we guess parameters until they match. But sometimes they cannot match at the minimum value, so maybe there should be a theoretical error term that completes the model and explains why the model doesn't match.

### Differentiated Products Demand
**Berry, Levinsohn, Pakes (1995 ECMA)**

Estimate discrete choice demand for differentiated products , they use product-level data , with endogeneous prices and random effect on consumer preferences for the cars. 

Consumer $i$ has indirect utility for product $j$ that is a function of product characteristics $(x,ξ)$, price $(p)$, and a random match term $ε$: 
$$
u_{ijt} = x_{jt}\beta + ν σ_i x_{jt} - α p_{jt}  + ξ_{jt} + ε_{ijt}
$$
Decisions are made among all products considering an outside option. if we assume epsilon is a extreme value distribution, we have a logit model, this means that the choice probability for any given good has an analytical form (avoiding thenasty integral): 

Choice probabilities are 
$$
s_{ijt} = \frac{exp(\delta_{jt} + ν σ_i x_{jt})}{1 + \sum_{k\in\mathcal{J}} exp(\delta_{kt} + ν σ_i x_{kt})}
$$

The data i have is just market share , so thsi eq at tjhe individual level is not helping, so what we do is to aggregate the choice probabilities weighted by the distribution we assumed by the sigma. 
$$
s_{jt} = \int\frac{exp(\delta_{jt} + ν σ_i x_{jt})}{1 + \sum_{k\in\mathcal{J}} exp(\delta_{kt} + ν σ_i x_{kt})} dG(\sigma)
$$
we evaluate this funciton many times and weighted accordingly to get s

Estimation: 
note that we cannot just take the likelihood between estiamted shares and real share because prices are endogenous 
BLP propose to estimate this model using other estimating equations, to get the endogeneity. They use instrumental variables.
$$
\delta_{jt} = x_{jt}\beta - α p_{jt}  + ξ_{jt}
$$
we cannot take delta and run an IV for prices, but delta is not data is an additional variable, 

But we will keep the restriction on the  distribution on $ ξ_{jt}$ to create the moment equality. 
We can know real delta when we match the theoretical shares and the real shares because this would generate a single delta that allows that match!
$$
s^{data} = s_{jt}(δ;θ)
$$
So as the gravity example, we can get this delta using a contraction mapping .

once we get the new delta, we know can infer the associated $ ξ_{jt}$ and minimize the momemnt vector of this parameter to update the guessed $ ν $ 



