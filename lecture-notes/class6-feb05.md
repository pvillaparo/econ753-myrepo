# Class 5
**Date:** 2025-02-05  
**Source:** [Lecture]  

---
#### notes from BINARY CHOICE
besides the DGP we need to cosnider if the idea of approx this dgp with linear can cause issues, especially with the residuals
### Endogeneity 
- We saw we can deal it using MLE, but we need a full specidication of the structural form $X^\epsilon$. 
- Control Fucntion: we only need to assume a first stage relatioship  
Example: $q= \beta price + \epsilon $ assume a control funtion for $price$ but requires stronge requirements than the calssic 2sls
BUt htey rely on continuous variables , shouldn't be discrete
- Special Regressor:  $D  = I ( X' \beta + V + \varepsilon \geq 0 )$
 we need a big support of V such we ensure variability in D.

Special regressors is like a linear regression weighted by the normalized distribution of V.
V is going to be the reason why x and $\epsilon$ are moving , therefore anything left is not driven by endogeneity but between X and Y.

We can also enrich V with instruments Z . So this kind of approach is important when we have an endogeneity problem with discrete choice. 
- What should we do? try all estimators and check robustness 

Empirical Example: Dong and Lewbel: migration decision, we can test all these approaches (although the control function would be misspecified because they have a discrete variable V).
Compares models with theoretically correct aproaches to bad ones. 


## MULTINOMIAL CHOICE 

Deicisons agents make multinomial decisions :
 ### Non parametric Setup
 period t, J alternatives, agents i, agent choose j with probaility $P_{ijt}$ , agent i receives utility $U_{ij}$ from choosing j. 

 $P_{ij} = Prob( U_{ij} > U_{ik} \quad \forall j \neq k)$

 We can separate the utility into observed and unobserved components. Consider additive separability condition. 
$$
P_{ij} = Prob( U_{ij} > U_{ik} \quad \forall j \neq k)\\
 =Prob( V_{ij} + \varepsilon_{ij} > V_{ik} + \varepsilon_{ij} \quad \forall j \neq k)\\
 = Prob( \varepsilon_{ij}-\varepsilon_{ik} > V_{ik} - V_{ij} \quad \forall j \neq k)
$$
$$
P_{ij} = Prob( \varepsilon_{ij}-\varepsilon_{ik} > V_{ik} - V_{ij} \quad \forall j \neq k)\\
= \int I( \varepsilon_{ij}-\varepsilon_{ik} > V_{ik} - V_{ij} ) f( \varepsilon_i) \partial \varepsilon_i 
$$
- We need to compute J dimension integral. and important to check assitive separatibility.
- The true structural Utility  is not observed because we only see the  factor that enters in the utility. Example adding a constant to the utility don't change the choice decision.  
- We cannot have individual specific factos into utility : no income effects. 
- we shut down unobserved correlation of error terms with nromal logit, probit .
- we normalized the variance of the EVT1 with $\pi/6$
#### Identification
Non parametritation of $V$ or $U$
### Multinomial Logit (MNL)
- Logit has closed form choice probabilities  
- Expected utility also has a closed form
- this allos a simple computation of Consuemr Surplus 
- In general these models assume you have individual data
#### Identification
- we can only identify the difference in indirect utilities not levels. 
- most sensititve normalization is to allow for an outside option.
- we can rescale every coeficient by sigma.
- having a raking data type is more powerful. 