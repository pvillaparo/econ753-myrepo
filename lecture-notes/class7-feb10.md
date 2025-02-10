# Class 7
**Date:** 2025-02-10  
**Source:** [Lecture]  

---
## Multinomial Logit
#### Identification 
set and outside option is important 
#### Scale of Utility
    Important is that we have to rescale tuility definition by sigma, this is important because we will have betas rescale to sigmas and able to compare one to each other
#### Taste Variation
- we can include taste variation across individuals in different ways. 
- having an outside option allows us to have IIA
#### IIA
- this feature IIA allows you to only take share between option and estimate the model wihtout actually needing the full data.
- we can capture the substitution and cross substitution patterns 
- also we can get the elasticities associate to the disbstitutions partterns.
- we want to make the model richer 
- Cross eleasticity evaluates the sensitivity if one of the goods changes in characteristics how does affect it to other options.. 

#### Estimation
- estimation is straightforward with Multinomial Logit
##### Aggregate Data
- when cna assume individuals are similar and aggregate them to a share level. 
- For example: Choice probabilities por high income and for low income we can use it to calculate the model.
- the prices elasticities we get initially only have price effect and we don't have income effect .

## Can we do better?
- Multinomial Probit 
    - but no close form for the shares and choice probabilities . 
    - proweful when we don't have many options 

## Relaxing IIA
- we make $\epsilon_{ij}$ more flexible and add an additional extrem valua associate to group distribution.
- as a result we have GEV  

$u_{ij} =   x_{ij} \beta  + \eta_{g} + \varepsilon_{ij}$
- Nested Logit the inmmediate application. 

## Nested Logit
-  No cross group sustitution 
- We can calculate the substitutions and elasticities 
- we have IIA property but withtin the group G , and IIA across groups g,

## Mixed Logit
- Relax IIA porperty haivng  a continuous logits 

$ u_{ijt} = x_j \beta + \mu_{ij} + \varepsilon_{ij} $

$ s_{ij} = \int \frac{\exp[x_{j} \beta + \mu_{ij} ]}{1+\sum_k \exp[x_{k} \beta + \mu_{ik} ]} f(\mu_i | \theta) $

### Mixed Random Coefficient logits 
- introducre more person specific variables
- we can introduce different kinds of heterogeneity  and make a more complex assignation of individual taste and make the substitution patterns more different.
- Choice porb  would be: 

$P_{ij}(\theta) = \int \frac{ e^{V_{ij}(\nu_i,\theta)}}{\sum_k e^{V_{ik}(\nu_i,\theta)}} f(\nu_i) \partial \nu$

- to estiamte we can use Monte carlo integration, but Quadrature cna be more powerful

## Even more Flexibility
- disrtibution of preferences  can not be even normal
- we will get some weights $w$ for each share. 