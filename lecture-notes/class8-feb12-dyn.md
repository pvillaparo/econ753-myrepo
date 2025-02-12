# Class 9
**Date:** 2025-02-12  
**Source:** [Lecture]  

---

# DYNAMICS DISCRETE CHOICE
- estimating and modelling dynamics is very computational, but more restrictive assumptions about endogeneity.
- many economic problems are dynamic by nature 
## Basic model
- discrete time t, states, discrite actions 
- agents beliefs : is a function of tomorrows state depending on states today and actions today. 
- agents optimization 
- Here error is structural  , an state itself , so it's part of the model 
- We estiamte: structural parameters governing preferences
    -   trnasition probabilities : ove from one state to another. 

### Dynamic Programming 
- transition between states follows conditional independence of X . 
## Aproches 
### John Rust 
- Optimal stopping problem with a cuttoff mileage threshold X* above whichc a bus will have its engine replaced
- is the agent in this model sophiftificated such actualy the person is solving in this way? 
#### model 
- parameters of cost function, parameters of distribution , parameters that govern state transtition 
- sequence of actions: replace or not , generatinf a flow utility 
- x is a parameter about mileage , depends of trnasition parameters and milieage yesterday. Trnasition process can be estiamted independlty because epsilon is not interacting with X
- replacement cost and flow cost are the dynamic parameters, we need to use revelead preference for this. 
- ll of observing:  replace transtition + state transition
- this LL replace the problem from the individual 
- important to incorporate unobservables 
#### Estimation
- the choice probabilities includes an expected value that we don't know 

    -  here we are going to estiamte EV , expected value fucntion that depends of other EV , we can get the correct vector EV using a iteration process to get convergence 
    

