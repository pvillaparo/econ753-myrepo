#=
OPTIMIZATION
=#
#using Pkg
#Pkg.add("Optim")
"""Grid Search
A more rudementary search but gives you the whole vision of the search=#
"""

"""Golden Search method 
given a funtion  we try to find the minimum,  and we want to make sure the interiorm point is lower
    to the exterior points , then we keep updating such the middle point is the lowest
     this is the computational application of extreme value .
     wen want eproportional intervales analysis for each case, we will find the ratio of the intervales = golden ratio
    to get the least itme of interval evaluations
"""

"""Nelder-Mead optimization
if we don't know the derivative, this method is useful. 
To find max: start with 3 starting points , and using reflect we find that point c gives a funtion with gih value than the other 2 points
we can keep expanding, contracting, and shrinking to find if this is the worse or best point.
Note that we cannot certainly say we will be able to find the global min 
"""

using Optim  # For optimization routines

# Banana function:
f(x) = -100*(x[2]-x[1]^2)^2-(1-x[1])^2
#import Pkg; Pkg.add("BenchmarkTools")
using BenchmarkTools  # For timing
@time result = optimize(x -> -f(x), [1.0, 0.0], NelderMead())
x = Optim.minimizer(result)

"""
Newton Raphson Method 
we use quadratic approximations, at each point of analysis we calculate as if our funtion is quadratic, 
this can be problematic if our funciton is kinda flat (we use the minimum of the first quadratic and go to next point)
we start with a guess and continue with a 2nd order taylor expansion, this approx relies on the sjhape of the function at the first guess (second derivative)
note that the second derivative (Hessian) has to well-behave = Negaticve definite, remeber that the hessian gives you the direction of the approximation.
that's why a "easy" way to fix the hessian is assigning it a constant value
"""

"""
Quasi-Newton methods
approxiamte the hessian as a identity -> Steepest Ascentt, works but steps can move slower
Also other methods DFP, BFGS .  Different ways to update the hessian, the steps actually depend on the shape of the funciton
"""

"""
COmparison
neldermead takes more iterations, don't use any derivative info, has to evaluate the function more times, but BFGS is doing more things in the computer
gradientdescent takes soooo many iterations and actually prob requires way more iterations to converge (for banana function)
    

"""