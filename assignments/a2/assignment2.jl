#=       
            ECON753 - assignment 2
            Paola Villa Paro
=#
cd("/Users/villap/Documents/GitHub/econ753-myrepo/assignments/a2")
#--- packages 
#using Pkg
#Pkg.add(["CSV", "DataFrames"])
#Pkg.add("SpecialFunctions")
using CSV, DataFrames
using DataFrames, Statistics
using Optim
using LinearAlgebra
using SpecialFunctions  #for gamma funtion
#= #########################################
        QUESTION 1
Estimate the parameter vector β using maximum likelihood. Use as the starting value a vector
of zeros. Use four algorithms
######################################### =#

#SETUP DATABASE
data = CSV.read("psychtoday.csv", DataFrame)
display(describe(data))

#=
 Quasi-Newton with BFGS and a numerical derivative 
=#
# Define the Poisson log-likelihood function
function poisson_loglikelihood(beta, X, y)
        lambda = exp.(X * beta)  # Poisson rate parameter
        loglik = sum(y .* log.(lambda) .- lambda .- log.(gamma.(y .+ 1)))  # Log-likelihood
        return -loglik  # Return negative log-likelihood for minimization
end
    
# Load the dataset 
    y = data[:, 1]  # Dependent variable: number of affairs
    X = Matrix(data[:, 2:end])  # Independent variables: constant, age, years married, religiousness, occupation, marriage rating
    
# Starting value for beta (vector of zeros)
    initial_beta = zeros(size(X, 2))
    
# Optimize using BFGS with numerical derivatives
    result = optimize(beta -> poisson_loglikelihood(beta, X, y), initial_beta, BFGS(), autodiff=:forward)
    
# Extract the estimated beta values
    estimated_beta = Optim.minimizer(result)
    println("Estimated beta coefficients: ", estimated_beta)



