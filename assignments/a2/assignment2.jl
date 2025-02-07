#=       
            ECON753 - assignment 2
            Paola Villa Paro
=#
cd("/Users/villap/Documents/GitHub/econ753-myrepo/assignments/a2")
#--- packages 
#using Pkg
#Pkg.add("LaTeXStrings")
#Pkg.add("Latexify")
#Pkg.add("PrettyTables")
#Pkg.add(["CSV", "DataFrames"])
#Pkg.add("SpecialFunctions")
using CSV, DataFrames
using DataFrames, Statistics
using Optim
using LinearAlgebra
using SpecialFunctions  #for gamma funtion
using Dates
using BenchmarkTools
using Printf
using PrettyTables
using Latexify
using LaTeXStrings
using DelimitedFiles

#= #########################################
        QUESTION 1
Estimate the parameter vector β using maximum likelihood. Use as the starting value a vector
of zeros. Use four algorithms
######################################### =#

#SETUP DATABASE
data = CSV.read("psychtoday.csv", DataFrame)
display(describe(data))

#= ----------------------------------------------
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

# Start time tracking
start_time = time()
# Optimize using BFGS with numerical derivatives
QNN_result = optimize(beta -> poisson_loglikelihood(beta, X, y), initial_beta, BFGS(), autodiff=:forward)

# Stop time tracking
QNN_time_elapsed = time() - start_time  # Compute elapsed time in seconds

# Extract results
QNN_estimated_beta = Optim.minimizer(QNN_result)  # Optimized parameters
QNN_iterations = Optim.iterations(QNN_result)  # Number of iterations
QNN_func_evals = Optim.f_calls(QNN_result)  # Number of function evaluations

# Ensure estimated_beta is a 1D vector (convert column vector if needed)
QNN_estimated_beta = vec(QNN_estimated_beta)


    
#= ----------------------------------------------
 Quasi-Newton with BFGS and an analytical derivative
=#
# Define the Poisson log-likelihood function <- done previously

# Define the gradient of the Poisson log-likelihood function
function poisson_gradient!(gradient, beta, X, y)
        lambda = exp.(X * beta)  # Poisson rate parameter
        gradient .= X' * (lambda .- y)  # Analytical gradient
        return gradient
    end
# Starting value for beta (vector of zeros)
initial_beta = zeros(size(X, 2))

# Define the objective function and gradient for Optim
function objective(beta)
    return poisson_loglikelihood(beta, X, y)
end

function gradient!(gradient, beta)
    return poisson_gradient!(gradient, beta, X, y)
end

# Start time tracking
start_time = time()

# Optimize using BFGS with analytical gradient
QNA_result = optimize(objective, gradient!, initial_beta, BFGS())

# Stop time tracking
QNA_time_elapsed = time() - start_time  # Compute elapsed time in seconds

# Extract results
QNA_estimated_beta = Optim.minimizer(QNA_result)  # Optimized parameters
QNA_iterations = Optim.iterations(QNA_result)  # Number of iterations
QNA_func_evals = Optim.f_calls(QNA_result)  # Number of function evaluations

# Ensure estimated_beta is a 1D vector (convert column vector if needed)
QNA_estimated_beta = vec(QNA_estimated_beta)


 

#= ----------------------------------------------
        Nelder Mead algorithm
=#
# Define the Poisson log-likelihood function <- done previously

# Starting value for beta (vector of zeros)
initial_beta = zeros(size(X, 2))

# Start time tracking
start_time = time()

# Define the objective function
function objective(beta)
    return poisson_loglikelihood(beta, X, y)
end

# Optimize using Nelder-Mead algorithm
NM_result = optimize(objective, initial_beta, NelderMead())

# Stop time tracking
NM_time_elapsed = time() - start_time  # Compute elapsed time in seconds

# Extract results
NM_estimated_beta = Optim.minimizer(NM_result)  # Optimized parameters
NM_iterations = Optim.iterations(NM_result)  # Number of iterations
NM_func_evals = Optim.f_calls(NM_result)  # Number of function evaluations

# Ensure estimated_beta is a 1D vector
NM_estimated_beta = vec(NM_estimated_beta)


#= ----------------------------------------------
        BHH - Hessian Approximation 
=#

# the BHHH algorithm
function bhhh_optimization(X, y, initial_beta; max_iter=1000, tol=1e-6)
    beta = copy(initial_beta)  # Initial parameter vector
    n = length(y)  # Number of observations
    p = length(beta)  # Number of parameters
    iter = 0  # Iteration counter
    func_evals = 0  # Function evaluation counter
    start_time = now()  # Start time

    while iter < max_iter
        iter += 1  # Manually increment iteration counter

        # Compute the score function (gradient) for each observation
        lambda = exp.(X * beta)
        residuals = lambda .- y # Residuals (y_i - lambda_i)
        G = X' .* residuals'  # Score function for each observation (n x p matrix)

        # Approximate the Hessian using the outer product of gradients (BHHH)
        # Replace expectation with sample average (drop the (1/n) scaling factor for simplicity)
        H = G * G'  # Hessian approximation (p x p matrix)

        # Compute the total gradient
        total_gradient = sum(G, dims=2)

        # Increment function evaluations
        func_evals += 1

        # Check for convergence
        if norm(total_gradient) < tol
            println("Converged after $iter iterations.")
            break
        end

        # Update beta using Newton-Raphson step
        beta -= inv(H) * total_gradient

    end
    # Compute elapsed time
    elapsed_time = (now() - start_time).value / 1000  # Convert to seconds

    return beta, iter, func_evals, elapsed_time
end

# Starting value for beta (vector of zeros)
initial_beta = zeros(size(X, 2))

# Run BHHH optimization
BHHH_estimated_beta, BHHH_iterations, BHHH_func_evals, BHHH_time_elapsed = bhhh_optimization(X, y, initial_beta)

# Ensure estimated_beta is a 1D vector (convert from column vector if needed)
BHHH_estimated_beta = vec(BHHH_estimated_beta)  # Converts a column vector to a 1D array

#-------------------------- Construct a single DataFrame with results from all models---------------
df_results = DataFrame(
    Parameter = vcat(
        ["Constant", "Age", "Years Married", "Religiousness", "Occupation", "Marriage Rating"], 
        ["Number of Iterations", "Function Evaluations", "Elapsed Time (seconds)"]
    ),
    QNN  = vcat(QNN_estimated_beta[:, 1], QNN_iterations, QNN_func_evals, QNN_time_elapsed),
    QNA  = vcat(QNA_estimated_beta[:, 1], QNA_iterations, QNA_func_evals, QNA_time_elapsed),
    NM   = vcat(NM_estimated_beta[:, 1], NM_iterations, NM_func_evals, NM_time_elapsed),
    BHHH = vcat(BHHH_estimated_beta[:, 1], BHHH_iterations, BHHH_func_evals, BHHH_time_elapsed),
    )
# Export to LaTeX
open("linear_models.tex", "w") do io
    pretty_table(io, df_results, backend=Val(:latex), alignment=[:l, :c, :c, :c, :c, :c])
end

#= #########################################
        QUESTION 2
 Report the eigenvalues for the Hessian approximation for the
 BHHH MLE method from the last question
######################################### =#

# Function to compute BHHH Hessian approximation at given beta
function compute_bhhh_hessian(X, y, beta)
    lambda = exp.(X * beta)
    residuals = lambda .- y
    G = X' .* residuals'  # Score function for each observation
    H = G * G'  # BHHH Hessian approximation
    return H
end

# Compute Hessian approximation at initial_beta
H_initial = compute_bhhh_hessian(X, y, initial_beta)
eigenvalues_initial = eigvals(H_initial)

# Compute Hessian approximation at estimated parameters
H_estimated = compute_bhhh_hessian(X, y, BHHH_estimated_beta)
eigenvalues_estimated = eigvals(H_estimated)

# Print results
println("Eigenvalues of Initial Hessian Approximation:")
println(eigenvalues_initial)

println("Eigenvalues of Hessian at Estimated Parameters:")
println(eigenvalues_estimated)

# Function to format a matrix and eigenvalues as a LaTeX table
function matrix_eigen_to_latex(H, eigenvalues, caption, label)
    rows, cols = size(H)
    latex_str = "\\begin{table}[H]\n\\centering\n"
    latex_str *= "\\caption{$caption}\n"
    latex_str *= "\\label{$label}\n"
    latex_str *= "\\begin{tabular}{" * "c " ^ cols * "}\n\\hline\n"

    # Add Hessian matrix
    latex_str *= "\\multicolumn{$cols}{c}{\\textbf{Hessian Matrix}} \\\\ \\hline\n"
    for i in 1:rows
        row_str = join(string.(round.(H[i, :], digits=6)), " & ") * " \\\\ \\hline\n"
        latex_str *= row_str
    end

    # Add eigenvalues row
    latex_str *= "\\multicolumn{$cols}{c}{\\textbf{Eigenvalues}} \\\\ \\hline\n"
    eigen_str = join(string.(round.(eigenvalues, digits=6)), " & ") * " \\\\ \\hline\n"
    latex_str *= eigen_str

    latex_str *= "\\end{tabular}\n\\end{table}\n"
    return latex_str
end

# Generate LaTeX tables
latex_table_initial = matrix_eigen_to_latex(H_initial, eigenvalues_initial, 
    "Hessian Approximation and Eigenvalues at Initial Parameters", "tab:hessian_initial")

latex_table_estimated = matrix_eigen_to_latex(H_estimated, eigenvalues_estimated, 
    "Hessian Approximation and Eigenvalues at Estimated Parameters", "tab:hessian_estimated")

# Save LaTeX tables to a .tex file
open("hessian_eigen_tables.tex", "w") do f
    write(f, latex_table_initial * "\n\n" * latex_table_estimated)
end

#= #########################################
        QUESTION 3
estimate the model using the NLLS method we went over in class
######################################### =#
#Following steps from class: 
#=
1. Initialize $θ₀$
2. Define Jacobian 
3. The gradient of $S(θ) = \sum_N J(θ)^{\top}f(θ)$.
4. The Hessian 
5. Approximate the Hessian by dropping the second term (to guarentee positive definite matrix). 
6. Search step is then
7. Repeat until convergence
=#
# Compute residuals f(θ) - y
function residuals(beta, X, y)
    lambda = exp.(X * beta)
    return lambda - y  # Residuals should be (N × 1)
end

# Compute the Jacobian matrix J(θ)
function compute_jacobian(X, beta)
    lambda = exp.(X * beta)  # Poisson mean
    return X .* lambda  # Jacobian should be (N × p)
end
# Gauss-Newton Algorithm for Poisson NLLS

# Gauss-Newton Algorithm for Poisson NLLS
function gauss_newton_poisson(X, y, beta0; tol=1e-6, max_iter=100)
    beta = copy(beta0)  # Ensure it's (p × 1)
    iter = 0
    func_evals = 0
    start_time = now()

    while iter < max_iter
        iter += 1
        r = residuals(beta, X, y)  # Residuals (N × 1)
        J = compute_jacobian(X, beta)  # Jacobian (N × p)
        func_evals += 1

        # Compute Hessian approximation H ≈ JᵀJ
        H = J' * J  # (p × p)
        # Compute gradient g = Jᵀr
        g = J' * r  # (p × 1)

        # Compute search direction d = - (JᵀJ)⁻¹ Jᵀr
        d = -H \ g  # Ensure d is (p × 1)

        # Update parameters (ensure dimensions match)
        beta = beta .+ d  # Element-wise update to avoid broadcasting issues

        # Check convergence
        if norm(d) < tol
            println("Converged in $iter iterations.")
            break
        end
    end

    elapsed_time = (now() - start_time).value / 1000  # Convert to seconds
    return beta, iter, func_evals, elapsed_time
end


# Starting value for beta (vector of zeros)
initial_beta = zeros(size(X, 2))

# Run Gauss-Newton optimization 
NLLS_estimated_beta, NLLS_iterations, NLLS_func_evals, NLLS_time_elapsed = gauss_newton_poisson(X, y, initial_beta)
NLLS_estimated_beta = vec(NLLS_estimated_beta)  # Converts a column vector to a 1D array

# Construct a single DataFrame with results from all models
all_models = DataFrame(
    Parameter = vcat(
        ["Constant", "Age", "Years Married", "Religiousness", "Occupation", "Marriage Rating"], 
        ["Number of Iterations", "Function Evaluations", "Elapsed Time (seconds)"]
    ),
    QNN  = vcat(QNN_estimated_beta[:, 1], QNN_iterations, QNN_func_evals, QNN_time_elapsed),
    QNA  = vcat(QNA_estimated_beta[:, 1], QNA_iterations, QNA_func_evals, QNA_time_elapsed),
    NM   = vcat(NM_estimated_beta[:, 1], NM_iterations, NM_func_evals, NM_time_elapsed),
    BHHH = vcat(BHHH_estimated_beta[:, 1], BHHH_iterations, BHHH_func_evals, BHHH_time_elapsed),
    NLLS = vcat(NLLS_estimated_beta[:, 1], NLLS_iterations, NLLS_func_evals, NLLS_time_elapsed)
    )
# Export to LaTeX
open("allmodels.tex", "w") do io
    pretty_table(io, all_models, backend=Val(:latex), alignment=[:l, :c, :c, :c, :c, :c])
end