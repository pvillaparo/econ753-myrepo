using Plots
using Printf
using LinearAlgebra
using NonlinearSolve
#using DifferentialEquations
#using NLsolve
#=      
Bisection Method
operationalize the intermediate Value Theorem to find the root or fixed point, but the function can be not easy to evaluate 
=#
function bisect(f, a, b; tol=1e-4)
    s = sign(f(a))
    x = (a + b) / 2
    d = (b - a) / 2
    xsave = [x]
    
    while d > tol
        d = d / 2
        if s == sign(f(x))
            x = x + d
        else
            x = x - d
        end
        push!(xsave, x)
    end
    
    return x, xsave
end

# Example usage
f(x) = x^3
a, b = -6.0, 12.0
x_root, iterations = bisect(f, a, b)
println("Root found at x = $x_root")
# Visualize the function and iterations
x_plot = range(-4, 4, length=100)
p = plot(x_plot, f.(x_plot), label="f(x) = x³", legend=:topleft)
plot!(p, x_plot, zeros(length(x_plot)), label="y = 0")
scatter!(p, iterations, f.(iterations), label="Iterations")
display(p)

#=
 Function iteration
start with a first guess and update the rule , we give the computer the function i want to find the root  
=#
function fixpoint(g, x0; tol=1e-4, max_iter=100)
    x = x0
    x_history = [x]
    error = Inf
    iter = 0
    
    while error > tol && iter < max_iter
        x_new = g(x)
        error = abs(x_new - x)
        x = x_new
        push!(x_history, x)
        iter += 1
    end
    
    return x, x_history
end

# Example: Fixed point iteration for g(x) = √x
g(x) = sqrt(x)

# Try from below fixed point
x_fp1, hist1 = fixpoint(g, 0.1)
println("Fixed point from below: $x_fp1")

# Try from above fixed point
x_fp2, hist2 = fixpoint(g, 1.8)
println("Fixed point from above: $x_fp2")

#=
Newton's Method
we can give the computer the derivative of the function instead, then the root of the linerize function is our best gift 
=#
    # Define Newton's Method
function newton_method(f, df, x0; tol=1e-6, max_iter=100)
    x = x0
    for i in 1:max_iter
        fx = f(x)
        dfx = df(x)
        
        if abs(dfx) < 1e-10
            println("Derivative too small; stopping iteration.")
            return x
        end
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < tol
            println("Converged after $i iterations.")
            return x_new
        end
        
        x = x_new
    end
    println("Did not converge within $max_iter iterations.")
    return x
end
# Example Function: f(x) = x^2 - 2 (Root: √2)
f(x) = x^2 - 2
df(x) = 2x
# Initial Guess
x0 = 1.0
# Run Newton's Method
root = newton_method(f, df, x0)
println("Approximate root: $root")
println("f(root) = $(f(root))")  # Verify the root

#=
Quasi-newton Methods
Secent Method
Approximates derivative using finite differences:
=#
function secant(f, x0, x1; tol=1e-4, max_iter=20)
    x_prev = x0
    x = x1
    x_history = [x_prev, x]
    error = Inf
    iter = 0
    while error > tol && iter < max_iter
        x_new = x - f(x) * (x - x_prev)/(f(x) - f(x_prev))
        error = abs(x_new - x)
        x_prev = x
        x = x_new
        push!(x_history, x)
        iter += 1
    end
    return x, x_history
end
# Example using secant method
x_root, hist = secant(f, 0.1, 0.2)
println("Root found using secant method: $x_root")
# Visualize secant method
x_plot = range(0.1, 0.8, length=100)
p = plot(x_plot, f.(x_plot), label="f(x)", legend=:topleft)
plot!(p, x_plot, zeros(length(x_plot)), label="y = 0")
scatter!(p, hist, f.(hist), label="Secant iterations")
display(p)

#=
Quasi-newton Methods
Broyden's Method
This is a multidimensional version of the Secent method.
=#
using DifferentialEquations
#using NLsolve
# Define the nonlinear system
function f!(F, x, p)
    F[1] = x[1]^2 + x[2]^2 - 1
    F[2] = x[1] - x[2]
end
# Initial guess
x0 = [2.0, 1.0]
# Set up problem
prob = NonlinearProblem(f!, x0, nothing)
# Solve using Newton-Raphson
sol_newton = solve(prob, NewtonRaphson(), abstol=1e-8)
# Solve using Broyden
sol_broyden = solve(prob, Broyden(), abstol=1e-8)
println("Newton-Raphson solution: ", sol_newton.u)
#println("Newton-Raphson iterations: ", sol_newton.nlsolver.iterations)
println("\nBroyden solution: ", sol_newton.u)
#println("Broyden iterations: ", sol_broyden.nlsolver.iterations)