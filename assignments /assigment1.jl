#=      
Broyden Method - Paola Villa
Write an algorithm in your language of choice to solve my Broyden example (not using a canned solver from a package in your language). 
note: I am going to use only matrices and vectors.
=#
using LinearAlgebra
using Plots

# system of nonlinear equations (from class)
function f(x)
    return [x[1]^2 + x[2]^2 - 1, x[1] - x[2]]
end

# Broyden's Method Implementation
function broyden_method(f, x0; max_iter=100, tol=1e-8)
    # Initial guess and initial Jacobian approximation
    # Identity matrix as initial Jacobian
    x = x0
    B = Matrix{Float64}(I, length(x0), length(x0))  
    # To store the iteration points for plotting
    points = [x]
    for k in 1:max_iter
        # Step 1: Compute the function value at the current guess
        fx = f(x)
        
        # Step 2: Solve for the direction using the current Jacobian approximation
        d = -inv(B) * fx        # or d = -B \ fx which can be more efficient  
        
        # Step 3: Update the guess
        x_new = x + d
        
        # Step 4: Compute the changes in x and f(x)
        dx = x_new - x
        df = f(x_new) - fx
        
        # Step 5: Update the Jacobian approximation using Broyden's update formula
        B += (df - B * dx) * dx' / (dx' * dx)
        
        # Step 6: Convergence check
        if norm(f(x_new)) < tol
            println("Converged after ", k, " iterations.")
            break
        end
        # Update the guess for the next iteration
        x = x_new
        push!(points, x)  # Store the new point
    end
    return points, x
end

# Initial guess
x0 = [2.0, 1.0]

# Solve using Broyden's method
points, final_solution = broyden_method(f, x0)
println("Final Solution: ", final_solution)

# Plotting the iteration points
x_vals_points = [p[1] for p in points]
y_vals_points = [p[2] for p in points]

# Plot the iterated points
scatter(x_vals_points, y_vals_points, label="Iterations", marker=:o, color=:green, markersize=6)

title!("Broyden's Method: Iteration Points")


#=
Converged after 14 iterations.
Final Solution: [0.7071130442108156, 0.7071130442106373]
=#