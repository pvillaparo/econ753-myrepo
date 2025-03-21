#=       
            ECON753 - assignment 3
            Paola Villa Paro
=#
cd("/Users/villap/Documents/GitHub/econ753-myrepo/assignments/a3")
#--- packages 
using Printf, Random
using Optim  # We'll use the Optim.jl package for maximizing the likelihood


#############################################################################
#                   PART A
#       Estimate(θ) using Rust’s NFP approach
##############################################################################

# -------------------------------------------------------
# 3. Value Function Solver via Contraction Mapping
# -------------------------------------------------------
"""
Solves for V[1..5] for the bus-engine problem with T1EV errors.
"""
function solve_value_function(mu, R, beta; gamma=0.5772, maxiter=1000, tol=1e-7)
    # Initialize guesses for V(a)
    V = fill(0.0, 5)  # we'll store V[1],...,V[5]
    for iter in 1:maxiter
        Vnew = similar(V)
        for a in 1:5
            # Choice-specific deterministic parts:
            # V0(a): no replace => mu*a + beta*V(min(a+1,5))
            # V1(a): replace    => R + beta*V(1)
            V0 = mu*a + beta*V[min(a+1,5)]
            V1 = R     + beta*V[1]
            # Log-sum formula:
            Vnew[a] = gamma + log(exp(V0) + exp(V1))
        end
        # Check convergence
        diff = maximum(abs.(Vnew .- V))
        V .= Vnew
        if diff < tol
           #@printf("Value function converged at iter=%d (diff=%.2e)\n", iter, diff)
            break
        end
        if iter == maxiter
            #@warn "Value iteration: did not fully converge."
        end
    end
    return V
end


# -------------------------------------------------------
# 4. Simulate Data
# -------------------------------------------------------
"""
Generates T observations (a_t, i_t) from the policy implied by V(a)
"""
function simulate_data(V, mu, R, beta, T; gamma=0.5772)
    a_data = Vector{Int}(undef, T)
    i_data = Vector{Int}(undef, T)
    # Probability of replacement given state a:
    function p_replace(a)
        V0 = mu*a + beta*V[min(a+1,5)]
        V1 = R     + beta*V[1]
        return exp(V1)/(exp(V0) + exp(V1))
    end
    # Start from a=1
    a = 1
    for t in 1:T
        a_data[t] = a
        # Draw uniform(0,1)
        u = rand()
        if u < p_replace(a)
            i_data[t] = 1
            a = 1
        else
            i_data[t] = 0
            a = min(a+1,5)
        end
    end
    return a_data, i_data
end

# -------------------------------------------------------
# 5. Use the data from (4) to estimate θ using Rust’s NFP approach
# -------------------------------------------------------

#------------------Probability that i_t is chosen

"""
Sums the log of these probabilities => log-likelihood.
"""
function log_likelihood(mu, R, beta, a_data, i_data; gamma=0.5772)
    # Solve for V with the current guess
    V = solve_value_function(mu, R, beta; gamma=gamma)
    loglike = 0.0
    for t in eachindex(a_data)
        a = a_data[t]
        i = i_data[t]
        # choice-specific values
        V0 = mu*a + beta*V[min(a+1,5)]
        V1 = R     + beta*V[1]
        # Probability i=1 => exp(V1)/[exp(V0)+exp(V1)]
        # Probability i=0 => exp(V0)/[exp(V0)+exp(V1)]
        denom = log(exp(V0) + exp(V1))
        if i == 1
            # i=1 => log( p_replace(a) ) = V1 - log( exp(V0) + exp(V1) )
            loglike += (V1 - denom)
        else
            # i=0 => no replace => log(1 - p_replace(a)) = V0 - log(exp(V0)+exp(V1))
            loglike += (V0 - denom)
        end
    end
    return loglike
end

#-----------Estimate (mu, R) via NFP + MLE
"""
Maximize log_likelihood w.r.t. mu, R. We keep beta fixed.
"""
function estimate_parameters(a_data, i_data, beta)
    function obj(x)
        mu_guess = x[1]
        R_guess  = x[2]
        # negative log-likelihood (since we want to *minimize* for the solver)
        return -log_likelihood(mu_guess, R_guess, beta, a_data, i_data)
    end
    # initial guess
    x0 = [-2.0, -4.0]
    res = optimize(obj, x0, BFGS())
    return res
end


# -------------------------------------------------------
# Run Functions
# -------------------------------------------------------
Random.seed!(609)
    # 1) True parameters for DGP
    mu_true   = -1.0
    R_true    = -3.0
    beta_true = 0.9
    T         = 20000

    println("Solving the model at true (mu,R)=($mu_true,$R_true)")
    Vtrue = solve_value_function(mu_true, R_true, beta_true)
    println("Value function at true params:")
    for a in 1:5
        @printf("  V(%d)=%.3f\n", a, Vtrue[a])
    end

    # 2) Simulate data
    println("\nSimulating T=$T data...")
    a_data, i_data = simulate_data(Vtrue, mu_true, R_true, beta_true, T)
    println("First 10 observations:")
    for t in 1:10
        println((a_data[t], i_data[t]))
    end

    # 3) Estimate (mu, R)
    println("\nEstimating (mu,R) via Nested Fixed Point MLE...")
    res = estimate_parameters(a_data, i_data, beta_true)
    mu_hat = Optim.minimizer(res)[1]
    R_hat  = Optim.minimizer(res)[2]
    finalLL = -Optim.minimum(res)  # final log-likelihood

    println("\nRESULTS:")
    println("-----------------------------------------")
    @printf("  mu_hat = %.3f  (true=%.3f)\n", mu_hat, mu_true)
    @printf("  R_hat  = %.3f  (true=%.3f)\n", R_hat,  R_true)
    @printf("  LogLik = %.3f\n", finalLL)
    println("-----------------------------------------")


#############################################################################
#                   PART B
#Estimate(θ) using Hotz and Miller’s CCP approach with forward simulation
##############################################################################

# -------------------------------------------------------
# 6a. Estimate the replacement probabilities
# -------------------------------------------------------
"""
Compute P_hat(1|a) = fraction of times i=1 among all t with a_t=a for a in {1,2,3,4,5}. 
"""
function compute_ccp(a_data::Vector{Int}, i_data::Vector{Int})
    counts_state = fill(0,5)    # n(a)
    counts_rep   = fill(0,5)    # n(a,1)
    for (a,i) in zip(a_data, i_data)
        counts_state[a] += 1
        counts_rep[a]   += (i==1 ? 1 : 0)
    end
    ccp = similar(counts_state, Float64)
    for a in 1:5
        if counts_state[a]>0
            ccp[a] = counts_rep[a]/counts_state[a]
        else
            ccp[a] = 0.0  # or handle unobserved states differently
        end
    end
    return ccp
end
# -------------------------------------------------------
# 6b Forward Simulation
# -------------------------------------------------------

#------Construct Transition Matrices F0 , F1
function make_F0_F1()
    F0 = fill(0.0, 5,5)
    F1 = fill(0.0, 5,5)
    for a in 1:5
        # No replace => row a => col min(a+1,5) = 1
        F0[a,min(a+1,5)] = 1.0
        # Replace => row a => col=1 => 1
        F1[a,1] = 1.0
    end
    return F0, F1
end
#------Unconditional Transition Matrix T
function make_uncond_transition(ccp::Vector{Float64}, F0, F1)
    T = fill(0.0,5,5)
    for a in 1:5
        for j in 1:5
            T[a,j] = (1.0-ccp[a])*F0[a,j] + ccp[a]*F1[a,j]
        end
    end
    return T
end

#----- Forward Simulation to Compute (Vbar0, Vbar1)
"""
We do a direct fixed-point iteration here, not random Monte Carlo. 
"""
function forward_value_functions(ccp::Vector{Float64}, mu::Float64, R::Float64, beta::Float64; maxiter=1000, tol=1e-7)
    F0, F1 = make_F0_F1()
    # unconditional mixture T
    Tmat = make_uncond_transition(ccp, F0, F1)
    Vbar = fill(0.0,5)
    for iter in 1:maxiter
        newVbar = similar(Vbar)
        for a in 1:5
            # expected immediate payoff:
            # E[ payoff ] = ccp[a]*R + (1-ccp[a])*(mu*a)
            immediate = (1.0-ccp[a])*(mu*a) + ccp[a]*R
            # discounted future:
            # sum_j T(a,j)* Vbar(j)
            disc = 0.0
            for j in 1:5
                disc += Tmat[a,j]*Vbar[j]
            end
            newVbar[a] = immediate + beta*disc
        end
        diff = maximum(abs.(newVbar .- Vbar))
        Vbar .= newVbar
        if diff<tol
            break
        end
    end

    # Now define the choice-specific ones:
    F0bar = fill(0.0,5)
    F1bar = fill(0.0,5)
    F0, F1 = make_F0_F1()
    for a in 1:5
        # no replace payoff
        pay0 = mu*a
        # future:
        future0 = 0.0
        for j in 1:5
            future0 += F0[a,j]*Vbar[j]
        end
        F0bar[a] = pay0 + beta*future0
        # replace payoff
        pay1 = R
        future1 = 0.0
        for j in 1:5
            future1 += F1[a,j]*Vbar[j]
        end
        F1bar[a] = pay1 + beta*future1
    end

    return F0bar, F1bar
end
# -------------------------------------------------------
# 6c Final Steps for Estimation
# -------------------------------------------------------
#----------- Log-likelihood & Estimation
"""
compute log-likelihood using a second-stage approach:
p_hat(1|a) = exp(Vbar1(a)) / [exp(Vbar0(a)) + exp(Vbar1(a))]
Then accumulate log-likelihood.
"""
function ccp_log_likelihood(a_data, i_data, mu, R, beta, ccp)
    # 1) get forward-simulation choice-specific values:
    Vbar0, Vbar1 = forward_value_functions(ccp, mu, R, beta)
    # 2) compute p_model(1|a)
    #    p_model(1|a) = exp(Vbar1[a]) / (exp(Vbar0[a]) + exp(Vbar1[a]))
    # Then sum log-likelihood
    loglike = 0.0
    for t in eachindex(a_data)
        a = a_data[t]
        i = i_data[t]
        denom = log(exp(Vbar0[a]) + exp(Vbar1[a]))
        if i==1
            loglike += (Vbar1[a] - denom)
        else
            loglike += (Vbar0[a] - denom)
        end
    end
    return loglike
end

"""
Maximize the log-likelihood w.r.t. (mu,R) using BFGS.
"""
function ccp_mle(a_data, i_data, beta, ccp)
    function objective(x)
        mu_guess = x[1]
        R_guess  = x[2]
        return -ccp_log_likelihood(a_data, i_data, mu_guess, R_guess, beta, ccp)
    end
    x0 = [-2.0, -4.0]   # initial guess
    res = optimize(objective, x0, BFGS())
    return res
end


# -------------------------------------------------------
# Run Functions
# -------------------------------------------------------
Random.seed!(609)
mu_true   = -1.0
R_true    = -3.0
beta_true = 0.9
T         = 20000

# True prameters and data simultion 
Vtrue = solve_value_function(mu_true, R_true, beta_true)
a_data, i_data = simulate_data(Vtrue, mu_true, R_true, beta_true, T)

# 2) Nonparametric CCP from data
ccp = compute_ccp(a_data, i_data)
println("Nonparametric CCP estimates from data:")
for a in 1:5
    @printf("  P_hat(1|a=%d) = %.3f\n", a, ccp[a])
end

# 3) Estimate (mu, R) using Hotz–Miller approach:
println("\nEstimating (mu,R) via CCP approach + forward simulation.")
res = ccp_mle(a_data, i_data, beta_true, ccp)
mu_hat, R_hat = Optim.minimizer(res)
finalLL = -Optim.minimum(res)

println("\nRESULTS:")
println("-----------------------------------")
@printf("  mu_hat = %.3f  (true=%.3f)\n", mu_hat, mu_true)
@printf("  R_hat  = %.3f  (true=%.3f)\n", R_hat,  R_true)
@printf("  LogLik = %.3f\n", finalLL)
println("-----------------------------------")