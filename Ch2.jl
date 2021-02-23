using Plots
using StatsPlots
using Distributions
using Turing
using Optim


# R Code 2.1
function R2_1()
    ways = [0, 3, 8, 9, 0]
    ways ./ sum(ways)
end


# R Code 2.2
function R2_2()
    binom = Binomial(9, 0.5)
    pdf(binom, 6)
end



# Used for R Code 2.3-2.
# grid_size: number of values to test
# s: successes
# f: failures
function likelihood(grid_size, s, f)
    # Define the grid of values to test
    p_grid = collect(range(0, 1, length=grid_size))
    # Define the likelihood functions for each value in p_grid
    ds = Binomial.(Ref(s+f), p_grid)
    # Compute the likelihood of 6/9 successes for each value in p_grid
    likelihoods = pdf.(ds, Ref(s))
    return likelihoods, p_grid
end
    
# R Code 2.{3,4}
# Grid approximation with uniform prior for 6 successes and 3 failures.
function R2_3_4()
    likelihoods, p_grid = likelihood(20, 6, 3)
    # Define the uniform prior
    prior = repeat([1], length(p_grid))
    # Compute unstandardized posteriors
    posterior_unstd = likelihoods .* prior
    # Standardize the posteriors
    posterior = posterior_unstd / sum(posterior_unstd)
    # Plot the posterior
    plot(p_grid, posterior, xlabel="p water", ylabel="posterior",
         title="Uniform Prior")
end

# R Code 2.5L1
# Grid approximation with step prior
function R2_5L1()
    likelihoods, p_grid = likelihood(20, 6, 3)
    prior = p_grid .> 0.5
    posterior_unstd = likelihoods .* prior
    posterior = posterior_unstd / sum(posterior_unstd)
    plot(p_grid, posterior, xlabel="p water", ylabel="posterior",
         title="Step Prior")
end

# R Code 2.5L2
# Grid approximation with peaked prior
function R2_5L2()
    likelihoods, p_grid = likelihood(20, 6, 3)
    prior = @. exp(-5*abs(p_grid - 0.5))
    posterior_unstd = likelihoods .* prior
    posterior = posterior_unstd / sum(posterior_unstd)
    plot(p_grid, posterior, xlabel="p water", ylabel="posterior",
         title="Peaked Prior")
end


# R Code 2.6
# Since Turing doesn't have a quadratic optimizer, we'll just
# use the maximum likelihood estimator (MLE) here, since the 
# "Rethinking" text block on p. 44 says they are often equal.
# I don't know how to get any more summary statistics than the
# mean, so that's all I report here.
function R2_6()
    W = 6
    L = 3
    
    @model globe(s, f) = begin
        N = s + f
        p ~ Uniform()
        s ~ Binomial(N, p)
    end
    
    model = globe(W, L)
    mle_estimate = optimize(model, MLE())
    display(mle_estimate)
end

# R Code 2.7
function R2_7()
    W = 6
    L = 3
    beta = Beta(W+1, L+1)
    norm = Normal(0.67, 0.16)
    plot(0:0.05:1, pdf.(Ref(beta), collect(0:0.05:1));
         label="Analytic")
    plot!(0:0.05:1, pdf.(Ref(norm), collect(0:0.05:1));
          color=:red, label="MLE")
end


# R Code 2.{8,9}
function R2_8_9()
    n_samples = 1000
    p = [0.0 for _ in 1:n_samples]
    p[1] = 0.5
    W = 6
    L = 3
    for i in 2:n_samples
        norm = Normal(p[i-1], 0.1)
        p_new = rand(norm)
        if p_new < 0
            p_new = abs(p_new)
        end
        if p_new > 1
            p_new = 2 - p_new
        end
        binom0 = Binomial(W+L, p[i-1])
        q0 = pdf(binom0, W)
        binom1 = Binomial(W+L, p_new)
        q1 = pdf(binom1, W)
        if Random.rand() < q1/q0
            p[i] = p_new
        else
            p[i] = p[i-1]
        end
    end
    beta = Beta(W+1, L+1)
    plot(0:0.05:1, pdf.(Ref(beta), collect(0:0.05:1)); label="Analytic")
    StatsPlots.density!(p; xlim=(0,1), label="MCMC")
end

# Alternative to R Code 2.8
# using Turing.jl and Hamiltonian Monte Carlo sampler.
function R2_8_2()
    W = 6
    L = 3
    
    @model globe(s, f) = begin
        N = s + f
        p ~ Uniform()
        s ~ Binomial(N, p)
    end
    
    iterations = 1000
    ϵ = 0.05
    τ = 10
    chain = sample(globe(W, L), HMC(ϵ, τ), iterations, progress=true)
    display(chain)
    p_summary = chain[:p]
    beta = Beta(W+1, L+1)
    plot(0:0.05:1, pdf.(Ref(beta), collect(0:0.05:1)); label="Analytic")
    plot!(p_summary, seriestype=:density, color=:red, label="HMC")
end
