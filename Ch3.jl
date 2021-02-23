using Plots
using StatsBase
using StatsPlots
using Distributions
include("Intervals.jl")  # PI, HDPI


# R Code 3.1
function R3_1()
    Pr_positive_vamp = 0.95
    Pr_positive_mortal = 0.01
    Pr_vamp = 0.001
    Pr_positive = Pr_positive_vamp * Pr_vamp + Pr_positive_mortal * (1 - Pr_vamp)
    Pr_vamp_positive = (Pr_positive_vamp * Pr_vamp) / Pr_positive
end

# R Code 3.{2,3,4,5,6,7,8,9,10}
function R3_2_10()
    p_grid = collect(range(0, 1, length=1000))
    prob_p = [1.0 for _ in p_grid]
    binoms = Binomial.(Ref(9), p_grid)
    prob_data = pdf.(binoms, Ref(6))
    posteriors = prob_data .* prob_p
    posteriors = posteriors ./ sum(posteriors)

    # R Code 3.3
    samples = sample(p_grid, Weights(posteriors), 10000; replace=true)
    # R Code 3.{4,5}
    points_plt = plot(samples, seriestype=:scatter, legend=false)
    density_plt = density(samples, legend=false)
    display(plot(points_plt, density_plt))
    # R Code 3.6
    println(sum(posteriors[p_grid .< 0.5]))
    # R Code 3.7
    println(sum(samples .< 0.5) / 10000)
    # R Code 3.8
    println(sum(0.5 .< samples .< 0.75) / 10000)
    # R Code 3.9
    println(quantile(samples, 0.8))
    # R Code 3.10
    println(quantile(samples, (0.1, 0.9)))
end


# Percentile Interval
# Julia implementation of R function at
# https://github.com/rmcelreath/rethinking/blob/3b48ec8dfda4840b9dce096d0cb9406589ef7923/R/utilities.r#L132
function PI(samples, p)
    a = @. (1 - p) / 2
    x = quantile.(Ref(samples), (a, 1 .- a))
    x = transpose(hcat(x...))
    n = length(p)
    result = [0.0 for _ in 1:n*2]
    for i in 1:n
        low_idx = n + 1 - i
        up_idx = n + i
        result[low_idx] = x[1,i]
        result[up_idx] = x[2,i]
        a = (1 - p[i]) / 2
    end
    return result
end

# Highest Posterior Density Interval
# Modification of code at https://gist.github.com/ahwillia/59af6ad67021a4b91fd5
# to use samples from the posterior instead of the Distribution itself.
function HPDI(samples, p)
    total_area = 0.0  # Running sum of area
    cred_x = (Float64)[]

    areas = samples ./ sum(samples)
    sp = sortperm(areas)
    samples = samples[sp]
    areas = areas[sp]

    i = length(samples)
    while total_area < p
        total_area += areas[i]
        push!(cred_x, samples[i])
        i -= 1
    end

    cred_x = sort(cred_x)
    return (cred_x[1], cred_x[end])
end

# R Code 3.{11,12,13}
function R3_11_13()
    p_grid = collect(range(0, 1, length=1000))
    priors = [1.0 for _ in p_grid]
    binoms = Binomial.(Ref(3), p_grid)
    likelihoods = pdf.(binoms, Ref(3))
    posteriors = likelihoods .* priors
    posteriors = posteriors ./ sum(posteriors)
    samples = sample(p_grid, Weights(posteriors), 10000; replace=true)
    pred_x = PI(samples, 0.5)
    cred_x = HPDI(samples, 0.5)
    println(pred_x)
    println(cred_x)
    plot(samples, seriestype=:density, xlim=(0,1), label="samples")
    vline!([cred_x...], label="PI")
    display(vline!([pred_x...], label="HDPI"))
end


# R Code 3.{14,15,16}
# Point estimates
function R3_14_16()
    p_grid = collect(range(0, 1, length=1000))
    priors = [1.0 for _ in p_grid]
    binoms = Binomial.(Ref(3), p_grid)
    likelihoods = pdf.(binoms, Ref(3))
    posteriors = likelihoods .* priors
    posteriors = posteriors ./ sum(posteriors)
    samples = sample(p_grid, Weights(posteriors), 10000; replace=true)
    # R Code 3.14
    println(p_grid[argmax(posteriors)])
    # R Code 3.15
    println(mode(samples))
    # R Code 3.16
    println(mean(samples))
    println(median(samples))
end

# R Code 3.{17,18,19}
# Loss
function R3_17_19()
    p_grid = collect(range(0, 1, length=1000))
    priors = [1.0 for _ in p_grid]
    binoms = Binomial.(Ref(3), p_grid)
    likelihoods = pdf.(binoms, Ref(3))
    posteriors = likelihoods .* priors
    posteriors = posteriors ./ sum(posteriors)
    samples = sample(p_grid, Weights(posteriors), 10000; replace=true)
    # R Code 3.17
    println(sum(posteriors .* abs.(Ref(0.5) .- p_grid)))
    # R Code 3.18
    losses = [sum(posteriors .* abs.(Ref(p) .- p_grid))
              for p in p_grid]
    # R Code 3.19
    println(p_grid[argmin(losses)])
end

# R Code 3.{20,21,22,23}
function R3_20_23()
    # R Code 3.20
    binom = Binomial(2, 0.7)
    println(pdf.(Ref(binom), 0:2))
    # R Code 3.21
    println(rand(binom))
    # R Code 3.22
    println(rand(binom, 10))
    # R Code 3.23
    dummy_w = rand(binom, 100000)
    levels = sort(unique(dummy_w))
    freqs = counts(dummy_w) ./ 1e5
    println(collect(zip(levels, freqs)))
end

# R Code 3.24
function R3_24()
    binom = Binomial(9, 0.7)
    dummy_w = rand(binom, 100000)
    plot(dummy_w, seriestype=:histogram, xlabel="dummy water count",
         legend=false)
end

# R Code 3.{25,26}
function R3_25_26()
    # R Code 3.25
    binom = Binomial(9, 0.6)
    w = rand(binom, 10000)
    # Get samples
    p_grid = collect(range(0, 1, length=1000))
    priors = [1.0 for _ in p_grid]
    binoms = Binomial.(Ref(9), p_grid)
    likelihoods = pdf.(binoms, Ref(6))
    posteriors = likelihoods .* priors
    posteriors = posteriors ./ sum(posteriors)
    samples = sample(p_grid, Weights(posteriors), 10000; replace=true)
    # R Code 3.26
    binoms = Binomial.(Ref(9), samples)
    w_unc = rand.(binoms)
    plot(w, seriestype=:histogram, label="p=0.6",
         xlabel="num water samples", legend=true)
    plot!(w_unc, seriestype=:histogram, label="p~Binom", alpha=0.5)
end
