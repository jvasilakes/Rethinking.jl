using CSV
using Optim
using Plots
using Turing
using StatsBase
using DataFrames
using PlotThemes
using StatsPlots
using Distributions
include("StatsUtils.jl")  # PI, HPDI, extract_samples, link_lm

# Set plot theme
theme(:dark)


# R Code 4.1
function R4_1()
    unif = Uniform(-1, 1)
    pos = [sum(rand(unif, 16)) for _ in 1:1000]
    display(plot(pos; seriestype=:density))
end

# R Code 4.{2,3}
function R4_2_3()
    unif = Uniform(0.0, 0.1)
    growth = [prod(1 .+ rand(unif, 12)) for _ in 1:10000]
    display(plot(growth; seriestype=:density))
end

# R Code 4.4
function R4_4()
    big_unif = Uniform(0.0, 0.5)
    big = [prod(1 .+ rand(big_unif, 12)) for _ in 1:10000]
    small_unif = Uniform(0.0, 0.01)
    small = [prod(1 .+ rand(small_unif, 12)) for _ in 1:10000]
    big_plt = plot(big; seriestype=:density, title="big")
    small_plt = plot(small; seriestype=:density, title="small")
    display(plot(big_plt, small_plt, layout=2))
end

# R Code 4.5
function R4_5()
    big_unif = Uniform(0.0, 0.5)
    log_big = [log(prod(1 .+ rand(big_unif, 12))) for _ in 1:10000]
    display(plot(log_big; seriestype=:density))
end

# Skipped 4.6 because it's just a rehash of code from Ch. 3

# R Code 4.{7,8,9,10,11}
function R4_7()
    # 4.7
    d = DataFrame(CSV.File("data/Howell1.csv"))
    # 4.{8,9}
    display(d)
    # 4.10
    display(d.height)
    # 4.11
    d2 = d[d.age .≥ 18, :]
    println(size(d2))  # Should have 352 rows.
end

# R Code 4.{12,13}
function R4_12_13()
    norm = Normal(178, 20)
    μ = pdf.(Ref(norm), range(100, 250, length=1000))
    μ_plt = plot(range(100, 250, length=1000), μ, title="mean", legend=false)
    unif = Uniform(0, 50)
    σ = pdf.(Ref(unif), range(-10, 60, length=1000))
    σ_plt = plot(range(-10, 16, length=1000), σ, title="variance", legend=false)
    display(plot(μ_plt, σ_plt))
end

# R Code 4.14
function R4_14()
    @model heights() = begin
        μ ~ Normal(178, 20)
        σ ~ Uniform(0, 50)
        h ~ Normal(μ, σ)
    end
    chain = sample(heights(), SMC(), 1000)
    prior_hs = chain[:h].data
    density(prior_hs, legend=false)
end

# R Code 4.15
function R4_15()
    @model heights() = begin
        μ ~ Normal(178, 100)
        σ ~ Uniform(0, 50)
        h ~ Normal(μ, σ)
    end
    chain = sample(heights(), SMC(), 1000)
    prior_hs = chain[:h].data
    density(prior_hs, legend=false)
end

# R Code 4.{16,17,18}
function R4_16_18()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]

    μs = range(150, 160, length=100)
    σs = range(7, 9, length=100)
    post = hcat([[μ, σ] for μ=μs, σ=σs]...)
    post = DataFrame(post', [:μ, :σ])
    lls = Float64[]
    for row in 1:size(post, 1)
        μ = post.μ[row]
        σ = post.σ[row]
        norm = Normal(μ, σ)
        ll = sum(logpdf.(Ref(norm), df2.height))
        push!(lls, ll)
    end
    dμ = Normal(178, 20)
    dσ = Uniform(0, 50)
    prod = @. lls + logpdf(dμ, post.μ) + logpdf(dσ, post.σ)
    prob = exp.(prod .- maximum(prod))
    post[!, :LL] = lls
    post[!, :prod] = prod
    post[!, :prob] = prob
    display(plot3d(post.μ, post.σ, post.prob, alpha=0.5, legend=false))
    return post
end


# R Code 4.{19,20,21,22}
# post parameter the output of R4_16_18()
function R4_19_22(post)
    sample_rows = sample(1:size(post, 1), Weights(post.prob), 10000)
    sample_μ = post.μ[sample_rows]
    sample_σ = post.σ[sample_rows]
    display(histogram2d(sample_μ, sample_σ))
    print("[ENTER] to show next plot")
    readline()
    display(density(sample_μ, label="μ"))
    print("[ENTER] to show next plot")
    readline()
    display(density(sample_σ, label="σ"))
    println(HPDI(sample_μ, 0.975))
    println(HPDI(sample_σ, 0.975))
end


# R Code 4.{23,24,25}
function R4_23_25()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    df3 = sample(df2.height, 20)

    μs = range(150, 170, length=200)
    σs = range(4, 20, length=200)
    post = hcat([[μ, σ] for μ=μs, σ=σs]...)
    post = DataFrame(post', [:μ, :σ])
    lls = Float64[]
    for row in 1:size(post, 1)
        μ = post.μ[row]
        σ = post.σ[row]
        norm = Normal(μ, σ)
        ll = sum(logpdf.(Ref(norm), df3))
        push!(lls, ll)
    end
    dμ = Normal(178, 20)
    dσ = Uniform(0, 50)
    prod = @. lls + logpdf(dμ, post.μ) + logpdf(dσ, post.σ)
    prob = exp.(prod .- maximum(prod))
    post[!, :LL] = lls
    post[!, :prod] = prod
    post[!, :prob] = prob
    display(plot3d(post.μ, post.σ, post.prob, alpha=0.5, label="μ v. σ"))
    print("[ENTER] to show next plot")
    readline()

    sample_rows = sample(1:size(post, 1), Weights(post.prob), 10000)
    sample_μ = post.μ[sample_rows]
    sample_σ = post.σ[sample_rows]
    display(histogram2d(sample_μ, sample_σ, label="μ v. σ"))
    print("[ENTER] to show next plot")
    readline()
    display(density(sample_μ, label="μ"))
    print("[ENTER] to show next plot")
    readline()
    display(density(sample_σ, label="σ"))
end

# R code 4.{26,27,28,29}
function R4_26_29()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]

    @model heights(data) = begin
        μ ~ Normal(178, 20)
        σ ~ Uniform(0, 50)
        for n in 1:length(data)
            data[n] ~ Normal(μ, σ)
        end
    end
    model = heights(df2.height)
    m4_1 = optimize(model, MAP(), BFGS())
    display(vcov(m4_1))
    return model, m4_1
end

# Skipping R Code 4.30 because it is specific to quadratic approximation


# R Code 4.{31,32,33}
function R4_31_33()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]

    @model heights(data) = begin
        μ ~ Normal(178, 0.1)
        σ ~ Uniform(0, 50)
        for n in 1:length(data)
            data[n] ~ Normal(μ, σ)
        end
    end
    model = heights(df2.height)
    m4_2 = optimize(model, MAP(), BFGS())
    display(vcov(m4_2))
    return model, m4_2
end


# R Code 4.{34,35}
# model, estimate: model and estimation returned by R4_26_29() or R4_31_33()
function R4_34_35(model, estimate)
    chain = sample(model, SMC(), 10_000, init_theta=estimate.values.array)
    df = DataFrame((μ=chain[:μ], σ=chain[:σ]))
    display(describe(df, :mean, :std))
    return df
end

# Skipping R Code 4.36


# R Code 4.37
function R4_37()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    display(scatter(df2.height, df2.weight))
end


# R Code 4.{38,39}
function R4_38_39()
    Random.seed!(2971)
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]

    N = 100
    α_dist = Normal(178, 20)
    β_dist = Normal(0, 10)
    αs = rand(α_dist, N)
    βs = rand(β_dist, N)
    x̄ = mean(df2.weight)
    xfrom = minimum(df2.weight)
    xto = maximum(df2.weight)
    plt = plot(title="β ~ N(0, 10)", xlabel="weight",
               ylabel="height", leg=false)
    for i in 1:N
        f(x) = αs[i] + βs[i] * (x - x̄)
        plot!(f, xfrom, xto, color="grey", alpha=0.2)
    end
    hline!([272], color="white")
    hline!([0], color="white", line=(1, :dash))
    display(plt)
end

# R Code 4.40
function R4_40()
    d = LogNormal(0, 1)
    display(plot(d))
end

# R Code 4.41
function R4_41()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]

    N = 100
    α_dist = Normal(178, 20)
    β_dist = LogNormal(0, 1)
    αs = rand(α_dist, N)
    βs = rand(β_dist, N)
    x̄ = mean(df2.weight)
    xfrom = minimum(df2.weight)
    xto = maximum(df2.weight)
    plt = plot(title="β ~ N(0, 10)", xlabel="weight",
               ylabel="height", leg=false)
    for i in 1:N
        f(x) = αs[i] + βs[i] * (x - x̄)
        plot!(f, xfrom, xto, color="grey", alpha=0.2)
    end
    hline!([272], color="white")
    hline!([0], color="white", line=(1, :dash))
    display(plt)
end


# R Code 4.42
function R4_42()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    x̄ = mean(df2.weight)

    @model regression(weights, heights) = begin
        α ~ Normal(178, 20)
        β ~ LogNormal(0, 1)
        σ ~ Uniform(0, 1)
        for n in 1:length(weights)
            μ = α + β * (weights[n] - x̄)
            heights[n] ~ Normal(μ, σ)
        end
    end
    model = regression(df2.weight, df2.height)
    m4_3 = optimize(model, MAP(), BFGS())
    return m4_3
end


# R Code 4.43
function R4_43()
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]

    @model regression(weights, heights) = begin
        α ~ Normal(178, 20)
        logβ ~ Normal(0, 1)
        σ ~ Uniform(0, 1)
        for n in 1:length(weights)
            μ = α + exp(logβ) * (weights[n] - x̄)
            heights[n] ~ Normal(μ, σ)
        end
    end
    model = regression(df2.weight, df2.height)
    m4_3b = optimize(model, MAP(), BFGS())
    # NB. exp(m4_3b[:logβ]) == m4_3[:β]
    return m4_3b
end


# Skipping R Code 4.44 because StatsBase.coeftable is broken as of 18/2/2021

# R Code 4.45
# m4_3 is the output from R4_42 or R4_43
function R4_45(m4_3)
    display(round(vcov(m4_3); digits=5))
end


# R Code 4.46
# m4_3 is the output from R4_42 or R4_43
function R4_46(m4_3)
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    scatter(df2.weight, df2.height, leg=false)
    post = extract_samples(m4_3; N=10_000)
    α_map = mean(post.α)
    β_map = mean(post.β)
    x̄ = mean(df2.weight)
    xfrom = minimum(df2.weight)
    xto = maximum(df2.weight)
    f(x) = α_map + β_map * (x - x̄)
    plot!(f, xfrom, xto, leg=false, color="white",
          xlabel="Weight", ylabel="Height")
end

# Skipping R Code 4.47 as it is a trivial slice of post

# R Code 4.{48,49}
function R4_48_49(N)
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    dfN = df2[1:N, :]
    x̄ = mean(dfN.weight)

    # Define the regression model
    @model regression(weights, heights) = begin
        α ~ Normal(178, 20)
        β ~ LogNormal(0, 1)
        σ ~ Uniform(0, 50)
        for n in 1:length(weights)
            μ = α + β * (weights[n] - x̄)
            heights[n] ~ Normal(μ, σ)
        end
    end
    modelN = regression(dfN.weight, dfN.height)
    mN = optimize(modelN, MAP(), BFGS())
    # Sample
    post = extract_samples(mN; N=10_000)

    plt = scatter(dfN.weight, dfN.height, leg=false)
    for i ∈ 1:N
        f_i(x) = post.α[i] + post.β[i] * (x - x̄)
        xfrom = minimum(dfN.weight)
        xto = maximum(dfN.weight)
        plot!(f_i, xfrom, xto, color="grey", alpha=0.3, leg=false) 
    end
    display(plt)
end


# R Code 4.{50,51,52}
# m4_3: optimised model returned by R4_42 or R4_43
function R4_50_52(m4_3)
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    x̄ = mean(df2.weight)
    post = extract_samples(m4_3; N=10_000)
    μ_at_50 = post.α + post.β * (50 - x̄)
    display(density(μ_at_50, xlabel="μ|weight=50", leg=false))
    println(PI(μ_at_50, 0.89))
end


# R Code 4.{53,54,55}
# m4_3: optimised model returned by R4_42 or R4_43
function R4_53_55(m4_3)
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    μ = link_lm(m4_3, df2.weight)
    println(μ[1:3], " ... ", μ[end-3:end])
    weight_seq = range(25, 70; step=1)
    μ = link_lm(m4_3, weight_seq)
    println(μ[1:3], " ... ", μ[end-3:end])
    plt = scatter(df2.weight, df2.height, leg=false, alpha=0.1)
    for i ∈ 1:length(weight_seq)
        x = repeat([weight_seq[i]], size(μ, 1))
        scatter!(x, μ[:,i], alpha=0.1, color="red")
    end
    display(plt)
    return μ
end

# R Code 4.{56,57}
# μ: return value of R4_53_55()
function R4_56_57(μ)
    df = DataFrame(CSV.File("data/Howell1.csv"))
    df2 = df[df.age .≥ 18, :]
    weight_seq = range(25, 70; step=1)
    μ_mean = mapslices(mean, μ; dims=1)[1,:]
    μ_PI = mapslices((xs) -> PI(xs, 0.89), μ; dims=1)
    plt = scatter(df2.weight, df2.height, alpha=0.25, leg=false)
    lower_bound = abs.(μ_PI[1,:] - μ_mean)
    upper_bound = abs.(μ_PI[2,:] - μ_mean)
    plot!(weight_seq, μ_mean, leg=false, ribbon=(lower_bound, upper_bound))
    display(plt)
end
