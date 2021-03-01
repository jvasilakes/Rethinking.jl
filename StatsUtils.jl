export PI, HDPI, extract_samples

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

# Extract posterior parameter samples from a model estimated by Turing
# Julia implementation of R extract.samples function at 
# https://github.com/rmcelreath/rethinking/blob/3b48ec8dfda4840b9dce096d0cb9406589ef7923/R/map-quap-class.r#L89
function extract_samples(model; N=10_000)
    means = coef(model).array
    covmat = Float32.(vcov(model).array)
    mvnorm = MvNormal(means, covmat)
    post = DataFrame(rand(mvnorm, N)')
    rename!(post, names(model.values)[1])
    return post
end


function link_lm(model, data; N=1_000)
    μ_samples = []
    x̄ = mean(data)
    for weight in data
        post = extract_samples(model; N=N)
        μ_at_datum = post.α + post.β * (weight - x̄)
        push!(μ_samples, μ_at_datum)
    end
    return hcat(μ_samples...)
end
