#=
Tests for paired_interference.jl

Remarks
-------


=#

"""
    Distribution for the binary vector ``[D_A(10), D_A(11), D_B(01), D_B(11)]``
        parametrised by (2^4)-vector `p` of joint probabilitites.

##### Fields
- `p`::Dict{String, Float64} : vector of joint probabilities satisfying
`p`["0000"] = ℙ(D_A(10)=0, D_A(11)=0, D_B(01)=0, D_B(11)=0)
`p`["1000"] = ℙ(D_A(10)=1, D_A(11)=0, D_B(01)=0, D_B(11)=0)
⋮
`p`["1111"] = ℙ(D_A(10)=1, D_A(11)=1, D_B(01)=1, D_B(11)=1).
"""
struct 𝓓
    p::Dict{String, Float64}
    function 𝓓(p::Dict{String, Float64})
        if all(0 .<= values(p) .<= 1) && length(p)==2^4
            sum(values(p))==1 ? new(p) : error("Probabilities in `p` must sum to 1")
        else
            error("Probabilites in `p` must be in [0,1]; length(`p`)=2^4 required")
        end
    end
end

"""
    Evaluates the probability mass function of distribution `dist` at point `d`.
"""
function pmf(dist::𝓓, d::Array{<:Real, 1})
    (length(d) != 4) && error("length(`d`)=4 required")
    dist.p[join(i for i in d)]
end

"""
    rand(dist::𝓓)

Randomly samples `n` 4-long vectors from ``[D_A(10), D_A(11), D_B(01), D_B(11)]``,
based on distribution `dist::𝓓`.
"""
function Base.rand(dist::𝓓, n::Int)
    p = dist.p
    # combinations for possible values of the 4-long vector
    combinations = collect(Iterators.product(0:1, 0:1, 0:1, 0:1))[:]
    combinations_string = [join(i for i in v) for v in combinations]
    # correspondingly ordered probabilities
    prob = [p[v] for v in combinations_string]
    # sample indices of combinations
    d = DiscreteNonParametric(1:length(combinations), prob)
    idx = rand(d, n)
    # transform indices to values
    d_sample = zeros(Int, n, 4)
    for i in 1:n
        d_sample[i, :] .= collect(combinations[idx[i]])
    end
    return d_sample
end


function _gen_d(p::Dict{String, Float64}, n::Int)
    d_a00 = d_a01 = d_b00 = d_b10 = zeros(n)
    d_sample = rand(𝓓(p), n)
    d_a10 = d_sample[:, 1]
    d_a11 = d_sample[:, 2]
    d_b01 = d_sample[:, 3]
    d_b11 = d_sample[:, 4]
    return (d_a00, d_a10, d_a01, d_a11, d_b00, d_b10, d_b01, d_b11)
end


function _compute_θ(μy::Function, p::Dict{String,<:Real})
    pmfp(x::Array{<:Real, 1}) = pmf(𝓓(p), x)
    all_combinations = collect(Iterators.product(0:1, 0:1, 0:1, 0:1))[:] .|> collect
    # θ_1 = 𝔼[Y(00)]
    combinations = all_combinations
    θ_1 = sum(pmfp(d)==0 ? 0 : μy(d...)[1] * pmfp(d) for d in combinations)
    # θ_2 = 𝔼[Y(10)-Y(00) | D_A(10)=1, D_A(11)=1]
    combinations = collect(Iterators.product(1, 1, 0:1, 0:1))[:] .|> collect
    denom = sum(pmfp(d) for d in combinations)  # ℙ(D_A(10)=D_A(11)=1)
    θ_2 = sum(pmfp(d)==0 ? 0 : (μy(d...)[2]-μy(d...)[1]) * pmfp(d)/denom for d in combinations)
    # θ_3 = 𝔼[Y(01)-Y(00) | D_B(01)=1, D_B(11)=1]
    combinations = collect(Iterators.product(0:1, 0:1, 1, 1))[:] .|> collect
    denom = sum(pmfp(d) for d in combinations)  # ℙ(D_A(10)=D_A(11)=1)
    θ_3 = sum(pmfp(d)==0 ? 0 : (μy(d...)[3]-μy(d...)[1]) * pmfp(d)/denom for d in combinations)
    # θ_4 = spillover-like effect
    θ_4 = Inf
    return [θ_1, θ_2, θ_3, θ_4]
end



@testset "late_estimator" begin
    # ---- DGP parameters
    n = 1000  # number of pairs in sample
    # -- 𝓓
    p = Dict{String, Float64}()  #p.m.f values for [D_A(10), D_A(11), D_B(01), D_B(11)]
    # unnormalised values
    p["0000"] = 1
    p["1000"] = 0  # by Monotonicity
    p["0100"] = 1
    p["0010"] = 0  # by Monotonicity
    p["0001"] = 1
    p["1100"] = 3
    p["0110"] = 0  # by Monotonicity
    p["0011"] = 4
    p["1001"] = 0  # by Monotonicity
    p["0101"] = 2
    p["1010"] = 0  # by Monotonicity
    p["0111"] = 2
    p["1011"] = 0  # by Monotonicity
    p["1101"] = 2
    p["1110"] = 0  # by Monotonicity
    p["1111"] = 1  # > 0 required
    # normalise p
    mass = sum(values(p))
    for (key, value) in p; p[key] = p[key] / mass; end

    # -- 𝓨
    # distribution of [Y(00), Y(10), Y(01), Y(11)] | [D_A(10), D_A(11), D_B(01), D_B(11)]
    # conditional mean vector
    function μy(a10::T, a11::T, b01::T, b11::T) where {T<:Real}
        mean_table = Dict{String, Array{<:Real, 1}}()
        mean_table["0000"] = [0, 0, 0, 0]
        mean_table["0100"] = [0, 2.0, 0, 0]
        mean_table["0001"] = [0, 0, 2.0, 0]
        mean_table["1100"] = [0, 2.0, 0, 5.0]
        mean_table["0011"] = [0, 0, 2.0, 5.0]
        mean_table["0101"] = [0, 3.0, 0, 6.7]
        mean_table["0111"] = [0, 0, 3.1, 4.3]
        mean_table["1101"] = [0, 0.4, 0, 2.0]
        mean_table["1111"] = [0, 9.0, 9.0, 9.0]
        return mean_table[join(i for i in (a10, a11, b01, b11))]
    end
    # covar matrix
    m = randn(4,4)
    Σ = m' * m
    # ---- monte carlo
    mc_reps = 999
    latehats_sum = zeros(4)
    for i in 1:mc_reps
        # -- generate data
        # treatment assignments
        z_a = [rand() <= 0.5 for _ in 1:n]
        z_b = [rand() <= 0.5 for _ in 1:n]
        # potential treatment participations
        (d_a00, d_a10, d_a01, d_a11, d_b00, d_b10, d_b01, d_b11) = _gen_d(p, n)
        # observed treatment participations
        d_a = @. z_a*z_b*d_a11 + z_a*(1-z_b)*d_a10 + (1-z_a)*z_b*d_a01 + (1-z_a)*(1-z_b)*d_a00
        d_b = @. z_a*z_b*d_b11 + z_a*(1-z_b)*d_b10 + (1-z_a)*z_b*d_b01 + (1-z_a)*(1-z_b)*d_b00
        # potential outcomes
        y_sample = zeros(n, 4)
        for i in 1:n
            μ = μy(d_a10[i], d_a11[i], d_b01[i], d_b11[i])
            y_sample[i,:] .= rand(MvNormal(μ, Σ))
        end
        y00 = y_sample[:, 1]
        y10 = y_sample[:, 2]
        y01 = y_sample[:, 3]
        y11 = y_sample[:, 4]
        # observed outcome
        y = @. d_a*d_b*y11 + d_a*(1-d_b)*y10 + (1-d_a)*d_b*y01 + (1-d_a)*(1-d_b)*y00
        # -- estimation
        pim = PairedInterferenceModel(y, d_a, d_b, z_a, z_b)
        latehats_sum += late_estimator(pim)
    end
    # ---- compute population expected values
    θ = _compute_θ(μy, p)
    # ---- compare
    @test all(isapprox.(θ[1:3], latehats_sum[1:3]/mc_reps, atol=1e-1))
end
