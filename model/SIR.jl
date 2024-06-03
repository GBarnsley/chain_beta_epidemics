include("../functions/functions.jl");

struct SIRstruct
    N::Int
    t_steps::Int
    β_prior::real_postive_distribution
    γ_prior::real_postive_distribution
    I₀_prior::unit_interval_distribution
    #add check that sum to 1
end

function simulate_model(problem::SIRstruct, θ, N_samples; rng = Random.default_rng())
    @unpack N, t_steps = problem;
    @unpack I₀, β, γ = θ;

    S = Array{eltype(I₀)}(undef, t_steps, N_samples);
    I = Array{eltype(I₀)}(undef, t_steps, N_samples);
    R = Array{eltype(I₀)}(undef, t_steps, N_samples);

    #initial conditions
    S[1, :] .= 1.0 - I₀;
    I[1, :] .= I₀;
    R[1, :] .= 0.0;

    #simulate transtitions
    for t in 1:(t_steps - 1)
        infections = S[t, :] .* sample_transitions(S[t, :], rate_to_prob(β .* I[t, :]), N, rng);
        recoveries = I[t, :] .* sample_transitions(I[t, :], rate_to_prob(γ), N, rng);
        S[t+1, :] = S[t, :] - infections;
        I[t+1, :] = I[t, :] + infections - recoveries;
        R[t+1, :] = R[t, :] + recoveries;
    end

    return (S = S, I = I, R = R)
end

function iterate_model(problem::SIRstruct, θ)
    @unpack N, t_steps = problem;
    @unpack I₀, δI, δR  = θ;

    S = Vector{eltype(I₀)}(undef, t_steps);
    I = Vector{eltype(I₀)}(undef, t_steps);
    R = Vector{eltype(I₀)}(undef, t_steps);

    #initial conditions
    S[1] = 1.0 - I₀;
    I[1] = I₀;
    R[1] = 0.0;

    #simulate transtitions
    for t in 1:(t_steps - 1)
        infections = δI[t] * S[t];
        recoveries = δR[t] * I[t];
        S[t+1] = S[t] - infections;
        I[t+1] = I[t] + infections - recoveries;
        R[t+1] = R[t] + recoveries;
    end

    return (S = S, I = I, R = R)
end

function (problem::SIRstruct)(θ)
    @unpack N, β_prior, γ_prior, I₀_prior = problem;
    @unpack β, γ, I₀, δI, δR  = θ;

    ld = 0.0;
    #priors, none for now
    ld += 
        calculate_lprior(β_prior, β) +
        calculate_lprior(γ_prior, γ) +
        calculate_lprior(I₀_prior, I₀)

    #run model
    @unpack S, I = iterate_model(problem, θ);
   
    #likelihood of transitions
    ld += ld_transitions(δI, S, β .* I, N) +
        ld_transitions(δR, I, γ, N);

    #likelihood in terms of data
    
    return ld
end

function define_variable_transforms(problem::SIRstruct)
    @unpack t_steps = problem

    #model parameters
    as_tuple_names = (:β, :γ, :I₀);
    as_tuple_values = (as_positive_real, as_positive_real, as_unit_interval);

    #augmentation parameters
    as_tuple_names = (as_tuple_names..., :δI);
    as_tuple_values = (as_tuple_values..., as(Vector, as_unit_interval, t_steps));

    as_tuple_names = (as_tuple_names..., :δR);
    as_tuple_values = (as_tuple_values..., as(Vector, as_unit_interval, t_steps));

    as(
        (; zip(as_tuple_names, as_tuple_values)...)
    )
end

#using Test, BenchmarkTools
#pars = rand(LogDensityProblems.dimension(transformed_model));
#θ = TransformVariables.transform(variable_transform, pars);
#model_gradient = ADgradient(:ForwardDiff, transformed_model);
#problem(θ)
#LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#@benchmark LogDensityProblems.logdensity_and_gradient(model_gradient, pars) #119.211 μs ± 96.460 μs #31 with no pre-allocation
#import ReverseDiff
#model_gradient = ADgradient(:ReverseDiff, transformed_model, compile = Val(false));
#LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#@benchmark LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#import Zygote
#model_gradient = ADgradient(:Zygote, transformed_model);
#@time LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#@benchmark LogDensityProblems.logdensity_and_gradient(model_gradient, pars)