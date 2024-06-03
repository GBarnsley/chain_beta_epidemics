struct SIRstruct
    S::Vector{Real}
    I::Vector{Real}
    R::Vector{Real}
    N::Int
    t_steps::Int
    #add check that sum to 1
    SIRstruct(N, t_steps; type = Real) = new(
        Vector{type}(undef, t_steps), Vector{type}(undef, t_steps), Vector{type}(undef, t_steps), 
        N, t_steps
    )
end

function ld_beta_approx_bin(size, prob_exp, value)
    N_m = max.(size, 1.001) .- 1; #need to fix this?, just use another when size is close to 1
    α₁ = N_m .* (1 .- prob_exp);
    α₂ = N_m .* prob_exp;

    return sum(
        ld_beta(α₁, α₂, value)
    )
end

function ld_beta_alternative(size, prob, value)
    α₁ = size .* prob;
    α₂ = size - α₁;

    return sum(
        ld_beta(α₁, α₂, value)
    )
end

function ld_beta(α₁, α₂, value)
    return ((α₁ .- 1) .* log.(value)) .+ ((α₂ .- 1) .* log.(1 .- value)) .- logbeta.(α₁, α₂) #do I need the Log-Beta?
end

function ld_gamma(α, β, value)
    return (α .- 1) .* log.(value) .- (β .* value)# .- lgamma.(α) .+ (α .* log.(β))
end

function simulate_transition(compartment, transition, exponent_terms, N)
    #ld_beta_alternative(compartment .* N, 1 .- exp.(-exponent_terms), transition)
    ld_beta_approx_bin(compartment .* N, exp.(-exponent_terms), transition)
end

function (problem::SIRstruct)(θ)
    @unpack S, I, R, N, t_steps = problem;
    @unpack β, γ, I₀, δI, δR  = θ;

    ld = 0.0;
    #priors, none for now

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

    #likelihood of transitions
    ld += simulate_transition(S, δI, β .* I, N) +
        simulate_transition(R, δR, γ, N);

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
#LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#@benchmark LogDensityProblems.logdensity_and_gradient(model_gradient, pars) #119.211 μs ± 96.460 μs
#import ReverseDiff
#model_gradient = ADgradient(:ReverseDiff, transformed_model, compile = Val(false));
#LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#@benchmark LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#import Zygote
#model_gradient = ADgradient(:Zygote, transformed_model);
#@time LogDensityProblems.logdensity_and_gradient(model_gradient, pars)
#@benchmark LogDensityProblems.logdensity_and_gradient(model_gradient, pars)