using LogDensityProblems, LogDensityProblemsAD, SimpleUnPack, Random, TransformVariables, TransformedLogDensities, SpecialFunctions
import Distributions
include("structs.jl");

function rate_to_prob(rate)
    return 1.0 .- exp.(-rate)
end

#Prior Functions
function calculate_lprior(distribution::Beta_distribution, value)
    @unpack α₁, α₂ = distribution;
    return ((α₁ - 1) * log(value)) + ((α₂ - 1) * log(1 - value))
end

function calculate_lprior(distribution::Gamma_distribution, value)
    @unpack α, β = distribution;
    return (α - 1) * log(value) - (β * value)
end

function calculate_lprior(distribution::Null_distribution, value)
    return 0.0 #does nothing
end

#likelihoods (to be defined on a model/use basis)
function calculate_likelihood(distribution::Null_likelihood, model_output)
    return 0.0 #does nothing
end

#log density
function ld_binomial(n, p, value)
    return ld_binomial_inner(n, log.(p), log.(1 .- p), value)
end

function ld_binomial_inner(n, lp, lp_inv, value)
    return loggamma.(n .+ 1) .- loggamma.(value .+ 1) .- loggamma.(n .- value .+ 1) .+
        (value .* lp) .+ ((n .- value) .* lp_inv)
end

function ld_transitions(transition, compartment, prob, N)
    #binomial that works with continuous values
    compartment_size = compartment .* N;
    transition_size = transition .* compartment_size;
    log_prob = log.(prob);
    log_prob_inv = log.(1 .- prob);
    #edge cases, when transition_size is close to 0 and prob is close to 0 we set log(prob) = 0 to avoid mulitplying inf
    log_prob[(transition_size .< 0.1) .& isinf.(log_prob)] .= 0.0;
    return sum(ld_binomial_inner(compartment_size, log_prob, log_prob_inv, transition_size))
end

function ld_transitions(transition, compartment, prob::Real, N)
    #binomial that works with continuous values
    compartment_size = compartment .* N;
    transition_size = transition .* compartment_size;
    return sum(ld_binomial(compartment_size, prob, transition_size))
end

#random sampling
function sample_transitions(compartment, prob, N, rng)
    whole_transitions = rand.(rng, Distributions.Binomial.(round.(compartment .* N), prob))
    proportional = whole_transitions ./ N;
    proportional[proportional .> compartment] .= compartment[proportional .> compartment];
    return proportional
end