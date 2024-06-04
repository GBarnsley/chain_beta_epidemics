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

#Beta transition functions
function beta_parameters(size, prob)
    N_m = max.(size, 1.001) .- 1; #need to fix this?, just use another when size is close to 1
    α₁ = N_m .* prob;
    α₂ = N_m .* (1 .- prob);
     
    return (α₁, α₂)
end

function beta_parameters_alternative(size, prob)
    α₁ = size .* prob;
    α₂ = size .- α₁;

    return (α₁, α₂)
end

#log density
function ld_beta(α₁, α₂, value) #could make this more efficient with the sums?
    return sum(((α₁ .- 1) .* log.(value)) .+ ((α₂ .- 1) .* log.(1 .- value)) .- logbeta.(α₁, α₂)) #do I need the Log-Beta?
end

function ld_transitions(transition, compartment, prob, N)
    #has issues when prob = 1 or 0
    α₁, α₂ = beta_parameters(compartment .* N, prob)
    #α₁, α₂ = beta_parameters_alternative(compartment .* N, prob)

    return ld_beta(α₁, α₂, transition)
end

#random sampling
function sample_transitions_unsafe(compartment, prob, N, rng)

    α₁, α₂ = beta_parameters(compartment .* N, prob)
    #α₁, α₂ = beta_parameters_alternative(compartment .* N, prob)

    return rand.(rng, Distributions.Beta.(α₁, α₂))
end

function index_prob(prob, index)
    return prob
end

function index_prob(prob::AbstractArray, index)
    return prob[index]
end

function sample_transitions(compartment, prob, N, rng)
    output = Array{eltype(compartment)}(undef, size(compartment, 1));
    #need to catch when exponent_terms or compartment is 0 and just set output to 0
    non_zero_prob_or_compartment = (prob .> 0.0) .& (compartment .> 0.0);
    output[.!non_zero_prob_or_compartment] .= 0.0;

    output[non_zero_prob_or_compartment] .= sample_transitions_unsafe(
        compartment[non_zero_prob_or_compartment], index_prob(prob, non_zero_prob_or_compartment), N, rng
    );

    return output
end