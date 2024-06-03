using LogDensityProblems, LogDensityProblemsAD, SimpleUnPack, Random, TransformVariables, TransformedLogDensities, SpecialFunctions
import Distributions
include("structs.jl");

#Prior Functions
function calculate_lprior(distribution::Beta_distribution, value)
    @unpack α₁, α₂ = distribution;
    return ((α₁ - 1) * log(value)) + ((α₂ - 1) * log(1 - value))
end

function calculate_lprior(distribution::Gamma_distribution, value)
    @unpack α, β = distribution;
    return (α - 1) * log(value) - (β * value)
end

#Beta transition functions
function beta_parameters(size, prob_exp)
    N_m = max.(size, 1.001) .- 1; #need to fix this?, just use another when size is close to 1
    α₁ = N_m .* (1 .- prob_exp);
    α₂ = N_m .* prob_exp;
     
    return (α₁, α₂)
end

function beta_parameters_alternative(size, prob)
    α₁ = size .* prob;
    α₂ = size - α₁;

    return (α₁, α₂)
end

#log density
function ld_beta(α₁, α₂, value) #could make this more efficient with the sums?
    return sum(((α₁ .- 1) .* log.(value)) .+ ((α₂ .- 1) .* log.(1 .- value)) .- logbeta.(α₁, α₂)) #do I need the Log-Beta?
end

function ld_transitions(transition, compartment, exponent_terms, N)
    α₁, α₂ = beta_parameters(compartment .* N, exp.(-exponent_terms))
    #α₁, α₂ = beta_parameters_alternative(compartment .* N, 1 .- exp.(-exponent_terms))

    return ld_beta(α₁, α₂, transition)
end

#random sampling

function sample_transitions(compartment, exponent_terms, N, rng)
    α₁, α₂ = beta_parameters(compartment .* N, exp.(-exponent_terms))
    #α₁, α₂ = beta_parameters_alternative(compartment .* N, 1 .- exp.(-exponent_terms))

    return rand.(rng, Distributions.Beta.(α₁, α₂))
end
