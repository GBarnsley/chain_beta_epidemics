using DynamicHMC, DynamicHMC.Diagnostics, MCMCDiagnosticTools, CSV, DataFrames, Plots, JLD2
include("../model/SIR.jl");

t_max = 100;
t_per_day = 100;
total_steps = t_max;
N = 5000;
true_β = 1/2;
true_γ = 1/5;
true_I₀ = 1/N;
print("R₀:", true_β/true_γ)

#first generate some data, first we'll do many simulations
simulations = simulate_model(
    SIRstruct(
        N,
        total_steps
    ), 
    (β = true_β, γ = true_γ, I₀ = true_I₀),
    500
);

#now I is just daily postivity and we sample 100 times a day
import Distributions
postive_samples = rand.(Distributions.Binomial.(100, simulations.I[t_per_day:t_per_day:total_steps, :]))

#define likelihood
struct SIR_likelihood <: likelihood_struct
    postive_samples::Matrix{Int}
end

function calculate_likelihood(distribution::SIR_likelihood, model_output)
    positivity = model_output.I[2:end];
    return sum(
        (distribution.postive_samples .* log.(positivity)) .+ ((100 .- distribution.postive_samples) .* log.(1 .- positivity))
    )
end

#now define model with uniformative priors
problem = SIRstruct(
    N,
    t_max + 1,
    Null_distribution(),
    Null_distribution(),
    Beta_distribution(1, 1),
    SIR_likelihood(postive_samples)
);

variable_transform = define_variable_transforms(problem);

transformed_model = TransformedLogDensity(variable_transform, problem);

model_gradient = ADgradient(:ForwardDiff, transformed_model);

results = map(_ -> mcmc_with_warmup(Random.default_rng(), model_gradient, 1000; reporter = ProgressMeterReport()), 1:1);

@save "data/derived/results.jld2" results;

#quick diagnostics 
results_stacked = stack_posterior_matrices(results);

ess_rhat(results_stacked[:, :, 1:3]) #only care about the actual parameters

map(x -> summarize_tree_statistics(x.tree_statistics), results)

map(x -> EBFMI(x.tree_statistics), results)

#transform variables
results_pooled = pool_posterior_matrices(results);
results_pooled = TransformVariables.transform.(variable_transform, eachcol(results_pooled));
chn = reduce(hcat, map(x -> collect(x), results_pooled));
chn = chn[1:3, :];

import Statistics

print(true_β * t_per_day)
print(Statistics.quantile(chn[1, :], [0.025, 0.25, 0.5, 0.75, 0.975]))

print(true_γ * t_per_day)
print(Statistics.quantile(chn[2, :], [0.025, 0.25, 0.5, 0.75, 0.975]))

print(true_I₀)
print(Statistics.quantile(chn[3, :], [0.025, 0.25, 0.5, 0.75, 0.975]))

print(true_β/true_γ)
print(Statistics.quantile(chn[1, :] ./ chn[2, :], [0.025, 0.25, 0.5, 0.75, 0.975]))

#now with just one
postive_samples = postive_samples[:, findfirst(sum(postive_samples, dims = 1)[1, :] .> 0)];

#define likelihood
struct SIR_likelihood_single <: likelihood_struct
    postive_samples::Vector{Int}
end

function calculate_likelihood(distribution::SIR_likelihood_single, model_output)
    positivity = model_output.I[2:end];
    return sum(
        (distribution.postive_samples .* log.(positivity)) .+ ((100 .- distribution.postive_samples) .* log.(1 .- positivity))
    )
end

#now define model with uniformative priors
problem = SIRstruct(
    N,
    t_max + 1,
    Null_distribution(),
    Null_distribution(),
    Beta_distribution(1, 1),
    SIR_likelihood_single(postive_samples)
);

variable_transform = define_variable_transforms(problem);

transformed_model = TransformedLogDensity(variable_transform, problem);

model_gradient = ADgradient(:ForwardDiff, transformed_model);

results = map(_ -> mcmc_with_warmup(Random.default_rng(), model_gradient, 1000; reporter = ProgressMeterReport()), 1:1);

@save "data/derived/results.jld2" results;

#quick diagnostics 
results_stacked = stack_posterior_matrices(results);

ess_rhat(results_stacked[:, :, 1:3]) #only care about the actual parameters

map(x -> summarize_tree_statistics(x.tree_statistics), results)

map(x -> EBFMI(x.tree_statistics), results)

#transform variables
results_pooled = pool_posterior_matrices(results);
results_pooled = TransformVariables.transform.(variable_transform, eachcol(results_pooled));
chn = reduce(hcat, map(x -> collect(x), results_pooled));
chn = chn[1:3, :];

import Statistics

print(true_β * t_per_day)
print(Statistics.quantile(chn[1, :], [0.025, 0.25, 0.5, 0.75, 0.975]))

print(true_γ * t_per_day)
print(Statistics.quantile(chn[2, :], [0.025, 0.25, 0.5, 0.75, 0.975]))

print(true_I₀)
print(Statistics.quantile(chn[3, :], [0.025, 0.25, 0.5, 0.75, 0.975]))

print(true_β/true_γ)
print(Statistics.quantile(chn[1, :] ./ chn[2, :], [0.025, 0.25, 0.5, 0.75, 0.975]))
