using DynamicHMC, DynamicHMC.Diagnostics, MCMCDiagnosticTools, CSV, DataFrames, Plots
include("../model/SIR.jl");

t_max = 100;
t_per_day = 100;
total_steps = t_max * t_per_day;
N = 5000;
true_β = 1/2/t_per_day;
true_γ = 1/5/t_per_day;
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
    @unpack postive_samples = distribution;
    positivity = model_output.I[2:end];
    return sum(
        (postive_samples .* log.(positivity)) .+ ((100 .- postive_samples) .* log.(1 .- positivity))
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

#quick diagnostics 
results_stacked = stack_posterior_matrices(results);

ess_rhat(results_stacked[:, :, 1:3]) #only care about the actual parameters

map(x -> summarize_tree_statistics(x.tree_statistics), results)

map(x -> EBFMI(x.tree_statistics), results)

#transform variables
results_pooled = pool_posterior_matrices(results);
results_pooled = TransformVariables.transform.(variable_transform, eachcol(results_pooled));
chn = reduce(hcat, map(x -> collect(x), results_pooled));

