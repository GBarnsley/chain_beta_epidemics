using DynamicHMC, DynamicHMC.Diagnostics, MCMCDiagnosticTools, CSV, DataFrames, Plots
include("../model/SIR.jl");

t_max = 100;
t_per_day = 100;

problem = SIRstruct(
    5000,
    t_max * t_per_day,
    Gamma_distribution(1.0, 1.0),
    Gamma_distribution(1.0, 1.0),
    Beta_distribution(1.0, 1.0)
)

simulations = simulate_model(problem, (β = 1/3/t_per_day, γ = 1/5/t_per_day, I₀ = 1/problem.N), 500);

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

