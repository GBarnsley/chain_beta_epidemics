using DynamicHMC, DynamicHMC.Diagnostics, MCMCDiagnosticTools, LogDensityProblems, LogDensityProblemsAD, SimpleUnPack, Random, CSV, DataFrames, TransformVariables, TransformedLogDensities, SpecialFunctions
include("../model/chain_beta.jl");

problem = SIRstruct(1000, 20)

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

