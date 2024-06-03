using DynamicHMC, DynamicHMC.Diagnostics, LogDensityProblems, LogDensityProblemsAD, SimpleUnPack, Random, CSV, DataFrames, TransformVariables, TransformedLogDensities, SpecialFunctions
include("../model/chain_beta.jl");

problem = SIRstruct(1000, 20)

variable_transform = define_variable_transforms(problem);

transformed_model = TransformedLogDensity(variable_transform, problem);

model_gradient = ADgradient(:ForwardDiff, transformed_model);

