abstract type distribution end;

abstract type real_postive_distribution <: distribution end;

struct Gamma_distribution <: real_postive_distribution
    α::Real
    β::Real
end

abstract type unit_interval_distribution <: distribution end;

struct Beta_distribution <: unit_interval_distribution
    α₁::Real
    α₂::Real
end

struct Null_distribution <: real_postive_distribution end

abstract type likelihood_struct end;

struct Null_likelihood <: likelihood_struct end