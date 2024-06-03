S <- 900
I <- 10
beta <- 0.01

samples <- 1000000

prop_binom <- rbinom(samples, S, 1 - exp(-beta * I))/S

mean(prop_binom)
var(prop_binom)

1 - exp(-beta * I) #analytical mean
(exp(-beta * I) * (1 - exp(-beta * I)))/S #analytical variance

alpha_1 <- (S - 1) * (1 - exp(-beta * I))
alpha_2 <- (S - 1) * exp(-beta * I)

alpha_1/(alpha_1 + alpha_2)
(alpha_1 * alpha_2)/((alpha_1 + alpha_2)^2 * (alpha_1 + alpha_2 + 1))

prop_beta <- rbeta(samples, alpha_1, alpha_2)

mean(prop_beta)
var(prop_beta)
