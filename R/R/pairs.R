with_names <- function(D, x) {
  dimnames(D) <- list(names(x), names(x))
  D
}

#' Pairwise trait distances
#'
#' `trait_distance()` gives `abs(x[i] - x[j])` for a numeric trait, and 0 where two species
#' share a value and 1 where they differ for any other trait (character, factor, logical).
#' `gower_distance()` is Gower's distance over one or more traits: numeric traits are scaled
#' by their range and the result is averaged over traits.
#'
#' @param x A trait, one value per species.
#' @param ... Traits, one value per species each, or a single data frame of traits.
#' @return A species by species matrix.
#' @export
trait_distance <- function(x) {
  with_names(jl("trait_distance", trait(x)), x)
}

#' @rdname trait_distance
#' @export
gower_distance <- function(...) {
  traits <- list(...)
  if (length(traits) == 1 && is.list(traits[[1]])) traits <- as.list(traits[[1]])
  D <- do.call(jl, c(list("gower_distance"), unname(lapply(traits, trait))))
  dimnames(D) <- NULL
  D
}

#' Standardise a matrix of pairwise values
#'
#' Centres and scales the off-diagonal elements of `X` to mean 0 and standard deviation 1,
#' and sets the diagonal to 0. With `g`, this is done within each group, and elements
#' between groups are set to 0. The default priors of [cooccurrence_regression()] assume
#' standardised explanatory variables.
#'
#' @param X A square matrix.
#' @param g Optional group of each species.
#' @return A matrix of the same size.
#' @export
standardise <- function(X, g = NULL) {
  X <- as.matrix(X)
  Z <- do.call(jl, c(list("standardise", X), drop_null(list(g = label(g)))))
  dimnames(Z) <- dimnames(X)
  Z
}

#' Random matrix with the same row and column totals
#'
#' A presence/absence matrix with the same number of species at every site and the same
#' prevalence for every species as `Y`, but no associations beyond those the totals impose
#' (curveball algorithm, Strona et al. 2014). Fit an analysis to such matrices to check that
#' it finds few associations.
#'
#' @inheritParams cooccurrence_counts
#' @param n_trades Number of trades; by default ten times the larger dimension of `Y`.
#' @param seed Optional seed for the random number generator.
#' @return A logical matrix.
#' @export
swap_null <- function(Y, n_trades = NULL, seed = NULL) {
  P <- presence(Y)
  Z <- do.call(jl, c(list("swap_null", P), drop_null(list(n_trades = n_trades, seed = seed))))
  dimnames(Z) <- dimnames(P)
  Z
}

#' Site effects
#'
#' Fits `logit P(Y[n, s] = 1) = a[s] + b[n]`, a logistic model with species and site
#' effects, with `Normal(0, prior_sd)` priors on all effects. Sites and species that carry
#' no information (present nowhere or everywhere) are removed first.
#'
#' @inheritParams cooccurrence_counts
#' @param prior_sd Prior standard deviation of the effects.
#' @return A list: `sites` and `species`, the rows and columns of `Y` kept; `b` and `a`,
#'   their effects; `D`, the curvature of the log posterior for each site effect.
#' @export
site_effects <- function(Y, prior_sd = 10) {
  r <- jl("site_effects", presence(Y), prior_sd = prior_sd)
  names(r) <- c("sites", "species", "b", "a", "D")
  r
}
