#' Bayesian dyadic regression of co-occurrence
#'
#' Regression of co-occurrence affinity on one or more pairwise explanatory variables. For
#' each pair of species `i > j`, with prevalences `m[i]`, `m[j]` and `k[i, j]`
#' co-occurrences at `N` sites,
#'
#' ```
#' k[i, j] ~ FNCH(m[i], m[j], N, alpha + lambda[i] + lambda[j] + sum_p beta[p] * X_p[i, j])
#' alpha ~ Normal(0, alpha_sd), beta[p] ~ Normal(0, beta_sd), lambda[i] ~ Normal(0, lambda_sd)
#' ```
#'
#' where FNCH is Fisher's noncentral hypergeometric distribution with log odds ratio given
#' by the linear predictor. The species effects `lambda` account for the non-independence
#' of pairs that share a species.
#'
#' With `g`, only pairs of species from the same group enter the likelihood. By default each
#' group `q` has its own coefficients `beta_mu[p] + beta_sigma[p] * beta[p, q]`, with
#' `beta[p, q] ~ Normal(0, 1)`, `beta_mu[p] ~ Normal(0, mu_sd)` and
#' `beta_sigma[p] ~ Gamma(shape = sigma_params[1], scale = sigma_params[2])`; with
#' `pooled = TRUE` the groups share one set of coefficients `beta[p]`.
#'
#' With `strata`, co-occurrences are counted within each type of site; the types share the
#' coefficients and each has its own intercept `alpha[l]` and species effects
#' `lambda[i, l]`.
#'
#' Sampling is by NUTS in Julia (Turing.jl). The chains are run one after another.
#'
#' Arguments left `NULL` take the defaults of the Julia package, given in brackets.
#'
#' @inheritParams cooccurrence_counts
#' @param X A species by species matrix of an explanatory variable (only the lower triangle
#'   is used), or a list of such matrices.
#' @param g Optional group of each species (column of `Y`), such as genus or guild.
#' @param n_iter Number of draws per chain after adaptation.
#' @param n_chains Number of chains.
#' @param alpha_sd,beta_sd,lambda_sd Prior standard deviations (4, 1, 1).
#' @param mu_sd,sigma_params Hyperpriors for the group coefficients (1; 2, 0.1). Only with `g`.
#' @param pooled If `TRUE`, groups share one set of coefficients (`FALSE`). Only with `g`.
#' @param group_range If `TRUE`, co-occurrence within a group is judged only at the sites
#'   where at least one of its species occurs (`FALSE`). Only with `g`.
#' @param strata Optional type of each site (row of `Y`). Not with `g`.
#' @param site_weights `TRUE` for the site-weighted likelihood, or a numeric vector of known
#'   site effects (`FALSE`). See [pairwise_affinity()]. Not with `g` or `strata`.
#' @param metric `"affinity"` or `"jaccard"`.
#' @param n_adapts,delta Number of adaptation steps and target acceptance rate (100, 0.65).
#' @param adtype Automatic differentiation backend, `"forwarddiff"` or `"mooncake"`. By
#'   default ForwardDiff for at most 20 parameters, Mooncake otherwise.
#' @param drop_lambda If `TRUE`, the species effects are left out of the result.
#' @param seed Optional seed for the random number generator.
#' @return A [posterior::draws_df] with variables `alpha`, `beta[p]` (in the order of `X`),
#'   `lambda[i]` (in the order of the columns of `Y`; by group then column when `g` is
#'   given) and, with `g`, `beta_mu[p]`, `beta_sigma[p]` and `beta[p, q]` (groups in
#'   sorted order).
#' @examples
#' \dontrun{
#' Y <- matrix(runif(30 * 12) < 0.5, 30, 12)
#' X <- standardise(trait_distance(rnorm(12)))
#' fit <- cooccurrence_regression(X, Y)
#' posterior::summarise_draws(fit)
#' }
#' @export
cooccurrence_regression <- function(X, Y, g = NULL, n_iter = 1000, n_chains = 4,
                                    alpha_sd = NULL, beta_sd = NULL, lambda_sd = NULL,
                                    mu_sd = NULL, sigma_params = NULL, pooled = NULL,
                                    group_range = NULL, strata = NULL, site_weights = NULL,
                                    metric = c("affinity", "jaccard"), n_adapts = NULL,
                                    delta = NULL, adtype = NULL, drop_lambda = FALSE,
                                    seed = NULL) {
  metric <- match.arg(metric)
  P <- presence(Y)
  S <- ncol(P)
  if (!is.list(X) || is.data.frame(X)) X <- list(X)
  X <- lapply(X, as.matrix)
  for (p in seq_along(X)) {
    if (!identical(dim(X[[p]]), c(S, S))) {
      stop(sprintf("explanatory variable %d is %d x %d; expected %d x %d to match the columns of Y",
                   p, nrow(X[[p]]), ncol(X[[p]]), S, S))
    }
  }
  X3 <- array(unlist(lapply(X, as.numeric)), dim = c(S, S, length(X)))

  if (is.null(g)) {
    if (!is.null(mu_sd) || !is.null(sigma_params) || !is.null(pooled) || !is.null(group_range)) {
      stop("mu_sd, sigma_params, pooled and group_range apply only when g is given")
    }
  } else {
    if (length(g) != S) stop("g must have one entry per species (column of Y)")
    if (!is.null(strata) || !is.null(site_weights)) stop("strata and site_weights cannot be used with g")
  }
  if (!is.null(strata) && length(strata) != nrow(P)) stop("strata must have one entry per site (row of Y)")
  if (!is.null(site_weights) && !is.logical(site_weights)) site_weights <- as.numeric(site_weights)

  kw <- drop_null(list(g = label(g), seed = seed, drop_lambda = drop_lambda,
                       alpha_sd = alpha_sd, beta_sd = beta_sd, lambda_sd = lambda_sd,
                       mu_sd = mu_sd, sigma_params = sigma_params, pooled = pooled,
                       group_range = group_range, strata = label(strata),
                       site_weights = site_weights, metric = metric, n_adapts = n_adapts,
                       delta = delta, adtype = adtype))
  r <- do.call(jl, c(list("regression", X3, P, n_iter, n_chains), kw))

  draws <- r[[2]]
  if (!is.matrix(draws)) draws <- matrix(draws, ncol = length(r[[1]]))
  colnames(draws) <- r[[1]]
  d <- as.data.frame(draws, check.names = FALSE)
  d$.chain <- as.integer(r[[3]])
  d$.iteration <- as.integer(r[[4]])
  if (r[[5]] > 0) warning(r[[5]], " divergent transitions")
  posterior::as_draws_df(d)
}
