#' Co-occurrence counts
#'
#' @param Y Presence/absence matrix or data frame, sites in rows and species in columns.
#'   Any value greater than zero counts as a presence.
#' @param occupied_only If `TRUE`, sites at which none of the species was found are left out.
#' @return A list: `K`, the species by species matrix of co-occurrence counts; `m`, the
#'   number of sites at which each species was found; `N`, the number of sites.
#' @export
cooccurrence_counts <- function(Y, occupied_only = FALSE) {
  P <- presence(Y)
  r <- jl("counts", P, occupied_only = occupied_only)
  K <- r[[1]]
  dimnames(K) <- list(colnames(P), colnames(P))
  m <- r[[2]]
  names(m) <- colnames(P)
  list(K = K, m = m, N = r[[3]])
}

#' Affinity of a pair of species
#'
#' Estimates of the affinity (log odds ratio) of two species that occupy `m_i` and `m_j`
#' of `N` sites and co-occur at `k` of them. The arguments are recycled, so several pairs
#' can be estimated in one call.
#'
#' `affinity_mle()` is the maximum likelihood estimate, which is infinite when `k` is the
#' largest or smallest value possible. `affinity_map()` is the maximum a posteriori estimate
#' under the prior `alpha ~ Normal(0, prior_sd)`. `affinity_posterior()` summarises the
#' posterior under the same prior, integrated numerically (no sampling).
#'
#' @param k Number of sites at which both species occur.
#' @param m_i,m_j Number of sites at which each species occurs.
#' @param N Number of sites.
#' @param prior_sd Prior standard deviation of the affinity.
#' @param level Probability of the central credible interval.
#' @param npoints Number of grid points for the numerical integration.
#' @return `affinity_mle()` and `affinity_map()` return a numeric vector.
#'   `affinity_posterior()` returns a data frame with columns `median`, `mean`, `lower`,
#'   `upper`, `pd` (probability of direction: posterior probability that the affinity has
#'   the sign of its median) and `p_positive` (posterior probability that it is positive).
#' @export
affinity_mle <- function(k, m_i, m_j, N) {
  jl("mle", k, m_i, m_j, N)
}

#' @rdname affinity_mle
#' @export
affinity_map <- function(k, m_i, m_j, N, prior_sd = 3) {
  jl("map_estimate", k, m_i, m_j, N, prior_sd = prior_sd)
}

#' @rdname affinity_mle
#' @export
affinity_posterior <- function(k, m_i, m_j, N, prior_sd = 4, level = 0.95, npoints = 1001) {
  r <- jl("posterior", k, m_i, m_j, N, prior_sd = prior_sd, level = level, npoints = npoints)
  names(r) <- c("median", "mean", "lower", "upper", "pd", "p_positive")
  as.data.frame(r)
}

#' Affinity of every pair of species
#'
#' Posterior summaries of the affinity of every pair of species, each pair analysed on its
#' own with the prior `alpha ~ Normal(0, prior_sd)`.
#'
#' @inheritParams cooccurrence_counts
#' @inheritParams affinity_mle
#' @param site_weights `FALSE`, `TRUE`, or a numeric vector of site effects (one per row of
#'   `Y`). With `TRUE`, sites are weighted by their effects in a logistic model with species
#'   and site effects, estimated with each pair left out, so that differences in richness
#'   between sites do not appear as associations.
#' @param metric `"affinity"` or `"jaccard"` (summaries of the logit of the Jaccard index).
#' @return A list of species by species matrices: `median`, `lower`, `upper`, `pd` and
#'   `p_positive` (see [affinity_posterior()]). Entries are `NaN` on the diagonal and for
#'   species found at none or all of the sites.
#' @export
pairwise_affinity <- function(Y, prior_sd = 4, level = 0.95, site_weights = FALSE,
                              metric = c("affinity", "jaccard")) {
  metric <- match.arg(metric)
  P <- presence(Y)
  if (!is.logical(site_weights)) site_weights <- as.numeric(site_weights)
  r <- jl("pairwise", P, prior_sd = prior_sd, level = level, site_weights = site_weights,
          metric = metric)
  names(r) <- c("median", "lower", "upper", "pd", "p_positive")
  lapply(r, function(x) {
    dimnames(x) <- list(colnames(P), colnames(P))
    x
  })
}

#' Fisher's noncentral hypergeometric distribution on the log odds scale
#'
#' `fnch_logpdf()` is the log probability of `k` co-occurrences of two species that occupy
#' `m_i` and `m_j` of `N` sites, given affinity `alpha`. `fnch_mean()` is the expected
#' number of co-occurrences. Arguments are recycled.
#'
#' @inheritParams affinity_mle
#' @param alpha Affinity (log odds ratio).
#' @return A numeric vector.
#' @export
fnch_logpdf <- function(k, m_i, m_j, N, alpha) {
  jl("fnch_logpdf", k, m_i, m_j, N, alpha)
}

#' @rdname fnch_logpdf
#' @export
fnch_mean <- function(m_i, m_j, N, alpha) {
  jl("fnch_mean", m_i, m_j, N, alpha)
}
