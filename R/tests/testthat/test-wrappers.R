skip_on_cran()
skip_if(Sys.which("julia") == "" && Sys.getenv("JULIA_HOME") == "", "Julia not found")

set.seed(1)
Y <- matrix(runif(20 * 8) < 0.5, 20, 8, dimnames = list(NULL, paste0("sp", 1:8)))

test_that("counts and pair estimates", {
  cc <- cooccurrence_counts(Y)
  expect_equal(cc$N, 20)
  expect_equal(unname(cc$m), unname(colSums(Y)))
  expect_equal(cc$K["sp1", "sp2"], sum(Y[, 1] & Y[, 2]))
  expect_equal(fnch_mean(cc$m[1], cc$m[2], cc$N, affinity_mle(cc$K[1, 2], cc$m[1], cc$m[2], cc$N)),
               cc$K[1, 2], tolerance = 1e-8)
  expect_equal(affinity_mle(5, 5, 9, 12), Inf)
  expect_length(affinity_map(c(1, 2, 3), 5, 9, 12), 3)
  post <- affinity_posterior(3, 5, 9, 12)
  expect_named(post, c("median", "mean", "lower", "upper", "pd", "p_positive"))
  expect_true(post$lower < post$median && post$median < post$upper)
})

test_that("pairwise affinity", {
  pw <- pairwise_affinity(Y)
  expect_equal(dim(pw$median), c(8, 8))
  expect_equal(rownames(pw$median), colnames(Y))
  expect_true(isSymmetric(unname(pw$median)))
  expect_equal(pw$median[2, 1], affinity_posterior(sum(Y[, 1] & Y[, 2]), sum(Y[, 2]), sum(Y[, 1]), 20)$median)
})

test_that("distances, nulls, site effects", {
  x <- rnorm(8)
  expect_equal(unname(trait_distance(x)), abs(outer(x, x, "-")))
  expect_equal(unname(trait_distance(c("a", "b", "a"))), matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3))
  Z <- standardise(trait_distance(x))
  expect_equal(mean(Z[row(Z) != col(Z)]), 0, tolerance = 1e-10)
  expect_equal(gower_distance(x), trait_distance(x) / diff(range(x)), ignore_attr = TRUE)
  N0 <- swap_null(Y, seed = 2)
  expect_equal(colSums(N0), colSums(Y))
  expect_equal(rowSums(N0), rowSums(Y))
  expect_identical(N0, swap_null(Y, seed = 2))
  expect_named(site_effects(Y), c("sites", "species", "b", "a", "D"))
})

test_that("regression", {
  X <- standardise(trait_distance(rnorm(8)))
  fit <- cooccurrence_regression(X, Y, n_iter = 30, n_chains = 2, drop_lambda = TRUE, seed = 3)
  expect_s3_class(fit, "draws_df")
  expect_equal(posterior::variables(fit), c("alpha", "beta[1]"))
  expect_equal(posterior::ndraws(fit), 60)
  expect_equal(posterior::nchains(fit), 2)

  g <- rep(c("a", "b"), each = 4)
  fit <- cooccurrence_regression(list(X, X), Y, g = g, n_iter = 30, n_chains = 1, pooled = TRUE)
  expect_true(all(c("beta[1]", "beta[2]", "lambda[8]") %in% posterior::variables(fit)))
  expect_error(cooccurrence_regression(X, Y, pooled = TRUE), "only when g")
  expect_error(cooccurrence_regression(X[1:4, 1:4], Y), "expected 8 x 8")
})
