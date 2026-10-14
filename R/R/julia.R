.cr <- new.env(parent = emptyenv())

julia_url <- "https://github.com/EvoArt/CooccurrenceRegression.jl"

#' Start Julia and load CooccurrenceRegression.jl
#'
#' Called by every other function in the package, so it only needs calling directly to
#' choose the Julia environment or to pass options to [JuliaCall::julia_setup()]. The first
#' call in a session takes a minute or more while Julia loads and compiles the package.
#'
#' The Julia package is kept in its own Julia environment, by default a folder under
#' `tools::R_user_dir("CooccurrenceRegression", "data")`, and installed there on first use.
#' To use another environment, for example one in which a local copy of the Julia package has
#' been added with `Pkg.develop`, give its folder as `project` or set the environment
#' variable `COOCCURRENCE_JULIA_PROJECT`.
#'
#' @param project Folder of the Julia environment.
#' @param ... Passed to [JuliaCall::julia_setup()], e.g. `JULIA_HOME`.
#' @return `TRUE`, invisibly.
#' @export
cr_setup <- function(project = NULL, ...) {
  if (isTRUE(.cr$ready)) return(invisible(TRUE))
  if (is.null(project)) project <- Sys.getenv("COOCCURRENCE_JULIA_PROJECT")
  if (!nzchar(project)) project <- file.path(tools::R_user_dir("CooccurrenceRegression", "data"), "julia")
  dir.create(project, recursive = TRUE, showWarnings = FALSE)
  JuliaCall::julia_setup(...)
  JuliaCall::julia_eval("import Pkg")
  JuliaCall::julia_call("Pkg.activate", normalizePath(project, winslash = "/"))
  if (!JuliaCall::julia_eval('haskey(Pkg.project().dependencies, "CooccurrenceRegression")')) {
    message("Installing CooccurrenceRegression.jl into ", project)
    JuliaCall::julia_call("Pkg.add", url = julia_url)
  }
  JuliaCall::julia_eval("import CooccurrenceRegression")
  .cr$ready <- TRUE
  invisible(TRUE)
}

jl <- function(fun, ...) {
  cr_setup()
  JuliaCall::julia_call(paste0("CooccurrenceRegression.Interop.", fun), ...)
}

drop_null <- function(x) x[!vapply(x, is.null, logical(1))]

presence <- function(Y) {
  Y <- as.matrix(Y)
  if (anyNA(Y)) stop("Y contains missing values")
  Y > 0
}

label <- function(x) if (is.null(x)) NULL else as.character(x)

trait <- function(x) if (is.numeric(x)) as.numeric(x) else as.character(x)
