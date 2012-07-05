\name{sgpca}
\alias{sgpca}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Sparse Generalized Principal Component Analysis
}
\description{
Computes the rank \code{K} sparse, sparse
non-negative, two-way sparse, and two-way sparse non-negative GPCA
solutions.  
}
\usage{
sgpca(X, Q, R, K = 1, lamu = 0, lamvs = 0, posu = FALSE, posv = FALSE, 
threshold = 1e-07, maxit = 1000, full.path = FALSE) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{The \code{n x p} data matrix.  \code{X} must be of
    class \code{matrix} with all numeric values.
}
  \item{Q}{The row generalizing operator, an \code{n x n}
    matrix. \code{Q} can be of class \code{matrix} or class
    \code{dcGMatrix}, must be positive semi-definite, and have operator
    norm one.
}
  \item{R}{The column generalizing operator, an \code{p x p}
    matrix. \code{R} can be of class \code{matrix} or class
    \code{dcGMatrix}, must be positive semi-definite, and have operator
    norm one.
}
  \item{K}{The number of GPCA components to compute. The default value
    is one.
  }
  \item{lamu}{The regularization parameter that determines the sparsity
    level for the row factor, \code{U}.  The default value is 0.  If the
    data is oriented with rows as samples, non-zero \code{lamu}
    corresponds to two-way sparse methods.
}
\item{lamvs}{A scalar or vector of regularization parameters that
  determine the sparsity level for the column factor, \code{V}.  The
  default is 0, with non-zero values corresponding to sparse or two-way
  sparse methods.  If
  \code{lamvs} is a vector, then the BIC method is used to select the
  optimal sparsity level.  Alternatively, if \code{full.path} is
  specified, then the solution at each value of \code{lamvs} is
  returned.  
  }
  \item{posu}{Flag indicating whether the row factor, \code{U} should be
    constrained to be strictly positive.  The default value is FALSE.
}
  \item{posv}{Flag indicating whether the column factor, \code{V} should be
    constrained to be strictly positive.  The default value is FALSE.
}
  \item{threshold}{Sets the threshold for convergence. The default value
    is \code{.0001}. 
}
  \item{maxit}{Sets the maximum number of iterations. The default value
    is \code{.0001}.
}
\item{full.path}{Flag indicating whether the entire solution path, or
  the solution at each value of \code{lamvs}, should be returned.  The
  default value is FALSE.
}
}
\details{
  The \code{sgpca} function has the flexibility to fit combinations of
  sparsity and/or non-negativity for both the row and column generalized
  PCs.   Regularization is used to
  encourage sparsity in the GPCA factors by placing an L1
  penalty on the GPC loadings, \code{V}, and or the sample GPCs,
  \code{U}.  Non-negativity constraints on \code{V} and/or \code{U}
  yield sparse non-negative and two-way non-negative GPCA.
  Generalizing operators as described for \code{\link{gpca}} can be used
  with this function and have the same properties.

  When \code{lamvs=0}, \code{lamu=0}, \code{posu=0}, and \code{posv=0},
  the GPCA solution also given by \code{\link{gpca}} is returned.  The
  magnitude of the regularization parameters, \code{lamvs} and
  \code{lamu}, determine the level of sparsity of the factors
  \code{U} and \code{V}, with higher regularization parameter values
  yielding sparser factors.    
  If more than one regularization value \code{lamvs} is given, then
  \code{sgpca} finds the optimal regularization parameter
  \code{lamvs} by minimizing the BIC derived from the generalized
  least squares update for each factor. 

  If \code{full.path = TRUE}, then the full path of solutions (\code{U},
  \code{D}, and \code{V}) is returned for each value of \code{lamvs}
  given.  This option is best used with 50 or 100 values of
  \code{lamvs} to well approximate the regularization paths.
  Numerically, the path begins with the GPCA solution, \code{lamvs=0},
  and uses warm starts at each step as \code{lamvs} 
  increases.  

  Proximal gradient descent is used to compute each rank-one solution.
  Multiple components are calculated in a greedy manner via deflation.
  Each rank-one solution is solved by iteratively fitting generalized
  least squares problems with penalties or non-negativity constraints.
  These regression problems are solved by the Iterative Soft-Thresholding
  Algorithm (ISTA) or projected gradient descent.
}
\value{
  \item{U}{The left sparse GPCA factors, an \code{n x K} matrix.  If
  \code{full.path} is specified with \code{r} values of \code{lamvs},
  then \code{U} is a \code{n x K x r} array.}
  \item{V}{The right sparse GPCA factors, a \code{p x K} matrix.  If
  \code{full.path} is specified with \code{r} values of \code{lamvs},
  then \code{V} is a \code{p x K x r} array.}
  \item{D}{A vector of the K sparse GPCA values.  If
  \code{full.path} is specified with \code{r} values of \code{lamvs},
  then \code{D} is a \code{K x r} matrix.}
  \item{cumulative.prop.var}{The cumulative proportion of variance explained 
  by the components}
  \item{bics}{The BIC values computed for each value of \code{lamvs} and each of
  the \code{K} components.}  
  \item{optlams}{Optimal regularization parameter as chosen by the BIC
  method for each of the \code{K} components.} 
}
\references{
Genevera I. Allen, Logan Grosenick, and Jonathan Taylor, "A generalized
least squares matrix decomposition", arXiv:1102.3074, 2011. 

Genevera I. Allen and Mirjana Maletic-Savatic, "Sparse Non-negative
Generalized PCA 
with Applications to Metabolomics", Bioinformatics, 27:21, 3029-3035,
2011.  
}
\author{
Frederick Campbell
}


\examples{
data(ozone2)
ind = which(apply(is.na(ozone2$y),2,sum)==0)
X = ozone2$y[,ind]
n = nrow(X)
p = ncol(X)
#Generalizing Operators - Spatio-Temporal Smoothers
R = Exp.cov(ozone2$lon.lat[ind,],theta=5)
er = eigen(R,only.values=TRUE);
R = R/max(er$values)
Q = Exp.cov(c(1:n),c(1:n),theta=3)
eq = eigen(Q,only.values=TRUE)
Q = Q/max(eq$values)

#Sparse GPCA
fit = sgpca(X,Q,R,K=1,lamu=0,lamvs=c(.5,1))
fit$prop.var #proportion of variance explained
fit$optlams #optimal regularization param chosen by BIC
fit$bics #BIC values for each lambda

#Sparse Non-negative GPCA
fit = sgpca(X,Q,R,K=1,lamu=0,lamvs=1,posv=TRUE)

#Two-way Sparse GPCA
fit = sgpca(X,Q,R,K=1,lamu=1,lamvs=1)

#Two-way Sparse Non-negative GPCA
fit = sgpca(X,Q,R,K=1,lamu=1,lamvs=1,posu=TRUE,posv=TRUE)

#Return full regularization paths for inputted lambda values
fit = sgpca(X,Q,R,K=1,lamu=0,lamvs=c(.1,.5,1),full.path=TRUE)
}

\keyword{ PCA }
\keyword{principal components analysis}
\keyword{multivariate analysis}
\keyword{matrix factorization}
\keyword{Sparse PCA}
\keyword{Non-negative Matrix Factorization}
\keyword{Non-negative PCA}

