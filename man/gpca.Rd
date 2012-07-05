\name{gpca}
\alias{gpca}

\title{
Generalized Principal Component Analysis
}
\description{
Computes the rank \code{K} Generalized PCA (GPCA) solution. 
}
\usage{
gpca(X, Q, R, K, deflation=FALSE)
}

\arguments{
  \item{X}{The \code{n x p} data matrix.  \code{X} must be of
    class \code{matrix} with all numeric values.   
}
  \item{Q}{The row generalizing operator, an \code{n x n}
    matrix. \code{Q} can be of class \code{matrix} or class
    \code{dcGMatrix}, but the function is optimized for sparse matrices of
    class \code{dCGMatrix}.  \code{Q} must also be positive
    semi-definite and be scaled to have operator norm one.  

    
}
  \item{R}{The column generalizing operator, an \code{p x p}
    matrix. \code{R} can be of class \code{matrix} or class
    \code{dcGMatrix}, but the function is optimized for sparse matrices of
    class \code{dCGMatrix}.  \code{R} must also be positive
    semi-definite and be scaled to have operator norm one.  
}
  \item{K}{The number of GPCA components to compute. The default value
    is one.
  }
  \item{deflation}{Algorithm used to calculate the solution.  Default is
    \code{deflation=FALSE} and most users should not deviate from this
    option.  See details.
  }
}
\details{
The Generalized PCA solution maximizes the sample variance of the data
in an inner-product space induced by the row and column generalizing
operators, \code{Q} and \code{R}, and also finds the best low-rank
approximation to the data as 
measured by a generalization of the Frobenius norm.  Note that the
resulting GPCA factors \code{U} and \code{V} are orthogonal with
respect to the row and column generalizing operators: \code{ U^T Q U = I} 
and \code{V^T R V = I}.  Generalized PCA can be interpreted as finding 
major modes of variation that are independent from the generalizing 
operators.  Thus, if \code{Q} and \code{R} encode noise structures 
(see \code{\link{laplacian}}) or noise covariances (see \code{\link{Exp.cov}}), 
then GPCA finds patterns separate from the structure of the noise.  


The generalizing operators, \code{Q} and \code{R}, must be positive
semi-definite and have operator norm one.  Note that if these are the
identity matrix, then GPCA is equivalent to PCA and \code{gpca} returns
the SVD of \code{X}.  Smoothers, such as covariances (see
\code{\link{Exp.cov}},\code{\link{Exp.simple.cov}},\code{\link{Rad.cov}},
\code{\link{stationary.cov}},\code{\link{cubic.cov}},\code{\link{stationary.taper.cov}},
\code{\link{wendland.cov}}), and inverse smoothers (see \code{\link{laplacian}}) 
can be used as generalizing operators for data in which variables are associated 
with a specific location (e.g. image data and spatio-temporal data). 


This function has the option of using one of two algorithms to compute
the solution.  The \code{deflation = FALSE} option computes the
eigen-decomposition 
of a quadratic form of dimension \code{min(n,p)} to find \code{U}
or \code{V} and finds the other factor by regression.  The
\code{deflation = TRUE}
option finds each factor using the generalized power algorithm and
performs to deflation to compute multiple factors.  The \code{deflation
  = FALSE}
option is generally faster, and especially so when one dimension is much
smaller than the other.  The option \code{deflation = TRUE} is faster only
if both dimensions are large \code{n,p > 5,000} and \code{K}
is small.   
  
}
\value{
  \item{U}{The left GPCA factors, an \code{n x K} matrix.}
  \item{V}{The right GPCA factors,  an \code{p x K} matrix.}
  \item{D}{A vector of the \code{K} PCA values.}
  \item{cumm.prop.var}{Cumulative proportion of variance explained by
    the first \code{K} components.}
  \item{prop.var}{Proportion of variance explained by each component.}
}

\references{
Genevera I. Allen, Logan Grosenick, and Jonathan Taylor, "A generalized
least squares matrix decomposition", arXiv:1102.3074, 2011. 
}
\author{
Frederick Campbell
}
\seealso{\code{\link{laplacian}}, \code{\link{Exp.cov}},
  \code{\link{sgpca}}} 
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

#SVD
fitsvd = gpca(X,diag(n),diag(p),1)

#GPCA
fitgpca = gpca(X,Q,R,1)
fitgpca$prop.var #proportion of variance explained

}
\keyword{PCA}
\keyword{SVD}
\keyword{principal components analysis}
\keyword{multivariate analysis}
\keyword{matrix factorization}
