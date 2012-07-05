\name{sGPCA}
\alias{sGPCA}
\docType{package}
\title{
Sparse Generalized Principal Component Analysis
}
\description{
Functions for computing Sparse Generalized Principal Components, including functions for modeling structured correlation
}
\details{
\tabular{ll}{
Package: \tab sgpca\cr
Type: \tab Package\cr
Version: \tab 1.0\cr
Date: \tab 2012-07-05\cr
License: \tab BSD-3\cr
}


}
\author{
Frederick Campbell

Maintainer: Frederick Campbell <frederick.campbell@rice.edu> 

}
\references{
Genevera I. Allen, Logan Grosenick, and Jonathan Taylor, "A generalized
least squares matrix decomposition", arXiv:1102.3074, 2011. 
}

\keyword{ package }

% \examples{

% ## Using  sGPCA 

% X = matrix(nrow = 4 , ncol = 5,c( 2.000,  9.00,  2.0  ,  6,3.000,
% 14.00,  1.2, 7, 0.500,  0.23,  3.0,  8, 0.333,  8.10,  4.0,
% 9, 7.500,  9.00,  5.0,    2))
% X = t(X)
% Q = matrix( nrow = 5,ncol = 5,c(2.1,0 -0.72, -0.63, -0.48, -0.27, 
% -0.72,  2.55, -0.72, -0.63, -0.48, -0.63, -0.72,  2.70, -0.72, 
% -0.63, -0.48, -0.63, -0.72,  2.55, -0.72, -0.27, -0.48, -0.63,
% -0.72,  2.10))
% Q =as.matrix(Q/max(eigen(Q)$values))
% R = as.matrix(laplacian(4))

% sgpca(X,Q,R,2)


% }
