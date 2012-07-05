\name{Laplacian}
\alias{laplacian}
\alias{distLaplacian}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Graph Laplacian Matrices
}
\description{
  Functions for generating graph laplacian matrices based on nearest
  neighbors in a grid structure or neighbors a certain distance apart.
}
\usage{
laplacian(dims)
distLaplacian(matrix,window=2)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{dims}{
A vector containing the dimensions of the underlying grid.  One, two and
three-dimensional grids are supported.   
}
\item{matrix}{A full distance matrix. This can be computed via any
  distance metric.}

\item{window}{A scalar denoting the maximum distance at which points are
considered neighbors, meaning they share an edge in the graph structure.}
}



\details{
Graph Laplacian matrices can be used as generalizing operators in the
GPCA framework.  These behave like inverse covariances and are useful
for recovering edges in the GPCA factors.  Laplacians are defined as the
degree matrix minus the 
adjacency matrix for a network structure.


For variables structured as a grid, Laplacians are constructed by
placing edges between neighbors in the grid structure.  For 
one dimensional grids (e.g. equally-spaced time points), this is a chain
graph.  Nearest neighbors for two and three dimensional grids
(e.g. image data) are defined using the Chebychev distance.   Laplacians
for grid structures are scaled to have operator norm one, which is
required for use with \code{gpca} and \code{sgpca}.  


Laplacians are constructed from a general distance matrix by placing
edges between variables less than or equal to \code{window} distance
apart.  These are returned without scaling to have operator norm one,
and must be appropriately scaled before used with \code{gpca} and
\code{sgpca}.  

  
% Laplacians on a grid are scaled to have operator norm 1. Laplacians on
% a general distance matrix must be scaled to have a maximum eigenvalue of
% 1 before use with SGPCA.  



% Using the laplacian on a grid as a generalizing operator is appropriate
% if the data has an underlying grid structure.  For example time series
% data can be represented by a one 
% dimensional graph. Images and pixels can be represented by a two
% dimensional grid. Three dimensional images like fMRI data can be
% represented by a 
% three dimensional grid where each node in the graph represents a voxel in the image. 


}
\value{
  \item{Q}{A sparse matrix of the class dgCMatrix.}
}
\author{
Frederick Campbell
}


\examples{
#Laplacians on a 1D, 2D, and 3D grid
Q = laplacian(10)
Q = laplacian(c(10,10))
Q = laplacian(c(10,10,10))

#Laplacians computed based on Euclidean distances
data(ozone2)
D = as.matrix(dist(ozone2$lon.lat))
Q = distLaplacian(D,2)
Q = Q/max(eigen(Q,only.values=TRUE)$values)
}

\keyword{ laplacian }

