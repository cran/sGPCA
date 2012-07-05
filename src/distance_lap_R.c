#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/************************************************* 
Returns a one dimensional laplacian matrix that
is weighted by either a tri-cube kernel or a 
epanichnov kernel
*************************************************/

SEXP distance_lap_R(SEXP distance_matrix_r, SEXP window_size_r, SEXP op_r,SEXP m_r){
	
	
	/*initialize variables*/
	double *distance_matrix = REAL(distance_matrix_r);
	int m = *INTEGER(m_r);
	int option = *INTEGER(op_r);
	int window = *INTEGER(window_size_r);
	
	int i;
	int j;
	
	int diag_location = 0;
	double diag_sum = 0;
	int pr_counter = 0;                                                                          
	double max = 0;
	double distance = 0;
	double *ir;
	double *jc;
	double *pr;
	
	int nz = 0;
	
	/*count the number of non zero elements*/
	for(i = 0; i< m*m; i++){	            
		if(distance_matrix[i] < window) pr_counter++;                
	}
	
	nz = pr_counter;
	
	/*allocating memory for return vectors*/
	SEXP return_object,jc_r, ir_r,pr_r;
	PROTECT(return_object = allocVector(VECSXP,3));
	
	PROTECT(jc_r = allocVector(REALSXP,nz));
	jc = REAL(jc_r);
	
	PROTECT(ir_r = allocVector(REALSXP,nz));
	ir = REAL(ir_r);
	
	PROTECT(pr_r = allocVector(REALSXP,nz));
	pr = REAL(pr_r);
	
	pr_counter = 0;
	
	
	/* Create matrix*/
	for(i = 0; i< m; i++){
		for(j = 0; j <m; j++){
			distance = distance_matrix[i*m + j];
			if(distance < window){
				if(i != j){
					if(distance != 0){
						max = 1/distance;
						diag_sum += max;
						max *= -1;
						jc[pr_counter] = i;
						pr[pr_counter] = max;
						ir[pr_counter] = j;
						pr_counter++;
					}
				}else{
					jc[pr_counter] = i;
					pr[pr_counter] = 0;
					ir[pr_counter] = i;
					diag_location = pr_counter;
					pr_counter++;
				}       
				
			}
		}
		pr[diag_location] = diag_sum;
		diag_sum = 0;
	}
	
	/*set elements in vector that will be returned*/
	SET_VECTOR_ELT(return_object,0,jc_r);
	SET_VECTOR_ELT(return_object,1,ir_r);
	SET_VECTOR_ELT(return_object,2,pr_r);
	UNPROTECT(4);
	return(return_object);
	

}
