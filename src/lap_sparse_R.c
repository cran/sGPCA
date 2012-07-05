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



/************************************************* 
 Returns a one dimensional laplacian matrix that
 is weighted by either a tri-cube kernel or a 
 epanichnov kernel
 *************************************************/

SEXP kern_sparse_R(SEXP passed_pr_r, SEXP passed_ir_r, SEXP passed_jc_r, SEXP window_r, SEXP op_r, SEXP m_r){
	
    /* read input*/
	int *irD = INTEGER(passed_ir_r);
	int *jcD = INTEGER(passed_jc_r);
	double *distance_matrix = REAL(passed_pr_r);
	int window = *INTEGER(window_r);
	int option = *INTEGER(op_r);
	int m = *INTEGER(m_r);
   	
	/*initialize variables*/
	int pr_counter = 0;                                                                          
	double max;
	int nz = 0;
	int diag_location = 0;
	double diag_sum = 0;
	
	int column;
	int row;
	int begin;
	int end;
	int i;
	double *jc;
	double *ir;
	double *pr;
	
	
	/*count the number of non zero elements*/
	for(i=0;i<jcD[m];i++){
		if(distance_matrix[i] < window) pr_counter++;
		nz = pr_counter+m;
	}
	
	
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
	int diag_set = 0;
	

	/*create Matrix*/
	for(column = 0; column<m; column++){
		begin = jcD[column];
		end = jcD[column+1];	
		
		for(row = begin; row<end;row++){
			if(irD[row] > column && diag_set == 0){
				diag_set = 1;
				pr[pr_counter]=1;
				ir[pr_counter]=column;
				jc[pr_counter]=column;
				diag_location = pr_counter;
				pr_counter++;
			}
			max = distance_matrix[row];
			if(max < window){
				max = 1/max;
				diag_sum += max;
				max *= -1;
				jc[pr_counter]=column;
				pr[pr_counter] = max;
				ir[pr_counter] = irD[row];
				pr_counter++;
			}
		} 
		if(diag_set==0){	
			jc[pr_counter]=column;
			pr[pr_counter]=1;
			ir[pr_counter]=column;
			diag_location = pr_counter;
			pr_counter++;
		}
		diag_set =0;
		pr[diag_location] = diag_sum;
		diag_sum =0;
	}            
	
	
	/*set elements in vector that will be returned*/
	SET_VECTOR_ELT(return_object,0,jc_r);
	SET_VECTOR_ELT(return_object,1,ir_r);
	SET_VECTOR_ELT(return_object,2,pr_r);
	UNPROTECT(4);
	return(return_object);
	
	
}
