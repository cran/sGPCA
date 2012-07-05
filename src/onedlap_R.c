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
Returns a one dimensional laplacian matrix
*************************************************/


SEXP onedlap_R(SEXP dims_r){
	
	int m = *INTEGER(dims_r);
	
	/*initialize variables*/
	int nz;
	double *pr;
	double *ir, *jc;
	
	int i;
	
	int i_win;
	
	int iplus;
	int iminus;
	
	int pr_counter = 0;                                                                          

	nz = 3*m - 2;
	
	
	SEXP return_object,jc_r, ir_r,pr_r;
	PROTECT(return_object = allocVector(VECSXP,3));
	
	PROTECT(jc_r = allocMatrix(REALSXP,nz,1));
	jc = REAL(jc_r);
	
	PROTECT(ir_r = allocMatrix(REALSXP,nz,1));
	ir = REAL(ir_r);
	
	PROTECT(pr_r = allocMatrix(REALSXP,nz,1));
	pr = REAL(pr_r);
	
	int diag_location=0;
	double diag_counter = 0;
	
	for(i = 0; i< m; i++){
		iplus = i+1;
		iminus = i-1;
		for(i_win = iminus; i_win <= iplus; i_win++){
			if(i_win > -1 && i_win < m){
				if(i != i_win){
					diag_counter++;
					jc[pr_counter] = i;
					pr[pr_counter] = -.25;
					ir[pr_counter] = i_win;
					pr_counter++;
				}else{
					jc[pr_counter] = i;
					pr[pr_counter] = 0;
					ir[pr_counter] = i_win;
					diag_location = pr_counter;
					pr_counter++;
				}
			}
		}
		pr[diag_location] += diag_counter/4;
		diag_counter =0;
	}
	
	
	
	
	SET_VECTOR_ELT(return_object,0,jc_r);
	SET_VECTOR_ELT(return_object,1,ir_r);
	SET_VECTOR_ELT(return_object,2,pr_r);
	UNPROTECT(4);
	return(return_object);
	
	
}
