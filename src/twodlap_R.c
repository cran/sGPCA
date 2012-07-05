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
Returns a two dimensional laplacian matrix     
*************************************************/

SEXP twodlap_R(SEXP dim_m_r,SEXP dim_p_r){
	
	int m = *INTEGER(dim_m_r);
	int p = *INTEGER(dim_p_r);
	
	/*initialize variables*/
	int nz;
	double *pr;
	double *ir, *jc;

	int i;
	int j;
	
	int i_win;
	int j_win;
	
	int pr_counter = 0;
	int iteration = 0;
	int jplus;
	int jminus;
	int iplus;
	int iminus;
	
	double diag_counter = 0;
	int diag_location=0;           
	
	
	nz = 9*m*p - 6*m - 6*p + 4;
	
	
	SEXP return_object,jc_r, ir_r,pr_r;
	PROTECT(return_object = allocVector(VECSXP,3));
	
	PROTECT(jc_r = allocMatrix(REALSXP,nz,1));
	jc = REAL(jc_r);
	
	PROTECT(ir_r = allocMatrix(REALSXP,nz,1));
	ir = REAL(ir_r);
	
	PROTECT(pr_r = allocMatrix(REALSXP,nz,1));
	pr = REAL(pr_r);
	
	
	for(j = 0; j < p; j++){
		for(i = 0; i< m; i++){
			
			
			jplus = j+1;
			jminus = j-1;
			for(j_win = jminus; j_win <= jplus; j_win++){
				if(j_win > -1 && j_win < p){
					iplus = i+1;
					iminus = i-1;
					for(i_win = iminus; i_win <= iplus; i_win++){
						if(i_win > -1 && i_win < m){
							if(!(i== i_win && j==j_win)){
								
								pr[pr_counter] = -.083333;
								int row = i_win + j_win*m;
								ir[pr_counter] = row;
								jc[pr_counter] = iteration;
								pr_counter++;
								diag_counter++;
							}else{
								jc[pr_counter] = iteration;
								pr[pr_counter] = 0;
								ir[pr_counter] = i_win+ j_win*m;
								diag_location = pr_counter;
								pr_counter++;
								
							}
						}
					}
				}
			}  
			
			pr[diag_location] += diag_counter/12;
			diag_counter =0;
			iteration++;
		}
	}
	
	
	
	SET_VECTOR_ELT(return_object,0,jc_r);
	SET_VECTOR_ELT(return_object,1,ir_r);
	SET_VECTOR_ELT(return_object,2,pr_r);
	UNPROTECT(4);
	return(return_object);
	
	
}


