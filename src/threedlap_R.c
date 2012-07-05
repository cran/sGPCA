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
Returns a three dimensional laplacian matrix     
*************************************************/

SEXP threedlap_R(SEXP dim_m_r,SEXP dim_p_r, SEXP dim_n_r){
	
	int m = *INTEGER(dim_m_r);
	int p = *INTEGER(dim_p_r);
	int n = *INTEGER(dim_n_r);
	
	/*initialize variables*/
	int nz;
	double *pr;
	double *ir, *jc;
	
	int i;
	int j;
	int k;
	int i_win;
	int j_win;
	int k_win;
	
	int kplus;
	int kminus;
	int jplus;
	int jminus;
	int iplus;
	int iminus;
	
	
	nz = (3 * m - 2) * (3 * p - 2) * (3 * n - 2);
	SEXP return_object,jc_r, ir_r,pr_r;
	PROTECT(return_object = allocVector(VECSXP,3));
	
	PROTECT(jc_r = allocMatrix(REALSXP,nz,1));
	jc = REAL(jc_r);
	
	PROTECT(ir_r = allocMatrix(REALSXP,nz,1));
	ir = REAL(ir_r);
	
	PROTECT(pr_r = allocMatrix(REALSXP,nz,1));
	pr = REAL(pr_r);
	
	int iteration = 0;
	int pr_counter = 0;
	
	double diag_counter = 0;
	int diag_location = 0;
	
	for(k = 0; k < n; k++){
		for(j = 0; j < p; j++){
			for(i = 0; i< m; i++){
				kplus = k + 1;
				kminus = k - 1;
				for(k_win = kminus; k_win <= kplus; k_win++){
					if(k_win > -1 && k_win < n){
						jplus = j+1;
						jminus = j-1;
						for(j_win = jminus; j_win <= jplus; j_win++){
							if(j_win > -1 && j_win < p){
								iplus = i+1;
								iminus = i-1;
								for(i_win = iminus; i_win <= iplus; i_win++){
									if(i_win > -1 && i_win < m){
										if(!(i== i_win && j==j_win && k==k_win)){  
  											pr[pr_counter] = -.027777;
											int row = i_win + j_win*m + k_win*m*p;
											ir[pr_counter] = row;	
											jc[pr_counter] = iteration;
											pr_counter++;
											diag_counter++;
										}else{
											pr[pr_counter] = 0;
											ir[pr_counter] = i_win + j_win*m + k_win*m*p;
											diag_location = pr_counter;
											jc[pr_counter] = iteration;
											pr_counter++;
											
										}
									}
								}
							}
						}
					}
				}
				pr[diag_location] += diag_counter/36;
				diag_counter =0;
				iteration++;
			}
		}
	}
	

	
	SET_VECTOR_ELT(return_object,0,jc_r);
	SET_VECTOR_ELT(return_object,1,ir_r);
	SET_VECTOR_ELT(return_object,2,pr_r);
	UNPROTECT(4);
	return(return_object);
	
}	
	
	
