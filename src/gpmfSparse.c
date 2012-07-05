/*
 *  RC_gpmf_sparse.c
 *  
 *
 *  Created by Frederick Campbell on 5/16/13.
 *  Copyright 2013 __MyCompanyName__. All rights reserved.
 *
 */

#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/Lapack.h>
#include <R_ext/BLAS.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double L = 1.1;

char *chn = "N";
char *cht = "T";

double neg_one = -1;
double one = 1; 
double zero = 0;
int stride = 1;                                                         

int i;
int option;
int max_it;



int *jcQ;
int *irQ;
int *jcR;
int *irR;

int c; 
int r;
int begin;
int end;
int column;
int row;
int inside;

double threshold;

int rlen;
int *ipiv;
double *work;
int k;



/*************************************************
 Sparse Matrix Vector Multiplication
 *************************************************/


void sparseMV(int column, int begin, int end, int row, double* v_i, double* R, double*
			  temp_p_vec, int r, int c, int p, int *jcs, int* irs){
	for(column = 0; column<p; column++){
		begin = jcs[column];
		end = jcs[column+1];
		
		for(row = begin; row<end;row++){
			r = irs[row];
			c = column;
			temp_p_vec[r] += v_i[c] * R[row];
		}
		
	}
	
	
}



/*************************************************
 Sparse Matrix Multiplication AB
 *************************************************/


void sparseRMM(int column,int begin, int end, int row, double* temp_matrix,double* R, double*
			   X_hat, int r, int c, int inside, int m, int p, int *jcs, int* irs){
	for(column = 0; column<p; column++){
		begin = jcs[column];
		end = jcs[column+1];
		for(row = begin; row<end;row++){  
			r = irs[row];
			c = column;
			
			for(inside = 0; inside<m; inside++){
				temp_matrix[c*m+inside] += X_hat[r*m+inside] * R[row];
			}
		}
        
	}
	
	
}

/*************************************************
 Sparse Matrix Multiplication A'B
 *************************************************/

void sparseQMM(int column,int begin, int end, int row, double* temp_matrix, double* Q, double*
			   X_hat, int r, int c, int inside, int m, int p, int *jcq, int* irq){
	for(column = 0; column<m; column++){
		begin = jcq[column];
		end = jcq[column+1];
        for(row = begin; row<end;row++){
			r = irq[row];
			c = column;
			
			for(inside = 0; inside<p; inside++){
				temp_matrix[c*p+inside] += X_hat[r+m*inside] * Q[row];
			}
        }
		
    }
	
}



void sparseQX(int column,int begin, int end, int row, double* temp_matrix,double* R, double*
			  X_hat, int r, int c, int inside, int m, int p, int *jcs, int*
			  irs){                           
	for(column = 0; column<m; column++){
		begin = jcs[column];
		end = jcs[column+1];    
		
		for(row = begin; row<end;row++){
			r = irs[row];
			c = column;
			
			for(inside = 0; inside<p; inside++){
				temp_matrix[inside*m + r] += X_hat[c + m*inside] * R[row];
			}
		}
		
	}
	
}



/*************************************************
 Performs Soft Thresholding
 
 Returns a double
 *************************************************/

double soft_threshold(double u, double lambda, int pos){
	
	double return_value = fmax(fabs(u)-lambda,0);
	if (!pos && u < 0) return_value = return_value*-1;
	return(return_value);
	
}



/*************************************************
 Solves a penalized optimization problem with
 Iterative Soft Thresholding Algorithm (ista)
 *************************************************/

void ista(double* y,double* Q, double* u, double* temp_n_vec, double* oldu, double lambda,
		  int m, int pos,  int isQ, int QisSparse, int RisSparse){
	
	double error=1;
	double divide;
	double ll = lambda/L;
	double temp = 1/L;
	int iter = 0;
	
	while (error>threshold && iter < max_it) {
		memcpy(temp_n_vec,y,sizeof(double)*m);
		
		/*sets olduv to the u_{t-1} before finding u_t*/
		memcpy(oldu,u,sizeof(double)*m);
		
		/* calculate u + Q*(y-u)/L */
		F77_CALL(daxpy)(&m, &neg_one, u, &stride, temp_n_vec, &stride);
		
		if(QisSparse && isQ){
			F77_CALL(dscal)(&m,&temp,temp_n_vec,&stride);
			sparseMV(column, begin, end, row, temp_n_vec, Q, u, r, c, m, jcQ, irQ);
		}else if(RisSparse && !isQ){
			F77_CALL(dscal)(&m,&temp,temp_n_vec,&stride);
			sparseMV(column, begin, end, row, temp_n_vec, Q, u, r, c, m, jcR, irR);
		}else{
			F77_CALL(dgemv)(chn, &m, &m, &temp, Q, &m, temp_n_vec, &stride, &one, u, &stride);
		}
		
		/* performs soft thresholding on each coordinate of u */
		
		for(i = 0; i < m; i++){
			u[i] = soft_threshold(u[i],ll,pos);
		}
		
		/* checks the error */
		F77_CALL(daxpy)(&m, &neg_one, u, &stride, oldu, &stride);
		error = F77_CALL(dnrm2)(&m, oldu, &stride);
		divide = F77_CALL(dnrm2)(&m, u, &stride);
		error = error/divide;
		iter++;
		
	}
	
	
}


/*************************************************
 Solves for one component either u or v with consideration for
 a diagonal quadratic operator using ISTA
 
 *************************************************/

int solveuv(double *w, double* y,double* Q, double *u, double* temp_m_vec, double* oldu, double* index, double lambda,
			int m, int pos, int option, int isQ, int QisSparse, int RisSparse){
	
	double temp=0;
	
	/* check if y is zero. If so return u = 0  */
	double solve_sum = 0;
	for(i = 0; i< m; i++){
		solve_sum += fabs(y[i]);
	}
	
	if(solve_sum == 0){   
		for(i = 0; i < m; i++){
			u[i]=0;
		}
		return(0);
	}
	
	
	
	/*check if lambda is zero. If so u = y/sqrt(y'*Q*y)*/
	
	if(lambda == 0){
		if(QisSparse && isQ){
			for(i = 0; i<m; i++){
				temp_m_vec[i] = 0;
			}
			sparseMV(column, begin, end, row, y, Q, temp_m_vec, r, c, m, jcQ, irQ);
		}else if(RisSparse && !isQ){
			for(i =0; i<m; i++){
				temp_m_vec[i] = 0;
			}
			sparseMV(column, begin, end, row, y, Q, temp_m_vec, r, c, m, jcR, irR);
		}else{
			F77_CALL(dgemv)(chn, &m, &m, &one, Q, &m, y, &stride, &zero, temp_m_vec, &stride);
		}
		temp = F77_CALL(ddot)(&m, y, &stride, temp_m_vec, &stride);
		temp = 1/sqrt(temp);
		memcpy(u, y, sizeof(double)*m);
		F77_CALL(dscal)(&m, &temp, u, &stride);
		
		
		return(0);
	}
	
	/*check if diagonal */
	double mat_sum = 0;
	double trace = 0;
	
	if(QisSparse && isQ){
		for(i = 0; i<jcQ[m];i++){
			mat_sum += Q[i];
		}
		for(column = 0; column<m; column++){
			begin = jcQ[column];
			end = jcQ[column+1];
			for(row = begin; row<end;row++){
				r = irQ[row];
				c = column;
				if(r == c){
					trace += Q[row];
				}
			}
		}
		
	}else if(RisSparse && !isQ){
		for(i = 0; i<jcR[m];i++){
			mat_sum += Q[i];
		}
		for(column = 0; column<m; column++){
			begin = jcR[column];
			end = jcR[column+1];
			for(row = begin; row<end;row++){
				r = irR[row];
				c = column;
				if(r == c){
					trace += Q[row];
				}
			}
		}       
        
	}else{  
		for(i = 0; i < m; i++){
			trace += Q[i*m + i];
		}
		int dim = m*m;
		mat_sum = F77_CALL(dasum)(&dim,Q,&stride);
		
	}
	
    if(trace == mat_sum){
		
		/*find lambda/Q[i,i]*/
		for(i = 0; i < m; i++){
			if(QisSparse && isQ){
				begin = jcQ[i];
				end = jcQ[i+1];
				for(row = begin; row<end; row++){
					r = irQ[row];
					c = i;
					if(r == c){
						temp = lambda/Q[row];
					}
				}
			}else if(RisSparse && !isQ){    
				begin = jcR[i];
				end = jcR[i+1];
				for(row = begin; row<end;row++){
					r = irR[row];
					c = i;
					if(r == c){
						temp = lambda/Q[row];
					}
				}
                
			}else{
				temp = lambda/Q[i*m+i];
			}
			u[i] = soft_threshold(y[i], temp, pos);
		}        
		temp = 0;
		for(i = 0; i<m; i++){
			temp += fabs(u[i]);
		}
 		if(temp == 0){  
			return(0);
		}else{                  
			if(QisSparse && isQ){ 
				sparseMV(column, begin, end, row, u, Q, temp_m_vec, r, c, m, jcQ, irQ);
			}else if(RisSparse && !isQ){
				sparseMV(column, begin, end, row, u, Q, temp_m_vec, r, c, m, jcR, irR);
			}else{
				F77_CALL(dgemv)(chn, &m, &m, &one, Q, &m, u, &stride, &zero, temp_m_vec, &stride);
			}
			temp = F77_CALL(ddot)(&m, u, &stride, temp_m_vec, &stride);
			temp = 1/sqrt(temp);
			F77_CALL(dscal)(&m, &temp, u, &stride);
			
			return(0);
		}
	}else{  
		/*otherwise solve with ista or coordinate descent*/ 
		
		/*check if norm(Qy) = 0 if so return flag for new start values for u and v*/
		if(QisSparse && isQ){
			sparseMV(column, begin, end, row, y, Q, temp_m_vec, r, c, m, jcQ, irQ);
		}else if(RisSparse && !isQ){
			sparseMV(column, begin, end, row, y, Q, temp_m_vec, r, c, m, jcR, irR);
		}else{
			F77_CALL(dgemv)(chn, &m, &m, &one, Q, &m, y, &stride, &zero, temp_m_vec, &stride);
		}
		
		temp = F77_CALL(dnrm2)(&m,temp_m_vec,&stride);
		if(temp==0){
			for (i = 0; i<m; i++) {
				u[i] = 0;
			}
			return(1);
		}
		
		/* otherwise use ista */
		ista(y, Q, u, temp_m_vec, oldu, lambda, m, pos,isQ,QisSparse,RisSparse);
		
		
		/*check if u is zero then normalize */
		temp = F77_CALL(dnrm2)(&m,u,&stride);
		if(temp == 0){
			return(0);      
		}else{
			
			if(QisSparse && isQ){
				for(i = 0; i < m; i++){
					temp_m_vec[i] = 0;
				}
				sparseMV(column, begin, end, row, u, Q, temp_m_vec, r, c, m, jcQ, irQ);
			}else if(RisSparse && !isQ){
				for(i = 0; i < m; i++){
					temp_m_vec[i] = 0;
				}
				sparseMV(column, begin, end, row, u, Q, temp_m_vec, r, c, m, jcR, irR);
			}else{
				F77_CALL(dgemv)(chn, &m, &m, &one, Q, &m, u, &stride, &zero, temp_m_vec, &stride);
			}
			temp = F77_CALL(ddot)(&m, u, &stride, temp_m_vec, &stride);
			temp = 1/sqrt(temp);
			F77_CALL(dscal)(&m, &temp, u, &stride);
		}
	}
	
	return(0);
}


/*************************************************
 
 
 *************************************************/


void gpmf(double *w, double *u_i, double *v_i, double *dii, double *X, double *Q, double *R, double *X_hat, double * temp_matrix, double *oldu, 
		  double *oldv,
		  double *temp_m_vec, double *temp_p_vec,double *XRv, double *XQu, int m, int p, double lambdau, double 
		  lambdav, int QisSparse, int RisSparse,int posu, int posv,double *gpmf_oldu, double *gpmf_oldv, double *pass_index){
	
	int num_iter =0;
	int isQ = 0;
	double error = 1;
	int flag = 0;
	
	
 	while(error>threshold  ){
		num_iter++;
		memcpy(gpmf_oldu, u_i, sizeof(double)*m);
		memcpy(gpmf_oldv, v_i, sizeof(double)*p);
		
		if(RisSparse){
			sparseRMM(column, begin, end, row, temp_matrix, R, X_hat, r, c, inside, m, p, jcR, irR);
		}else{
			F77_CALL(dgemm)(chn, chn, &m, &p, &p, &one, X_hat, &m, R, &p, &zero, temp_matrix, &m);
		}
		
		F77_CALL(dgemv)(chn, &m, &p, &one, temp_matrix, &m, v_i, &stride, &zero, XRv, &stride);
		isQ = 1;
		
		flag = solveuv(w, XRv, Q, u_i, temp_m_vec,  oldu, pass_index, lambdau, m, posu, option, isQ, QisSparse, RisSparse);
		
		if(QisSparse){
			for(i=0;i<m*p; i++){
				temp_matrix[i]=0;
			}
			sparseQMM(column, begin, end, row, temp_matrix, Q, X_hat, r, c, inside, m, p, jcQ, irQ);
		}else{
			F77_CALL(dgemm)(cht, chn, &p, &m, &m, &one, X_hat, &m, Q, &m, &zero, temp_matrix, &p);
		}
		F77_CALL(dgemv)(chn, &p, &m, &one, temp_matrix, &p, u_i, &stride, &zero, XQu, &stride);
		isQ = 0;
		
		flag = solveuv(w, XQu, R, v_i, temp_p_vec,  oldv, pass_index, lambdav, p, posv, option,isQ,QisSparse,RisSparse);
		
		
		F77_CALL(daxpy)(&m, &neg_one, u_i, &stride, gpmf_oldu, &stride);
		F77_CALL(daxpy)(&p, &neg_one, v_i, &stride, gpmf_oldv, &stride);
		
		error = F77_CALL(dnrm2)(&m, gpmf_oldu, &stride);
		error += F77_CALL(dnrm2)(&p, gpmf_oldv, &stride);
		
		if(RisSparse){
			for(i=0;i<p;i++){
				temp_p_vec[i]= 0;
				gpmf_oldv[i]=0;
			}
			
		}
		if(QisSparse){
			for(i=0;i<m;i++){
				temp_m_vec[i]=0;
				gpmf_oldu[i]=0;
			}
		}
		if(QisSparse || RisSparse){
			for(i = 0; i<m*p; i++){
				temp_matrix[i]=0;
			}
		}       
        
		if(num_iter >= max_it){  
			/*printf("maximum number of iterations reached \n");*/
			break;   
			
		}
		
	}
	
	
	if(RisSparse){
		sparseRMM(column, begin, end, row, temp_matrix, R, X, r, c, inside, m, p, jcR, irR);
	}else{
		F77_CALL(dgemm)(chn, chn, &m, &p, &p, &one, X, &m, R, &p, &zero, temp_matrix, &m);
	}               
	F77_CALL(dgemv)(chn, &m, &p, &one, temp_matrix, &m, v_i, &stride, &zero, temp_m_vec, &stride);
	if(QisSparse){
		for(i=0;i<m;i++){
			oldu[i]=0;
		}
		sparseMV(column, begin, end, row, temp_m_vec, Q, oldu, r, c, m, jcQ, irQ);
	}else{
		F77_CALL(dgemv)(chn, &m, &m, &one, Q, &m, temp_m_vec, &stride, &zero, oldu, &stride);
	}   
	dii[0] = F77_CALL(ddot)(&m, u_i, &stride, oldu, &stride);
}




SEXP gpmfSparse(SEXP X_r,SEXP Q_r, SEXP R_r, SEXP jcq_r,SEXP irq_r, SEXP jcr_r, SEXP irr_r, SEXP RisSparse_r, SEXP QisSparse_r, SEXP lamu_r, SEXP lamv_r, SEXP pu_r, SEXP pv_r, SEXP threshold_r, SEXP maxit_r, SEXP start_u_r, SEXP start_v_r){

	/* input output variables*/
	
	double *X, *Q, *R, *u, *v, *lamu, *lamv, *U,*V,*D,*ugmd,*vgmd;
	
	X = REAL(X_r);
	Q = REAL(Q_r);
	R = REAL(R_r);
	
	int QisSparse = *INTEGER(QisSparse_r);
	int RisSparse = *INTEGER(RisSparse_r);
	
	if(QisSparse){
		jcQ = INTEGER(jcq_r);
		irQ = INTEGER(irq_r);
	}
	
	if(RisSparse){
		jcR = INTEGER(jcr_r);
		irR = INTEGER(irr_r);
	}
	
	
	lamu = REAL(lamu_r);
	lamv = REAL(lamv_r);
	
	double lambdau = *lamu;
	double lambdav = *lamv;
	
	
	int posu = *INTEGER(pu_r);
	int posv = *INTEGER(pv_r);

	threshold = *REAL(threshold_r);
	
	max_it = *INTEGER(maxit_r);

	
	ugmd = REAL(start_u_r);
	vgmd = REAL(start_v_r);
	
	
	k = 1;
	rlen =1;
	
	SEXP Rdim = getAttrib(X_r,R_DimSymbol);
	int m = INTEGER(Rdim)[0];
	int p = INTEGER(Rdim)[1];
	int maxmp = m;
	if(p>m){
		maxmp = p;
	}
	
	



	
	SEXP return_object, U_r, V_r, D_r;
	PROTECT(return_object = allocVector(VECSXP,3));
	

	
	PROTECT(U_r = allocMatrix(REALSXP,m,k));
	U = REAL(U_r);
	
	PROTECT(V_r = allocMatrix(REALSXP,p,k));
	V = REAL(V_r);
	
	PROTECT(D_r = allocVector(REALSXP,k));
	D = REAL(D_r);
	
	/* temporary space */
	
	double  *X_hat, *temp_matrix, *oldu, *oldv, *temp_m_vec, *temp_p_vec, *w, *XRv, *XQu, *gpmf_oldu, *gpmf_oldv, *pass_index;
	
	u = (double*)R_alloc(m,sizeof(double));
	v = (double*)R_alloc(p,sizeof(double));
	temp_matrix = (double*)R_alloc(m*p,sizeof(double));
 	X_hat = (double*)R_alloc(m*p,sizeof(double));
	oldu = (double*)R_alloc(m,sizeof(double));
	oldv = (double*)R_alloc(p,sizeof(double));
	temp_m_vec = (double*)R_alloc(m,sizeof(double));
	temp_p_vec = (double*)R_alloc(p,sizeof(double));
	w =(double*)R_alloc(maxmp,sizeof(double));
	XRv = (double*)R_alloc(m,sizeof(double));
	XQu = (double*)R_alloc(p,sizeof(double));
	gpmf_oldu = (double*)R_alloc(m,sizeof(double));
	gpmf_oldv = (double*)R_alloc(p,sizeof(double));
	pass_index = (double*)R_alloc(maxmp,sizeof(double));
		
	
	int ipiv_ptr[k];
	double work_ptr[2*k];
	ipiv = ipiv_ptr;
	work = work_ptr;	
	
	
	for(i =0; i<maxmp; i++){
		pass_index[i]=1;
	}
	
	
	
	gpmf(w, ugmd, vgmd, D, X, Q, R, X, temp_matrix, oldu, oldv,temp_m_vec, temp_p_vec,XRv, XQu, m, p, 
		 lambdau,lambdav,QisSparse,RisSparse,posu, posv,gpmf_oldu,gpmf_oldv, pass_index);
	
	memcpy(U,ugmd,sizeof(double)*m);
	memcpy(V,vgmd,sizeof(double)*p);
	
	SET_VECTOR_ELT(return_object,0,U_r);
	SET_VECTOR_ELT(return_object,1,V_r);
	SET_VECTOR_ELT(return_object,2,D_r);
	
	
	UNPROTECT(4);
	return(return_object);

	
 }

