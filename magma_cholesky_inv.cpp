#include <stdio.h>
#include <stdlib.h>
#include <cublas.h>
#include "magma.h"
#include "magma_lapack.h"

void copy_upper_diag(double *a, int n) {
	int i,j;
	for(i=0; i<n; i++) {
 	for (j=0; j<i; j++)
  		a[i*n+j] = (a[j*n+i]);
 	}
}

void show_matrix(double *A, int n) {
    int i,j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++)
            printf("%2.5f ", A[i * n + j]);
        printf("\n");
    }
}

double *cholesky_gpu(double *ml, int m) {
 	magma_int_t mm = m*m;
 	magma_int_t info;
 	double *a;
 	double *d_a ;
 	magma_err_t err; 
 	err = magma_dmalloc_cpu ( &a , mm );
 	err = magma_dmalloc ( &d_a , mm );

 	magma_dsetmatrix ( m, m, ml, m, d_a , m );

 	magma_dpotrf_gpu('L',m,d_a,m,&info);
 	magma_dpotri_gpu('L',m,d_a,m,&info);

 	magma_dgetmatrix ( m, m, d_a , m, a, m );
 	magma_free (d_a );
 
 	return a;
}

double *cholesky(double *ml, int m) {
 	magma_int_t info;
 
 	magma_dpotrf('L',m,ml,m,&info);
 	magma_dpotri('L',m,ml,m,&info);
 
 	return ml;
}

double *generate_sym_matrix(int m) {
 	double *a;
 	magma_int_t i,j;
 	magma_int_t mm = m*m;
 	magma_err_t err;
 	err = magma_dmalloc_cpu ( &a , mm );

 	magma_int_t ione = 1;
 	magma_int_t ISEED [4] = {0 ,0 ,0 ,1};
 	lapackf77_dlarnv (&ione ,ISEED ,&mm ,a);
 	for(i=0; i<m; i++) {
 		MAGMA_D_SET2REAL (a[i*m+i],( MAGMA_D_REAL (a[i*m+i ])+1.* m ) );
 		for (j=0; j<i; j++)
 			a[i*m+j] = (a[j*m+i]);
 	} 

 	return a;
}


int main(int argc, char** argv) {
 	magma_init();
 	magma_timestr_t start , end;
 	double gpu_time ;
 	double *c;
 	int dim[] = {20000,30000,40000};
 	int i,n;
 	n = sizeof(dim) / sizeof(dim[0]);
 
 	for(i=0; i < n; i++) {
 		magma_int_t m = dim[i];
 		magma_int_t mm=m*m;
 		magma_err_t err;
 		err = magma_dmalloc_cpu ( &c , mm );
 		//double ml[] = {25, 15, -5, 15, 18, 0, -5, 0, 11};
 		double *ml = generate_sym_matrix(m);
 
 		start = get_current_time();
 
 		c = cholesky(ml, m);
 
 		end = get_current_time();
 
 		gpu_time = GetTimerValue(start,end)/1e3;
 
 		printf("gpu time for %dx%d: %7.5f sec\n", m, m, gpu_time);

 		copy_upper_diag(c,m);

 		free(c);
 	}
 
 	magma_finalize ();
 	return 0;
}
