#include <stdio.h>
#include <cuda.h>
#include "Rcpp.h"
#include "magma.h"
#include "magma_lapack.h"

using namespace Rcpp;

#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) < (b)) ? (b) : (a))

double *cholesky(double *X, int m)
{
 
 magma_int_t info;

 magma_dpotrf('L',m,X,m,&info);
 magma_dpotri('L',m,X,m,&info);

 return X;

} 

void copy_upper_diag(double *c, int n) 
{
 
        int i,j;
        for(i=0; i<n; i++) {
        for (j=0; j<i; j++)
                c[i*n+j] = (c[j*n+i]);
        } 

}

//http://stackoverflow.com/questions/26194225/rcpp-returning-c-array-as-numericmatrix-to-r
void cfunction(double *c, int n, double *y)
{
 for(int i = 0; i < n; i++) y[i] = c[i];
}

//[[Rcpp::export]]
NumericMatrix gpu_chol2inv(NumericMatrix X_)
{

 NumericMatrix X(X_);
 magma_init();
 double *c;
 
 magma_int_t n_rows = X.nrow(), n_cols = X.ncol(), m = min(n_rows, n_cols);
 magma_int_t mm = m*m;
 magma_err_t err;
 NumericMatrix y(n_rows, n_cols);
 err = magma_dmalloc_cpu(&c, mm);
 
 c = cholesky(&(X[0]), m);
 copy_upper_diag(c, m);
 cfunction(c, mm, y.begin());
 magma_finalize();

 return wrap(y); 

}



