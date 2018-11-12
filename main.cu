#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>  
#include<device_launch_parameters.h> 
#include<time.h>

#include"mmio.h"
#include "anonymouslib_cuda.h"

#ifndef NUM_RUN
#define NUM_RUN 1000
#endif

using namespace std;

typedef struct{
	double *val;
	unsigned int *row;
	unsigned int *col;
	unsigned int nz;
	unsigned int rows;
	unsigned int cols;
}COO_Matrix;

typedef struct{
	double *val;
	unsigned int *ptr;
	unsigned int *col;
	unsigned int *row;
	unsigned int nz;
	unsigned int rows;
	unsigned int cols;
}CSR_Matrix;

void coo_init_matrix(COO_Matrix *m);
void coo_free_matrix(COO_Matrix *m);
void coo_print_matrix(COO_Matrix *m);
void csr_init_matrix(CSR_Matrix *m);
void csr_free_matrix(CSR_Matrix *m);
void csr_print_matrix(CSR_Matrix m);
int coo_load_matrix(char* filename, COO_Matrix *coo);
int csr_load_matrix(char* filename, CSR_Matrix *csr);

int coo_load_matrix(char* filename, COO_Matrix *coo){
	FILE *file;
	MM_typecode code;
	int m, n, nz, i, ival;
	unsigned int dc = 0;

	file = fopen(filename, "r");
	mm_read_banner(file, &code);
	mm_read_mtx_crd_size(file, &m, &n, &nz);

	coo->rows = m;
	coo->cols = n;
	coo->nz = nz;

	coo->val = (double *)malloc(nz*sizeof(double));
	coo->row = (unsigned int *)malloc(nz*sizeof(unsigned int));
	coo->col = (unsigned int *)malloc(nz*sizeof(unsigned int));

	for (i = 0; i<nz; i++){
		if (mm_is_pattern(code)){
			if (fscanf(file, "%d %d\n", &(coo->row[i]), &(coo->col[i]))<2){
				fprintf(stderr, "ERROR1\n");
				exit(EXIT_FAILURE);
			}
			coo->val[i] = 1.0;
		}
		else if (mm_is_real(code)){
			if (fscanf(file, "%d %d %lg\n", &(coo->row[i]), &(coo->col[i]), &(coo->val[i]))<3){
				fprintf(stderr, "ERROR2\n");
				exit(EXIT_FAILURE);
			}
		}
		else if (mm_is_integer(code)){
			if (fscanf(file, "%d %d %d\n", &(coo->row[i]), &(coo->col[i]), &ival)<3){
				fprintf(stderr, "ERROR3\n");
				exit(EXIT_FAILURE);
			}
			coo->val[i] = (double)ival;
		}
		coo->row[i]--;
		coo->col[i]--;
		if (coo->row[i] == coo->col[i]){
			++dc;
		}
	}
	fclose(file);

	return mm_is_symmetric(code);
}

int csr_load_matrix(char* filename, CSR_Matrix *csr){
	FILE *file;
	MM_typecode code;
	int m, n, nz, i, ival;
	int issymmetric = 0;
	unsigned int dc = 0;

	if (mm_is_symmetric(code) || mm_is_hermitian(code)){
		issymmetric = 1;
	}

	file = fopen(filename, "r");
	mm_read_banner(file, &code);
	mm_read_mtx_crd_size(file, &m, &n, &nz);

	csr->rows = m;
	csr->cols = n;
	csr->nz = nz;

	unsigned int *csrptr = (unsigned int *)malloc((m + 1) * sizeof(unsigned int));
	memset(csrptr, 0, (m + 1) * sizeof(unsigned int));

	unsigned int *csrrow = (unsigned int *)malloc(nz * sizeof(unsigned int));
	unsigned int *csrcol = (unsigned int *)malloc(nz * sizeof(unsigned int));
	double *csrval = (double *)malloc(nz * sizeof(double));

	for (i = 0; i<nz; i++){
		if (mm_is_pattern(code)){
			if (fscanf(file, "%d %d\n", &(csrrow[i]), &(csrcol[i]))<2){
				fprintf(stderr, "ERROR1\n");
				exit(EXIT_FAILURE);
			}
			csrval[i] = 1.0;
		}
		else if (mm_is_real(code)){
			if (fscanf(file, "%d %d %lg\n", &(csrrow[i]), &(csrcol[i]), &(csrval[i]))<3){
				fprintf(stderr, "ERROR2\n");
				exit(EXIT_FAILURE);
			}
		}
		else if (mm_is_integer(code)){
			if (fscanf(file, "%d %d %d\n", &(csrrow[i]), &(csrcol[i]), &ival)<3){
				fprintf(stderr, "ERROR3\n");
				exit(EXIT_FAILURE);
			}
			csrval[i] = (double)ival;
		}
		csrrow[i]--;
		csrcol[i]--;
		csrptr[csrrow[i]]++;
	}
	fclose(file);

	if (issymmetric){
		for (int i = 0; i < nz; i++){
			if (csrrow[i] != csrcol[i])
				csrptr[csrcol[i]]++;
		}
	}

	double old_val, new_val;
	old_val = csrptr[0];
	csrptr[0] = 0;
	for (int i = 1; i <= m; i++){
		new_val = csrptr[i];
		csrptr[i] = old_val + csrptr[i - 1];
		old_val = new_val;
	}
	nz = csrptr[m];
	csr->val = (double *)malloc(nz*sizeof(double));
	csr->ptr = (unsigned int *)malloc((m + 1)*sizeof(unsigned int));
	memcpy(csr->ptr, csrptr, (m + 1)*sizeof(unsigned int));
	memset(csrptr, 0, (m + 1) * sizeof(unsigned int));
	csr->col = (unsigned int *)malloc(nz*sizeof(unsigned int));

	if (issymmetric){
		for (int i = 0; i < nz; i++){
			if (csrrow[i] != csrcol[i]){
				int offset = csr->ptr[csrrow[i]] + csrptr[csrrow[i]];
				csr->col[offset] = csrcol[i];
				csr->val[offset] = csrval[i];
				csrptr[csrrow[i]]++;

				offset = csr->row[csrcol[i]] + csrptr[csrcol[i]];
				csr->col[offset] = csrrow[i];
				csr->val[offset] = csrval[i];
				csrptr[csrcol[i]]++;
			}
			else{
				int offset = csr->ptr[csrrow[i]] + csrptr[csrrow[i]];
				csr->col[offset] = csrcol[i];
				csr->val[offset] = csrval[i];
				csrptr[csrrow[i]]++;
			}
		}
	}
	else{
		for (int i = 0; i < nz; i++){
			int offset = csr->ptr[csrrow[i]] + csrptr[csrrow[i]];
			csr->col[offset] = csrcol[i];
			csr->val[offset] = csrval[i];
			csrptr[csrrow[i]]++;
		}
	}

	return mm_is_symmetric(code);
}

void coo_print_matrix(COO_Matrix *m){
	unsigned int i;
	printf("val= ");
	for (i = 0; i<m->nz; i++){
		printf("%.2g ", m->val[i]);
	}
	printf("\nrow= ");
	for (i = 0; i<m->nz; i++){
		printf("%d ", m->row[i]);
	}
	printf("\ncol= ");
	for (i = 0; i<m->nz; i++){
		printf("%d ", m->col[i]);
	}
	printf("\nnz= %d\n", m->nz);
	printf("rows= %d\n", m->rows);
	printf("cols= %d\n", m->cols);
}

void csr_print_matrix(CSR_Matrix *mcsr){
	unsigned int i;
	printf("val= ");
	for (i = 0; i<mcsr->nz; i++){
		printf("%.2g ", mcsr->val[i]);
	}
	printf("\nptr= ");
	for (i = 0; i<(mcsr->rows) + 1; i++){
		printf("%d ", mcsr->ptr[i]);
	}
	printf("\ncol= ");
	for (i = 0; i<mcsr->nz; i++){
		printf("%d ", mcsr->col[i]);
	}
	printf("\nnz= %d\n", mcsr->nz);
	printf("rows= %d\n", mcsr->rows);
	printf("cols= %d\n", mcsr->cols);
}

void coo_init_matrix(COO_Matrix *m){
	m->val = NULL;
	m->row = NULL;
	m->col = NULL;
	m->nz = m->rows = m->cols = 0;
}

void csr_init_matrix(CSR_Matrix *m){
	m->val = NULL;
	m->row = NULL;
	m->col = NULL;
	m->ptr = NULL;
	m->nz = m->rows = m->cols = 0;
}

void coo_free_matrix(COO_Matrix *m){
	if (m->val != NULL) free(m->val);
	if (m->row != NULL) free(m->row);
	if (m->col != NULL) free(m->col);
	m->val = NULL;
	m->row = NULL;
	m->col = NULL;
	m->nz = m->rows = m->cols = 0;
}

void csr_free_matrix(CSR_Matrix *m){
	if (m->val != NULL) free(m->val);
	if (m->row != NULL) free(m->row);
	if (m->col != NULL) free(m->col);
	if (m->ptr != NULL) free(m->ptr);
	m->val = NULL;
	m->row = NULL;
	m->col = NULL;
	m->ptr = NULL;
	m->nz = m->rows = m->cols = 0;
}

//GPU with one thread per row
cudaError_t addWithCuda(CSR_Matrix *m, double *x, double *y);

__global__ void spmv_csr_scalar(unsigned int rows, unsigned int *ptr, unsigned int *col, double *val, double *x, double *y){
	int row = blockDim.x*blockIdx.x + threadIdx.x;

	if (row < rows){
		int dot = 0;

		int row_start = ptr[row];
		int row_end = ptr[row + 1];

		for (int jj = row_start; jj < row_end; jj++)
			dot += val[jj] * x[col[jj]];
			//dot += val[jj] * tryfetch_x<double>(col[jj],x);

		y[row] += dot;
	}
}

cudaError_t addWithCuda(CSR_Matrix *m, double *x, double *y){
	unsigned int *dev_ptr = 0;
	unsigned int *dev_col = 0;
	double *dev_val = 0;
	double *dev_x = 0;
	double *dev_y = 0;
	double threadtime = 0;
	double threadmalloctime=0;
	cudaError_t cudaStatus;

	double gb=getB<int,double>(m->rows,m->nz);
	double gflop=getFLOP<int>(m->nz);

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	anonymouslib_timer threadmalloc;
	threadmalloc.start();
	cudaStatus = cudaMalloc((void**)&dev_ptr, (m->rows + 1) * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_col, m->nz * sizeof(unsigned int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_val, m->nz * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_x, m->cols* sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_y, m->rows * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.  
	cudaStatus = cudaMemcpy(dev_ptr, m->ptr, (m->rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_col, m->col, m->nz * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_val, m->val, m->nz * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_x, x, m->cols * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	threadmalloctime = threadmalloc.stop();
	cout << "CSR-scalar malloc time = " << threadmalloctime << " ms." << endl;

	anonymouslib_timer threadperrow;
	threadperrow.start();
	// Launch a kernel on the GPU with one thread for each element. 
	spmv_csr_scalar << <8, 128 >> >(m->rows, dev_ptr, dev_col, dev_val, dev_x, dev_y);
	threadtime = threadperrow.stop();
	cout << "CSR-scalar spmv time = " << threadtime << " ms." << endl;

	// cudaThreadSynchronize waits for the kernel to finish, and returns  
	// any errors encountered during the launch.  
	cudaStatus = cudaThreadSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.  
	cudaStatus = cudaMemcpy(y, dev_y, m->rows * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cout<<"Bandwidth="<<gb/(1.0e+6*threadtime)<<"GB/s."<<endl;
	cout<<"GFlops="<<gflop/(1.0e+6*threadtime)<<"GFlops."<<endl;

Error:
	cudaFree(dev_ptr);
	cudaFree(dev_col);
	cudaFree(dev_val);
	cudaFree(dev_x);
	cudaFree(dev_y);

	return cudaStatus;
}

//32-thread warp per row
cudaError_t addWithCuda1(CSR_Matrix *m, double *x, double *y);

__global__ void spmv_csr_warp(unsigned int rows, unsigned int *ptr, unsigned int *col, double *val, double *x, double *y){
	__shared__ double vals[1024];

	int thread_id = 128*blockIdx.x + threadIdx.x;
	int warp_id = thread_id / 32;
	int lane = thread_id&(32 - 1);
	//int lane = threadIdx.x&(32 - 1);

	int row = warp_id;

	if (row < rows){
		int row_start = ptr[row];
		int row_end = ptr[row + 1];

		vals[threadIdx.x] = 0;
		for (int jj = row_start+lane; jj < row_end; jj+=32)
			vals[threadIdx.x] += val[jj] * x[col[jj]];

		if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
		if (lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8];
		if (lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4];
		if (lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2];
		if (lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1];

		if (lane==0)
			y[row] += vals[threadIdx.x];
	}
}

cudaError_t addWithCuda1(CSR_Matrix *m, double *x, double *y){
	unsigned int *dev_ptr = 0;
	unsigned int *dev_col = 0;
	double *dev_val = 0;
	double *dev_x = 0;
	double *dev_y = 0;
	double warptime = 0;
	double csrvectorspmv1=0;
	double csrvectormalloc1=0;
	cudaError_t cudaStatus1;

	double gb=getB<int,double>(m->rows,m->nz);
	double gflop=getFLOP<int>(m->nz);

	cudaStatus1 = cudaSetDevice(0);
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	anonymouslib_timer csrvectormalloc;
	csrvectormalloc.start();
	cudaStatus1 = cudaMalloc((void**)&dev_ptr, (m->rows + 1) * sizeof(unsigned int));
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus1 = cudaMalloc((void**)&dev_col, m->nz * sizeof(unsigned int));
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus1 = cudaMalloc((void**)&dev_val, m->nz * sizeof(double));
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus1 = cudaMalloc((void**)&dev_x, m->cols* sizeof(double));
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus1 = cudaMalloc((void**)&dev_y, m->rows * sizeof(double));
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.  
	cudaStatus1 = cudaMemcpy(dev_ptr, m->ptr, (m->rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus1 = cudaMemcpy(dev_col, m->col, m->nz * sizeof(unsigned int), cudaMemcpyHostToDevice);
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus1 = cudaMemcpy(dev_val, m->val, m->nz * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus1 = cudaMemcpy(dev_x, x, m->cols * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	csrvectormalloc1=csrvectormalloc.stop();
	//cout<<"CSR-vector malloc time="<<csrvectormalloc1<<" ms."<<endl;
	
	anonymouslib_timer csrvectorspmv;
	csrvectorspmv.start();
	// Launch a kernel on the GPU with one thread for each element.  
	spmv_csr_warp << <8, 128 >> >(m->rows, dev_ptr, dev_col, dev_val, dev_x, dev_y);
	csrvectorspmv1=csrvectorspmv.stop();
	//cout<<"CSR-vector spmv time="<<csrvectorspmv1<<" ms."<<endl;
	

	// cudaThreadSynchronize waits for the kernel to finish, and returns  
	// any errors encountered during the launch.  
	cudaStatus1 = cudaThreadSynchronize();
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaThreadSynchronize returned error code %d after launching addKernel!\n", cudaStatus1);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.  
	cudaStatus1 = cudaMemcpy(y, dev_y, m->rows * sizeof(double), cudaMemcpyDeviceToHost);
	if (cudaStatus1 != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//cout<<"Bandwidth="<<gb/(1.0e+6*csrvectorspmv1)<<"GB/s."<<endl;
	//cout<<"GFlops="<<gflop/(1.0e+6*csrvectorspmv1)<<"GFlops."<<endl;

Error:
	cudaFree(dev_ptr);
	cudaFree(dev_col);
	cudaFree(dev_val);
	cudaFree(dev_x);
	cudaFree(dev_y);

	return cudaStatus1;
}

int call_anonymouslib(CSR_Matrix *csr, double *x, double *y){
	// set device
	cudaSetDevice(0);
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	//cout << "Device [" << 0 << "] " << deviceProp.name << ", " << " @ " << deviceProp.clockRate * 1e-3f << "MHz. " << endl;

	double gb=getB<int,double>(csr->rows,csr->nz);
	double gflop=getFLOP<int>(csr->nz);	

	// Define pointers of matrix A, vector x and y
	unsigned int *dev_ptr = 0;
	unsigned int *dev_col = 0;
	double *dev_val = 0;
	double *dev_x = 0;
	double *dev_y = 0;
	cudaError_t cudaStatus2;

	anonymouslib_timer csr5malloc;
	csr5malloc.start();
	// Matrix A
	cudaStatus2 = cudaMalloc((void**)&dev_ptr, (csr->rows + 1) * sizeof(unsigned int));
	cudaStatus2 = cudaMalloc((void**)&dev_col, csr->nz * sizeof(unsigned int));
	cudaStatus2 = cudaMalloc((void**)&dev_val, csr->nz * sizeof(double));

	cudaStatus2 = cudaMemcpy(dev_ptr, csr->ptr, (csr->rows + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaStatus2 = cudaMemcpy(dev_col, csr->col, csr->nz * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaStatus2 = cudaMemcpy(dev_val, csr->val, csr->nz * sizeof(double), cudaMemcpyHostToDevice);

	// Vector x
	cudaStatus2 = cudaMalloc((void**)&dev_x, csr->cols* sizeof(double));
	cudaStatus2 = cudaMemcpy(dev_x, x, csr->cols * sizeof(double), cudaMemcpyHostToDevice);

	// Vector y
	cudaStatus2 = cudaMalloc((void**)&dev_y, csr->rows * sizeof(double));
	double csr5malloc1=csr5malloc.stop();
	cout<<"CSR5 malloc time="<<csr5malloc1<<" ms."<<endl;

	anonymouslibHandle<unsigned int, unsigned int, double> A(csr->rows, csr->cols);

	anonymouslib_timer csr5input;
	csr5input.start();
	A.inputCSR(csr->nz, dev_ptr, dev_col, dev_val);
	double csr5input1=csr5input.stop();
	cout<<"CSR5 input time="<<csr5input1<<" ms."<<endl;

	anonymouslib_timer csr5setx;
	csr5setx.start();
	A.setX(dev_x); 
	//cout << "setX err = " << err << endl;
	double csr5setx1=csr5setx.stop();
	cout<<"CSR5 setX time="<<csr5setx1<<" ms."<<endl;

	A.setSigma(ANONYMOUSLIB_AUTO_TUNED_SIGMA);

	anonymouslib_timer ascsr5;
	ascsr5.start();
	A.asCSR5();
	//cout << "ascsr5 err = " << err << endl;
	double ascsr51=ascsr5.stop();
	cout<<"CSR5 ascsr5 time="<<ascsr51<<" ms."<<endl;

	anonymouslib_timer csr5spmvtime;
	csr5spmvtime.start();
	A.spmv(1, dev_y);
	//cout << "spmv err = " << err << endl;
	double csr5spmvtime1=csr5spmvtime.stop();
	cout<<"CSR5 SPMV times="<<csr5spmvtime1<<" ms."<<endl;

	if(NUM_RUN){
		for(int i=0;i<50;i++)
			A.spmv(1,dev_y);
	}

	cudaDeviceSynchronize();

	anonymouslib_timer csr5spmvtimes;
	csr5spmvtimes.start();

	for(int i=0;i<NUM_RUN;i++)
		A.spmv(1,dev_y);
	cudaDeviceSynchronize();

	double csr5spmvtimes1=csr5spmvtimes.stop()/(double)NUM_RUN;
	cout<<"CSR5 SPMV time="<<csr5spmvtimes1<<" ms."<<endl;

	cudaStatus2 = cudaDeviceSynchronize();
	cudaStatus2 = cudaMemcpy(y, dev_y, csr->rows * sizeof(double), cudaMemcpyDeviceToHost);

	cout<<"Bandwidth="<<gb/(1.0e+6*csr5spmvtimes1)<<"GB/s."<<endl;
	cout<<"GFlops="<<gflop/(1.0e+6*csr5spmvtimes1)<<"GFlops."<<endl;

	A.destroy();

	cudaFree(dev_ptr);
	cudaFree(dev_col);
	cudaFree(dev_val);
	cudaFree(dev_x);
	cudaFree(dev_y);

	return cudaStatus2;
}

void spmv_serial(CSR_Matrix *m, double *x, double *y){
	for (int row = 0; row < m->rows; row++){
		double dot = 0;
		int row_start = m->ptr[row];
		int row_end = m->ptr[row + 1];

		for (int i = row_start; i < row_end; i++)
			dot += m->val[i] * x[m->col[i]];

		y[row] = dot;
	}
}


int main(){
	COO_Matrix coo;
	CSR_Matrix csr;

	coo_init_matrix(&coo);
	coo_load_matrix("webbase-1M.mtx", &coo);
	//coo_print_matrix(&coo);
	coo_free_matrix(&coo);

	csr_init_matrix(&csr);
	csr_load_matrix("webbase-1M.mtx", &csr);
	//csr_print_matrix(&csr);

	double *x,*y;
	x = (double *)malloc(csr.cols*sizeof(double));
	y = (double *)malloc(csr.rows*sizeof(double));
	for (int i = 0; i<csr.cols; i++)
		x[i] = 1;
	for (int i = 0; i<csr.rows; i++)
		y[i] = 0;

	double *x1,*y1;
	x1 = (double *)malloc(csr.cols*sizeof(double));
	y1 = (double *)malloc(csr.rows*sizeof(double));
	for (int i = 0; i<csr.cols; i++)
		x1[i] = 1;
	for (int i = 0; i<csr.rows; i++)
		y1[i] = 0;

	double *x2,*y2;
	x2 = (double *)malloc(csr.cols*sizeof(double));
	y2 = (double *)malloc(csr.rows*sizeof(double));
	for (int i = 0; i<csr.cols; i++)
		x2[i] = 2;
	for (int i = 0; i<csr.rows; i++)
		y2[i] = 0;

	double *x3, *y3;
	x3 = (double *)malloc(csr.cols*sizeof(double));
	y3 = (double *)malloc(csr.rows*sizeof(double));
	for (int i = 0; i<csr.cols; i++)
		x3[i] = 2;
	for (int i = 0; i<csr.rows; i++)
		y3[i] = 0;

	anonymouslib_timer serial;
	serial.start();
	spmv_serial(&csr, x, y);
	double serialtime = serial.stop();
	cout << "serial time = " << serialtime << " ms." << endl;

	anonymouslib_timer threadperrow;
	threadperrow.start();
	addWithCuda(&csr, x, y3);
	double threadtime=threadperrow.stop();
	cout << "thread time = " << threadtime << " ms." << endl;

	anonymouslib_timer warpperrow;
	warpperrow.start();
	addWithCuda1(&csr, x, y1);
	double warptime=warpperrow.stop();
	cout << "warp time = " << warptime << " ms." << endl;

	//spmv 1000 times.
	//anonymouslib_timer csr5;
	//csr5.start();
	call_anonymouslib(&csr, x, y2);
	//double csr5time=csr5.stop();
	//cout << "csr5 time = " << csr5time << " ms." << endl;

	/*for (int i = 0; i < csr.cols; i++){
		printf("%.2g\n", x[i]);
		printf("%.2g\n", y[i]);
	}

	for (int i = 0; i < csr.cols; i++){
		printf("%.2g\n", x3[i]);
		printf("%.2g\n", y3[i]);
	}
	for (int i = 0; i < csr.cols; i++){
		printf("%.2g\n", x1[i]);
		printf("%.2g\n", y1[i]);
	}
	for (int i = 0; i < csr.cols; i++){
		printf("%.2g\n", x2[i]);
		printf("%.2g\n", y2[i]);
	}*/



}