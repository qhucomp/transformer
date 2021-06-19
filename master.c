/*************************************************************************
	> File Name: convolution_forward.c
	> Author: 
	> Mail: 
	> Created Time: 
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <athread.h>

#include "args.h"
#include "util.h"

// extern void SLAVE_FUN(par_multihead_attn)(); //declare slave parallel method

static void _local_gemm_rcr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K)
{
    for(int i = 0;i < M; i ++)
        for(int j = 0; j < N; j ++)
            for(int k = 0; k < K; k ++)
                C[i*LDC+j] += A[i*LDA+k]*B[k+j*LDB];
}

static void _local_gemm_rrr(const float* A, const int LDA, const float* B, const int LDB, float* C, const int LDC, int M, int N, int K)
{
    for(int i = 0;i < M; i ++)
        for(int j = 0; j < N; j ++)
            for(int k = 0; k < K; k ++)
                C[i*LDC+j] += A[i*LDA+k]*B[k*LDB+j];
}



static void _local_trans_head(float* src, float* dst, int B, int S, int D, int N)
{
    int pD = D/N;
#define SRC(b, s, d) src[b*S*D+s*D+d]
#define DST(b, n, s, pd) dst[b*N*S*pD + n*S*pD + s*pD + pd]
    for(int b = 0; b < B; b ++)
        for(int n = 0; n < N; n ++)
            for(int s = 0; s < S; s ++)
                for(int pd = 0; pd < pD; pd ++)
                    DST(b,n,s,pd) = SRC(b,s,n*pD+pd);
}

static void _local_trans_head_back(float* src, float* dst, int B, int S, int D, int N)
{
    int pD = D/N;
#define D3(b, s, d) dst[b*S*D+s*D+d]
#define D4(b, n, s, pd) src[b*N*S*pD + n*S*pD + s*pD + pd]
    for(int b = 0; b < B; b ++)
        for(int n = 0; n < N; n ++)
            for(int s = 0; s < S; s ++)
                for(int pd = 0; pd < pD; pd ++)
					D3(b,s,n*pD+pd) = D4(b,n,s,pd);
}


static void _local_norm(float* buf, int len)
{
	double sum = 0.0f;
	for(int i = 0;i < len; i ++)
		sum += buf[i];
	for(int i = 0;i < len;i ++)
		buf[i] /= sum;
}

static void _print_buf(float* buf, int len, const char* name)
{
	printf("====%s\n", name);
	for(int i = 0; i < 10 && i < len; i ++)
		printf("%f ", buf[i]);
	printf("\n");
}
struct gemm_arg {
    float* A;
    int LDA;
    float* B;
    int LDB;
    float* C;
    int LDC;
    int M;
    int N;
    int K;
}
void fill_gemm_arg(struct gemm_arg *p,float* A, int LDA, float* B, int LDB, float* C, int LDC, int M, int N, int K) {
    p->A = A;
    p->LDA = LDA;
    p->B = B;
    p->LDB = LDB;
    P->C = C;
    p->LDC = LDC;
    p->M = M;
    p->K = K;
    p->N = N;
}
void calc_gemm(struct *p) {
    _local_gemm_rcr(p->A,p->LDA,p->B,p->LDB,p->C,p->LDC,p->M,p->K,p->N);
}

int naive_multihead_attention(Args_t arg)
{
    const int B = arg->B;
    const int S = arg->S;
    const int D = arg->D;
    const int N = arg->N;
    const float* x = arg->x;
    const float* w = arg->w;
    float* Q = arg->Q;
    float* K = arg->K;
    float* V = arg->V;
    float* QK = arg->QK;
    float* y = arg->y;
	const int PD = D/N;
    memset(Q, 0, sizeof(float)*B*S*D);
    memset(K, 0, sizeof(float)*B*S*D);
    memset(V, 0, sizeof(float)*B*S*D);
    memset(QK, 0, sizeof(float)*B*N*S*S);
    memset(y, 0, sizeof(float)*B*S*D);
	float* QN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	float* KN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
	float* VN = (float*)aligned_malloc(sizeof(float)*B*N*S*PD, 128);
    //calculate Q, K, V
    struct gemm_arg *arg = (struct gemm_arg)aligned_malloc(3*B*sizeof(gemm_arg));
    for(int b = 0; b < B; b ++)
    {
        fill_gemm_arg(arg + b,x+b*S*D, D, w, D, Q+b*S*D, D, S, D, D);

        athread_spawn(calc_gemm, arg + b); // spawn
        athread_join(); // wait for all slave threads finished

        fill_gemm_arg(arg + b + B,x+b*S*D, D, w+D*D, D, K+b*S*D, D, S, D, D);
        athread_spawn(calc_gemm, arg + b + B); // spawn
        athread_join(); // wait for all slave threads finished
        fill_gemm_arg(arg + b + 2*B,x+b*S*D, D, w+2*D*D, D, V+b*S*D, D, S, D, D);
        athread_spawn(calc_gemm, arg + b + 2*B); // spawn
        athread_join(); // wait for all slave threads finished

    }
    _local_trans_head(Q, QN, B, S, D, N);
    _local_trans_head(K, KN, B, S, D, N);
    _local_trans_head(V, VN, B, S, D, N);
#define NI(b,n,s,pd) ((((b)*N+n)*S+s)*PD+pd)
#define QKI(b,n,sh,sl) ((((b)*N+n)*S+sh)*S+sl)
	// QK = Q*KT
	for(int b = 0; b < B; b ++)
		for(int n = 0; n < N; n ++)
			_local_gemm_rcr(QN+NI(b,n,0,0), PD, KN+NI(b,n,0,0), PD, QK+QKI(b,n,0,0), S, S, S, PD);

	double norm = sqrt(PD*1.0);
	for(int i = 0; i < B*N*S*S; i ++)
		QK[i] /= norm;
	for(int b = 0; b < B; b ++)
		for(int n = 0; n < N; n ++)
			for(int s = 0; s < S; s ++)
				_local_norm(QK+QKI(b,n,s,0), S);

	// reuse Q
	memset(QN, 0, sizeof(float)*B*S*D);
	for(int b = 0; b < B; b ++)
		for(int n = 0; n < N; n ++)
			_local_gemm_rrr(QK+QKI(b,n,0,0), S, VN+NI(b,n,0,0), PD, QN+NI(b,n,0,0), PD, S, PD, S);
    //trans back
	_local_trans_head_back(QN, y, B, S, D, N);
    
	aligned_free(QN);
	aligned_free(KN);
	aligned_free(VN);
    return 0;
}

int multihead_attention(Args_t arg)
{
    athread_spawn(par_multihead_attn, arg); // spawn
    athread_join(); // wait for all slave threads finished
    return 0;
}

