#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

#define TWO (2.0)
#define ONE (1.0)
#define ZERO (0.0)
#define CONST (1.0E-8)

int N;

void sparse_matvec(int *IAP,int *JA,double *A,double *p,double *Ap){
  int i,j;

#pragma omp parallel private(j)
  {
  #pragma omp for
    for(i=0;i<N;i++){
      Ap[i]=ZERO;
      for(j=IAP[i];j<IAP[i+1];j++){
        Ap[i]=Ap[i]+(A[j])*p[JA[j]]; //A*pの計算
      }
    }
  }
  return;
}

void sparse_matvec3(int *IAP,int *JA,double *A,double *p,double *Ap,double *q,double *Aq){
  int i,j;
  double *Aq_tmp;

#pragma omp parallel private(j,Aq_tmp)
  {
    Aq_tmp=(double *)malloc(sizeof(double)*N);
    if(Aq_tmp==NULL){
      printf("out of memory\n");
    }

    for(i=0;i<N;i++){
      Aq_tmp[i]=ZERO;
    }

#pragma omp for nowait
    for(i=0;i<N;i++){
      Ap[i]=ZERO;
      for(j=IAP[i];j<IAP[i+1];j++){
        Ap[i]=Ap[i]+(A[j])*p[JA[j]]; //A*pの計算
        Aq_tmp[JA[j]]=Aq_tmp[JA[j]]+(A[j])*q[i]; //A^top*qの計算
    }
  }

#pragma omp for
    for(i=0;i<N;i++){
      Aq[i]=ZERO;
    }

#pragma omp critical
    {
      for(i=0;i<N;i++){
        Aq[i] = Aq[i] + Aq_tmp[i];
      }
    }

    free(Aq_tmp);
  }

  return;
}

int main(int argc,char *argv[]){
  double *A,*x,*r,*s,*p,*q,*Ap,*Aq,*b;
  double alpha,beta=0,rs,norm_b,norm_r,stop_c,inner_rs;
  int *IAP,*JA,i,j,k,l;
  FILE *fp;
  double t1,t2;

  srand(1);
  fp = fopen(argv[1],"r");
  fscanf(fp,"%d",&N);
  fscanf(fp,"%d",&l);

  x = (double *)malloc(sizeof(double)*N);
  r = (double *)malloc(sizeof(double)*N);
  s = (double *)malloc(sizeof(double)*N);
  p = (double *)malloc(sizeof(double)*N);
  q = (double *)malloc(sizeof(double)*N);
  b = (double *)malloc(sizeof(double)*N);
  Ap = (double *)malloc(sizeof(double)*N);
  Aq = (double *)malloc(sizeof(double)*N);
  IAP = (int *)malloc(sizeof(int)*(N+1));
  JA = (int *)malloc(sizeof(int)*l);
  A = (double *)malloc(sizeof(double)*l);

  if(x==NULL || r==NULL || s==NULL || p==NULL || q==NULL || b==NULL || Ap==NULL || IAP==NULL || JA==NULL || A==NULL){
    printf("Out of memory \n");
    fclose(fp);
    return 0;
  }

  k=0;
  l=0;
  IAP[0]=0;
  while(fscanf(fp,"%d %d %lf",&i,&j,&alpha) !=EOF){ //ファイルからAを読み込む
    if(i!=k){
      IAP[k+1]=l;
      k++;
    }
    JA[l]=j;
    A[l]=alpha;
    l++;
  }
  IAP[k+1]=l;
  fclose(fp);

  fp = fopen(argv[2],"r");
  l = 0;
  while(fscanf(fp,"%lf",&alpha) != EOF){ //ファイルからbを読み込む
    b[l] = alpha;
    l++;
  }
  fclose(fp);

  t1=omp_get_wtime();

  norm_b = ZERO;

#pragma omp parallel for reduction(+:norm_b)
  for(i=0;i<N;i++){
    norm_b = norm_b + b[i]*b[i];
  }
  norm_b = sqrt(norm_b); //bのノルムの計算

  #pragma omp parallel for
  for(i=0;i<N;i++){ //xの初期値
    x[i] = TWO*((double)rand()/RAND_MAX) - ONE;
  }

  sparse_matvec(IAP,JA,A,x,Ap); //ApにA*xを計算し代入

  #pragma omp parallel for
  for(i=0;i<N;i++){ //rの初期値
    r[i] = b[i] - Ap[i];
  }

  do{
  #pragma omp parallel for
    for(i=0;i<N;i++){ //sの初期値
      s[i] = TWO*((double)rand()/RAND_MAX) - ONE;
    }
    rs = ZERO;
    #pragma omp parallel for reduction(+:rs)
    for(i=0;i<N;i++){ //rsの内積
      rs = rs + s[i]*r[i];
    }
  }while(rs == ZERO);

#pragma omp parallel for
  for(j=0;j<N;j++){ //p,qの初期値
    p[j] = r[j] + beta*p[j];
    q[j] = s[j] + beta*q[j];
  }

  for(i=0;;i++){
    sparse_matvec3(IAP,JA,A,p,Ap,q,Aq); //ApにA*p,AqにA^top*qを代入

    alpha = ZERO;
#pragma omp parallel for reduction(+:alpha)
    for(j=0;j<N;j++){ //alphaの計算
      alpha = alpha + q[j]*Ap[j];
    }
    alpha = rs/alpha;

#pragma omp parallel for
    for(j=0;j<N;j++){ //xの更新
      x[j] = x[j] + alpha*p[j];
    }

#pragma omp parallel for
    for(j=0;j<N;j++){ //r,sの更新
      r[j] = r[j] - alpha*Ap[j];
      s[j] = s[j] - alpha*Aq[j];
    }

    inner_rs = ZERO;
    norm_r = ZERO;
#pragma omp parallel for reduction(+:norm_r)
    for(j=0;j<N;j++){ //rのノルムの更新
      norm_r = norm_r + r[j]*r[j];
    }

#pragma omp parallel for reduction(+:inner_rs)
    for(j=0;j<N;j++){ //rsの内積の更新
      inner_rs = inner_rs + s[j]*r[j];
    }

    stop_c = sqrt(norm_r)/norm_b;
    if(stop_c <= CONST) break; //終了判定

    beta = inner_rs/rs; //betaの計算

#pragma omp parallel for
    for(j=0;j<N;j++){ //p,qの更新
      p[j] = r[j] + beta*p[j];
      q[j] = s[j] + beta*q[j];
    }

    rs = inner_rs; //rsの更新
  }

  t2 = omp_get_wtime();
  printf("TIME=%10.5g\n",t2-t1);
  printf("LOOP=%d RELATIVE RESIDUAL=%10.5g\n",i,stop_c);

  return 0;
}