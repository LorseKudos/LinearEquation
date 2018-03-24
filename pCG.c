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
      Ap[i] = ZERO;
      for(j=IAP[i];j<IAP[i+1];j++){
        Ap[i] = Ap[i] + (A[j])*p[JA[j]]; //A*pの計算
      }
    }
  }
  return;
}

int main(int argc,char *argv[]){
  double *A,*x,*r,*p,*Ap,*b;
  double alpha,beta,r2,norm_b,stop_c,inner_r;
  int *IAP,*JA,i,j,k,l;
  FILE *fp;
  double t1,t2;

  srand(1);
  fp = fopen(argv[1],"r");
  fscanf(fp,"%d",&N);
  fscanf(fp,"%d",&l);

  x = (double *)malloc(sizeof(double)*N);
  r = (double *)malloc(sizeof(double)*N);
  p = (double *)malloc(sizeof(double)*N);
  b = (double *)malloc(sizeof(double)*N);
  Ap = (double *)malloc(sizeof(double)*N);
  IAP = (int *)malloc(sizeof(int)*(N+1));
  JA = (int *)malloc(sizeof(int)*l);
  A = (double *)malloc(sizeof(double)*l);

  if(x==NULL || r==NULL || p==NULL || b==NULL || Ap==NULL || IAP==NULL || JA==NULL || A==NULL){
    printf("Out of memory \n");
    fclose(fp);
    return 0;
  }

  k = 0;
  l = 0;
  IAP[0] = 0;
  while(fscanf(fp,"%d %d %lf",&i,&j,&alpha) != EOF){ //ファイルからAを読み込む
    if(i!=k){
      IAP[k+1] = l;
      k++;
    }
    JA[l] = j;
    A[l] = alpha;
    l++;
  }
  IAP[k+1] = l;
  fclose(fp);

  fp = fopen(argv[2],"r");
  l = 0;
  while(fscanf(fp,"%lf",&alpha) != EOF){ //ファイルからbを読み込む
    b[l] = alpha;
    l++;
  }
  fclose(fp);

  t1 = omp_get_wtime();

  norm_b = ZERO;

#pragma omp parallel for reduction(+:norm_b)
  for(i=0;i<N;i++){
    norm_b = norm_b + b[i]*b[i];
  }
  norm_b = sqrt(norm_b); //bのノルムの計算


  for(i=0;i<N;i++){ //xの初期値
    x[i] = TWO*((double)rand()/RAND_MAX) - ONE;
  }

  for(i=0;i<N;i++){ //xの初期値
    printf("%lf\n",x[i]);
  }

  sparse_matvec(IAP,JA,A,x,Ap); //ApにA*xを計算し代入

#pragma omp parallel for
  for(i=0;i<N;i++){ //rの初期値
    r[i] = b[i] - Ap[i];
  }

  r2 = ZERO;
#pragma omp parallel for reduction(+:r2)
  for(i=0;i<N;i++){ //r2の計算
    r2 = r2 + r[i]*r[i];
  }

#pragma omp parallel for
  for(i=0;i<N;i++){ //pの初期値
    p[i] = r[i];
  }

  for(i=0;;i++){
    sparse_matvec(IAP,JA,A,p,Ap);

    alpha = ZERO;
#pragma omp parallel for reduction(+:alpha)
    for(j=0;j<N;j++){ //alphaの更新
      alpha = alpha + p[j]*Ap[j];
    }
    alpha = r2/alpha;

#pragma omp parallel for
    for(j=0;j<N;j++){ //xの更新
      x[j] = x[j] + alpha*p[j];
    }

#pragma omp parallel for
    for(j=0;j<N;j++){ //rの更新
      r[j] = r[j] - alpha*Ap[j];
    }

    inner_r = ZERO;
#pragma omp parallel for reduction(+:inner_r)
    for(j=0;j<N;j++){ //rの内積の更新
      inner_r = inner_r + r[j]*r[j];
    }

    stop_c = sqrt(inner_r)/norm_b;
    printf("%30.20f\n",stop_c);
    if(stop_c <= CONST) break; //終了判定

    beta = inner_r/r2; //betaの更新

#pragma omp parallel for
    for(j=0;j<N;j++){ //pの更新
      p[j] = r[j] + beta*p[j];
    }

    r2 = inner_r; //r2の更新
  }

  t2 = omp_get_wtime();
  printf("TIME=%10.5g\n",t2-t1);
  printf("LOOP=%d RELATIVE RESIDUAL=%10.5g\n",i,stop_c);

  return 0;
}