#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define TWO (2.0)
#define ONE (1.0)
#define ZERO (0.0)
#define CONST (1.0E-8)

int N;

void sparse_matvec(int *IAP,int *JA,double *A,double *p,double *Ap){
  int i,j;

  for(i=0;i<N;i++){
    Ap[i]=ZERO;
    for(j=IAP[i];j<IAP[i+1];j++){
      Ap[i]=Ap[i]+(A[j])*p[JA[j]]; //A*pの計算
    }
  }
  return;
}

int main(int argc,char *argv[]){
  double *A,*x,*r,*s,*p,*q,*Ap,*Aq,*b;
  double alpha,beta,omega,r2,norm_b,stop_c,inner_r,inner_Aq;
  int *IAP,*JA,i,j,k,l;
  FILE *fp;
  double t1,t2;

  srand(1);
  fp=fopen(argv[1],"r");
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

  if(x==NULL || r==NULL || s==NULL || p==NULL || q==NULL || b==NULL ||Ap==NULL || Aq==NULL || IAP==NULL || JA==NULL || A==NULL){
    printf("Out of memory.\n");
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

  fp=fopen(argv[2],"r");
  l = 0;
  while(fscanf(fp,"%lf",&alpha) != EOF){ //ファイルからbを読み込む
    b[l] = alpha;
    l++;
  }
  fclose(fp);

  t1 = omp_get_wtime();

  norm_b = ZERO;
  for(i=0;i<N;i++){
    norm_b=norm_b+b[i]*b[i];
  }
  norm_b = sqrt(norm_b); //bのノルムの計算

  for(i=0;i<N;i++){ //xの初期値
    x[i] = TWO*((double)rand()/RAND_MAX) - ONE;
  }

  sparse_matvec(IAP,JA,A,x,Ap); //ApにA*xを代入

  for(i=0;i<N;i++){ //rの初期値
    r[i] = b[i] - Ap[i];
  }

  do{
    for(i=0;i<N;i++){
      s[i] = TWO*((double)rand()/RAND_MAX) - ONE;
    }

    r2 = ZERO;
    for(i=0;i<N;i++){
      r2 = r2 + s[i]*r[i];
    }
  }while(r2==ZERO); //sの初期値

  r2 = ZERO;
  for(j=0;j<N;j++){ //rsの内積の計算
    r2 = r2 + s[j]*r[j];
  }

  for(i=0;i<N;i++){ //pの初期値
    p[i] = r[i];
  }

  for(i=0;;i++){
    sparse_matvec(IAP,JA,A,p,Ap); //ApにA*pを代入
    alpha=ZERO;
    for(j=0;j<N;j++){ //alphaの計算
      alpha=alpha+s[j]*Ap[j];
    }
    alpha = r2/alpha;

    for(j=0;j<N;j++){ //qの更新
      q[j] = r[j] - alpha*Ap[j];
    }

    sparse_matvec(IAP,JA,A,q,Aq); //ApにA*qを代入

    omega=ZERO;
    inner_Aq=ZERO;
    for(j=0;j<N;j++){ //alphaの計算
      omega=omega+q[j]*Aq[j];
      inner_Aq=inner_Aq+Aq[j]*Aq[j];
    }
    omega = omega/inner_Aq;

    for(j=0;j<N;j++){ //xの更新
      x[j] = x[j] + alpha*p[j] + omega*q[j];
    }

    for(j=0;j<N;j++){ //rの更新
      r[j] = q[j] - omega*Aq[j];
    }

    inner_r = ZERO;
    for(j=0;j<N;j++){ //rのノルムの計算
      inner_r = inner_r + r[j]*r[j];
    }

    stop_c = sqrt(inner_r)/norm_b;
    if(stop_c<=CONST) break; //終了判定

    inner_r = ZERO;
    for(j=0;j<N;j++){ //rsの内積の計算
      inner_r = inner_r + s[j]*r[j];
    }

    beta = (alpha/omega)*inner_r/r2; //betaの更新

    for(j=0;j<N;j++){ //pの更新
      p[j] = r[j] + beta*(p[j] - omega*Ap[j]);
    }

    r2 = inner_r; //rsの更新
  }

  t2 = omp_get_wtime();
  printf("TIME=%10.5g\n",t2-t1);
  printf("LOOP=%d RELATIVE RESIDUAL=%10.5g\n",i,stop_c);

  return 0;
}