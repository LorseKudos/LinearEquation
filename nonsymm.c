#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ZERO (double)0.0
#define TWO (double)2.0

int main(int argc,char *argv[])
{
  long long i, j, k, n, m, l, z;
  double *a;
  double s;
  char filename[100];
  FILE *fp;

  n=atoi(argv[1]);

  m=atoi(argv[2])/2;

  l=atoi(argv[3]);

  srand(1);

  a=(double *)malloc(sizeof(double)*n*n);

  if(a==NULL){
    printf("out of memory\n");
    return 0;
  }

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      a[i*n+j]=ZERO;
    }
  }

  sprintf(filename,"%s_%s_%s_%s_b",argv[4],argv[1],argv[2],argv[3]);

  fp=fopen(filename,"w");
  for(i=0;i<n;i++){
    fprintf(fp,"%30.20f\n",TWO*(rand()/(double)(RAND_MAX))-1.0);
  }
  fclose(fp);

  for(i=0;i<n*m;i++){
    j=rand() % n;
    k=rand() % n;
    s=(double)rand()/(RAND_MAX);
    a[j*n+k]=s;
    a[k*n+j]=s;
  }

  for(i=0;i<l;i++){
    j=rand() % n;
    k=rand() % n;
    s=(double)rand()/(RAND_MAX);
    a[j*n+k]=s;
  }

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      if(a[i*n+j]!=ZERO)
  break;
    }

    if(j==n){
      k=rand() % n;
      s=(double)rand()/(RAND_MAX);
      a[i*n+k]=s;
      a[k*n+i]=s;
    }
  }

  z=0;
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      if(a[i*n+j]!=ZERO)
        z=z+1;
    }
  }

  sprintf(filename,"%s_%s_%s_%s_A",argv[4],argv[1],argv[2],argv[3]);

  fp=fopen(filename,"w");

  fprintf(fp,"%lld\n",n);

  fprintf(fp,"%lld\n",z);

  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      if(a[i*n+j]!=ZERO){
        fprintf(fp,"%lld %lld %30.20f\n",i,j,a[i*n+j]);
      }
    }
  }
  fclose(fp);

  return 0;

}
