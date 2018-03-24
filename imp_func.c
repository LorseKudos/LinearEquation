#include <stdio.h>
#include <math.h>
#include <omp.h>

#define DIV 100 //分割数
#define MAXITER 10 //最大反復数

#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define THREE 3.0
#define FOUR 4.0
#define EIGHT 8.0
#define EPS 1.0E-15

double fxy(double x,double y){ //描画する陰関数
  return x*x*x*x-FOUR*x*x+y*y;
}

double dfx(double x,double y){ //陰関数のx微分
  return FOUR*x*x*x-EIGHT*x;
}

double dfy(double x,double y){ //陰関数のy微分
  return TWO*y;
}

double newtonx(double x,double y){ //x軸方向のニュートン法
  int i;
  double x2;
  for(i=0;i<MAXITER;i++){
    x2 = x - fxy(x,y)/dfx(x,y);
    if(fabs(x2-x) <= EPS){
      return x2;
    }
    if(fabs(x2) > TWO){
      return x2;
    }
    x = x2;
  }
  return -THREE;
}

double newtony(double x,double y){ //y軸方向のニュートン法
  int i;
  double y2;
  for(i=0;i<MAXITER;i++){
    y2 = y - fxy(x,y)/dfy(x,y);
    if(fabs(y2-y) <= EPS){
      return y2;
    }
    if(fabs(y2) > TWO){
      return y2;
    }
    y = y2;
  }
  return -THREE;
}

int main(void){
  int i,j;
  double unit,x,y;
  FILE *fp;
  char filename[10];
  unit = TWO/DIV; //分割幅
  double t1,t2;

  t1 = omp_get_wtime();

  sprintf(filename,"ans_%d",0);
  fp = fopen(filename,"w");

  for(i=-DIV;i<=DIV;i++){
    for(j=-DIV;j<=DIV;j++){
      x = newtonx(unit*j,unit*i);
      if(fabs(x) <= TWO){
        fprintf(fp,"%f %f\n",x,unit*i);
      }
      y = newtony(unit*i,unit*j);
      if(fabs(y) <= TWO){
        fprintf(fp,"%f %f\n",unit*i,y);
      }
    }
  }
  fclose(fp);
  t2 = omp_get_wtime();
  printf("TIME=%10.5g\n",t2-t1);

  return 0;
}