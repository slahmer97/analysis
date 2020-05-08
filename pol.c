#include <stdio.h>

double regress(double x) {
  double terms[] = {
     1.1906207901367749e-001,
    -9.1103319111047951e-003,
     6.0626160696506065e-004
};
  
  size_t csz = sizeof terms / sizeof *terms;
  
  double t = 1;
  double r = 0;
  for (int i = 0; i < csz;i++) {
    r += terms[i] * t;
    t *= x;
  }
  return r;
}

void main(){
	printf("%f\n",regress(300));
}

