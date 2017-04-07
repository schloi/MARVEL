
#include <math.h>
#include "stats.h"

#define SQR(a) ((a)*(a))

static double inverfc(double p) 
{
  double x, err, t, pp;

  if (p >= 2.) return -100.;
  if (p <= 0.0) return 100.;

  pp=(p < 1.0)? p:2.-p;
  t=sqrt(-2.*log(pp/2));

  x= -0.70711*((2.30753+t*0.27061)/(1+t*(0.99229+t*0.04481))-t);

  int j;
  for (j=0;j<2;j++) 
  {
    err=erfc(x)-pp;
    x += err/(1.12837916709551257*exp(-x*x)-x*err);
  }
  
  return (p<1.0 ? x:-x);
}

double ln_invcdf(double p, double mu, double sig)
{
    return exp(-1.41421356237309505*sig*inverfc(2.*p)+mu);
}

double ln_p(double x, double mu, double sig)
{
    return (0.398942280401432678 / (sig*x)) * exp( -0.5*SQR((log(x)-mu)/sig) );
}

void ln_estimate(int* data, int n, double* mu, double* sig)
{
    double _mu = 0.0;
    double _sig = 0.0;

    int i;
    for (i = 0; i < n; i++)
    {
        _mu += log(data[i]);
    }
    
    _mu = _mu / n;
    
    for (i = 0; i < n; i++)
    {
        _sig += SQR(log(data[i]) - _mu);
    }
    
    _sig = sqrt( _sig / n );
    
    *mu = _mu;
    *sig = _sig;
}

void n_estimate(int* data, int n, double* mu, double* sig)
{
    double _mu = 0.0;
    double _sig = 0.0;
    
    int i;
    for (i = 0; i < n; i++)
    {
        _mu += data[i];
    }
    
    _mu = _mu / n;
    
    for (i = 0; i < n; i++)
    {
        _sig += SQR(data[i] - _mu);
    }
    
    _sig = sqrt( _sig / n );
    
    *mu = _mu;
    *sig = _sig;
}

void n_estimate_double(double* data, int n, double* mu, double* sig)
{
    double _mu = 0.0;
    double _sig = 0.0;
    
    int i;
    for (i = 0; i < n; i++)
    {
        _mu += data[i];
    }
    
    _mu = _mu / n;
    
    for (i = 0; i < n; i++)
    {
        _sig += SQR(data[i] - _mu);
    }
    
    _sig = sqrt( _sig / n );
    
    *mu = _mu;
    *sig = _sig;
}


