/*******************************************************************************
*
*  cpi_serial.c - Sequential (serial) version
*
*   Estimate the value of pi by means of the integral of 4 / ( 1 + x^2 ) using
*   tangent-trapezoidal rule using n trapezoids. The method is simple: the
*   integral is approximated by a sum of n intervals
*
*   Departamento de Electronica y Ciencias de la Computacion
*   Pontificia Universidad Javeriana - CALI
*
*******************************************************************************/

#include <stdio.h>
#include <math.h>

#define f(x) ( 4.0 / ( 1.0 + ( x ) * ( x ) ) )

int main ( int argc, char *argv[] )
{
  /* Reference */
  double PI25DT = 3.141592653589793238462643;

  int     n, /* number of intervals */
	  i;      

  double  h,
          x,
          m,
          pi;

  for ( ; ; )
  {
    printf ( "Enter the number of intervals: (0 quits) ");
    scanf ( "%d", &n );
    if ( n == 0 )
      break;

    pi = 0.0;
    h = 1.0 / ( double ) n;        /* width of trapezoid */
    m = h / 2;              /* middle point of trapezoid */
    
    for (i = 0; i < n; i++ )
    {
      x = h * ( double ) i + m;
      pi += h * f ( x );            /* area of trapezoid */
    }

    printf ( "pi is approximately %.16f, Error is %.16f\n",
              pi, fabs ( pi - PI25DT ) );
  }

  return ( 0 );
}
