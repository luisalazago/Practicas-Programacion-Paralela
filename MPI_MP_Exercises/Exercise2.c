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
#include <mpi.h>
#include <omp.h>

#define f(x) ( 4.0 / ( 1.0 + ( x ) * ( x ) ) )

int main ( int argc, char *argv[] )
{
  /* Reference */
  double PI25DT = 3.141592653589793238462643;

  int     n, /* number of intervals */
          p,
          t,
          my_rank;    

  double  pi;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank ( MPI_COMM_WORLD, &my_rank );
  for ( ; ; )
  {     
        pi = 0.0;
        if(my_rank == 0){
            printf ( "Enter the number of intervals: (0 quits) ");
            scanf ( "%d", &n );
            printf("%d n.n\n", n);
        }
        
        if(n == 0)
            break;
        
        #pragma omp parallel
        {
            t = omp_get_num_threads();
            printf("%d pp bb\n", t);
            double h = 1 / ( double ) n;        /* width of trapezoid */
            double m = h / 2;              /* middle point of trapezoid */
            printf("%f h\n", h);
            printf("%d--\n", n/t);
            
            for(int i = 0; i < n/t; ++i) {
                double x = (h) * (( double ) omp_get_thread_num() * (double) (n/t) + (double) i) + m;
                pi += h * f ( x );     /* area of trapezoid */
            }	
            
        } 
        if(my_rank == 0)
            printf ( "pi is approximately %.16f, Error is %.16f\n",
                    pi, fabs ( pi - PI25DT ) );
  }
  return ( 0 );
}
