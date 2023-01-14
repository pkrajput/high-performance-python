
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def trapezoidal(function, a, b, n, h):

    s = 0.5*(function(a) + function(b))
    
    for i in range(1,n,1):
        s = s + function(a + i*h)
    
    value = h*s
    
    return value

def function(x):
    return 1/(np.sqrt(1 + x**2))

def Get_data(rank, size, comm):
    a=None
    b=None
    n=None
    if rank == 0:
        a = 3
        b = 4
        n = 10000
        
    a=comm.bcast(a)
    b=comm.bcast(b)
    n=comm.bcast(n)
    return a,b,n

a,b,n = Get_data(rank, size, comm)

h = (b-a)/n # step size for trapezoidal rule is common for all processes

mpi_n = int(n/size) # number of trapezoids/size is also same variable for all calls 


mpi_a = a + rank*mpi_n*h
mpi_b = mpi_a + mpi_n*h
integral = trapezoidal(function, mpi_a, mpi_b, mpi_n, h)

total=comm.reduce(integral)

if (rank == 0):
    print(f'for n = {n} trapezoids the integral from {a} to {b} for our defined function = {total}')
    
MPI.Finalize
