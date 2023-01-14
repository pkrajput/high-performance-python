
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import warnings
import os
warnings.filterwarnings('ignore')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

step_count = np.power(10, 3)
x_initial_state = 0.1
r_slices = np.power(10 ,2)

r_range = np.linspace(1, 10, r_slices)

def get_next_step(r, x):
    return r * x * (1. - x)

size_per_rank = int(r_slices / size)



if rank == size - 1:
  size_per_rank += r_slices % size
  left_limit, right_limit = rank * size_per_rank, r_slices - 1

else:
  left_limit, right_limit = rank * size_per_rank, (rank + 1) * size_per_rank

iter_data = np.empty([size_per_rank, r_slices])




for i, r in enumerate(r_range[left_limit: right_limit]):
  x = [x_initial_state]
  for _ in range(step_count):
      x.append(get_next_step(r, x[-1]))
      
  iter_data[i] = x[500: 500 + r_slices]

iter_data = comm.gather((iter_data, rank) if size > 1 else iter_data, root=0)

MPI.Finalize()



if rank == 0:
  if size > 1:
    iter_data.sort(key = lambda tup: tup[1])
    iter_data = np.array(iter_data)[:, :-1].squeeze()
    iter_data = np.concatenate(iter_data, axis=0)
  else:
    iter_data = np.array(iter_data).squeeze()
    
  fig, ax = plt.subplots(figsize=(10, 8))

  fig.set_facecolor('red')

  ax.axis([min(r_range), 5, 0, 1 + 0.1])
  ax.set_facecolor('blue')

  ax.set_xlabel('r')
  ax.set_ylabel('population equlibrium')
  l, = ax.plot([], [], '.', color='green')

  t = [r * np.ones(iter_data.shape[0]) for r in r_range]
  
  dir_name = 'gif_pics'
  
  os.system(f'rm -rf {dir_name}')
  os.mkdir(dir_name)

  images = []

  for i, _ in enumerate(t):
      index = str(i).zfill(4)
      imgname = f'{dir_name}/bif_pic_{index}' 
      images.append(imgname)

      ax.scatter(t[:i], iter_data[:i], c='yellow')
      plt.savefig(images[-1])
