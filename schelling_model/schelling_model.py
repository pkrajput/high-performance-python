#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
#display in cell 4 is missing because of this. I had to use this to stop plots from rendering in notebook later
#it made the size huge
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# In[2]:


N = 200       # Grid will be N x N
R = np.linspace(0, 1, 9)
similarity = 1 - R  # Similarity threshold 1-R
EMPTY = 0.001  # Fraction of vacant cells
X_to_O = 1   # Ratio of blue to red people


# In[3]:


def rand_init(N, X_to_O, EMPTY):
    """ Random system initialisation.
    X  =  0
    O   =  1
    EMPTY = -1
    """
    vacant = int(N * N * EMPTY)
    population = N * N - vacant
    x = int(population * 1 / (1 + 1/X_to_O))
    o = population - x
    M = np.zeros(N*N, dtype=np.int8)
    M[:o] = 1
    M[-vacant:] = -1
    np.random.shuffle(M)
    return M.reshape(N,N)


# In[4]:


fig, ax = plt.subplots(figsize=(10, 6))

plt.imshow(rand_init(N, X_to_O, EMPTY), cmap = plt.cm.gnuplot2)

plt.axis('off')

plt.show()


# In[5]:


KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.int8)


# In[6]:


from scipy.signal import convolve2d

def evolve(M, boundary='wrap', SIM_T = 0.4):
    """
    Args:
        M (numpy.array): the matrix to be evolved
        boundary (str): Either wrap, fill, or symm
    If the similarity ratio of neighbours
    to the entire neighborhood population
    is lower than the SIM_T,
    then the individual moves to an empty house.
    """
    kws = dict(mode='same', boundary=boundary)
    b_neighs = convolve2d(M == 0, KERNEL, **kws)
    r_neighs = convolve2d(M == 1, KERNEL, **kws)
    neighs   = convolve2d(M != -1,  KERNEL, **kws)

    b_dissatisfied = (b_neighs / neighs < SIM_T) & (M == 0)
    r_dissatisfied = (r_neighs / neighs < SIM_T) & (M == 1)
    M[r_dissatisfied | b_dissatisfied] = - 1
    vacant = (M == -1).sum()

    n_b_dissatisfied, n_r_dissatisfied = b_dissatisfied.sum(), r_dissatisfied.sum()
    filling = -np.ones(vacant, dtype=np.int8)
    filling[:n_b_dissatisfied] = 0
    filling[n_b_dissatisfied:n_b_dissatisfied + n_r_dissatisfied] = 1
    np.random.shuffle(filling)
    M[M==-1] = filling
    
    dissatisfied_households = n_b_dissatisfied + n_r_dissatisfied
    
    return M, dissatisfied_households


# In[7]:


M = rand_init(N, X_to_O, EMPTY)
evolved_M, d_h = evolve(M)
print(np.shape(evolved_M))
print(d_h)


# In[8]:


from tqdm import tqdm
import imageio


# In[9]:


n_diss_households = [[]]


# 1. Create 9 gifs of map evolution for 9 values of R 

# In[10]:


for sim in tqdm(similarity):
    
    r = 1 - sim
    M = rand_init(N, X_to_O, EMPTY)
    plt_names = []
    diss_households = []

    for i in range(100):
        M, dis_h = evolve(M, SIM_T = sim)
        diss_households.append(dis_h)

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.ioff()
        plt.imshow(M, cmap = plt.cm.gnuplot2)
        plt.axis('off')

        plt.savefig(f'/home/prateek/sk_courses/hppl/schelling_model/{i}.png', dpi = 200)
        plt_names.append(f'{i}.png')
        
    n_diss_households.append(diss_households)
    
    images = []
    for i in plt_names:
        images.append(imageio.imread(f'/home/prateek/sk_courses/hppl/schelling_model/{i}'))
    imageio.mimsave(f'/home/prateek/sk_courses/hppl/schelling_model/r_{r}.gif', images)
    


# 2. Plot the number of households that want to move versus time for 9 values of R on one graph, label 9 curves, label the axes and title the graph

# In[12]:


fig, ax = plt.subplots(figsize=(14, 14))

ax.set_xlabel('time')
ax.set_ylabel('n_angry_agents')
ax.set_title('number of angry agents with time')


for d_h, r in zip(n_diss_households, R.tolist()):
        
    ax.plot(d_h, '-o', label=f'R={r}')
    
ax.legend()
plt.savefig(f'/home/prateek/sk_courses/hppl/schelling_model/angry_agents_with_time', dpi = 200)


# In[ ]:




