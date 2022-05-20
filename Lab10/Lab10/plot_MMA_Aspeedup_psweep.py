"""

  plot_MMA_Aspeedup_psweep.py - A Python script that uses NumPy and Matplotlib to plot
                                the speedup (according to Amdahl's Law) for MatMatAdd.py

  Notes:                        Teh matrix size is fixed

"""

import numpy as np
import matplotlib.pyplot as plt

# set a logarithmic, base 2, sequence from 2**startExp to 2**endExp for the x axis
startExp = 2
endExp = 19
xticks = endExp - startExp + 1
x = np.logspace ( startExp, endExp, xticks, base = 2 )

# set labels for x axis (processors)
labels = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
# speedups according to Amdahl's Law
y = [1.59, 1.76, 1.86, 1.92, 1.95, 1.96, 1.97, 1.97, 1.97, 1.97, 1.97, 1.97, 1.97, 1.98, 1.98,1.98,1.98,1.98]

# initialise the figure and axes
fig, ax = plt.subplots ()
# set the title
fig.suptitle ( "Speedup for MatMatAdd.py according to Amdahl's Law", fontweight = "bold" )
# set axes names
plt.xlabel ( 'processors' )
plt.ylabel ( "$\psi$" )
plt.gcf().subplots_adjust ( bottom = 0.15 )
# set the x scale as logarithmic, base 2
ax.set_xscale ( 'log', base = 2 )
# use rotated labels (30 degrees) in the x axis
plt.xticks ( x, labels )
plt.xticks ( rotation = 30 )

# plot using a grid and red circle markers
plt.plot ( x, y, 'ro' )
plt.grid ()

plt.savefig ( "MMA_Aspeedup_psweep.png" )
plt.show ()
