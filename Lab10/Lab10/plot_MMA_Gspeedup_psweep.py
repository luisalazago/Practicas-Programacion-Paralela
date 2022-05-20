"""

  plot_MMA_Aspeedup_psweep.py - A Python script that uses NumPy and Matplotlib to plot
                                the speedup (according to Gustafson's Law) for MatMatAdd.py

  Notes:                        Reads data from a csv file

"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# set the number of processors using a logarithmic, base 2, sequence from 2**startExp to 2**endExp
startExp = 2
endExp = 19
xticks = endExp - startExp + 1
p = np.logspace ( startExp, endExp, xticks, base = 2 )

# open .csv data file in read-only mode
csv_file = open ( 'MMA_Gsita.dat', 'r' )
# get labels and speedups from this file
labels = []
speedups = []

# iterate over each line in the csv file; split the fields in the line using a comma as delimiter
for label, speedup in csv.reader ( csv_file, delimiter = ',' ):
  # the first field in each line corresponds to the no of processors
  labels.append ( label )
  # the second field in each line corresponds to the speedup
  speedups.append ( float (speedup) )
#  speedups.append ( speedup )

# initialise the figure and axes
fig, ax = plt.subplots ()
# set the title
fig.suptitle ( "Speedup for MatMatAdd.py according to Gustafson's Law", fontweight = "bold" )
# set axes names
plt.xlabel ( 'processors' )
plt.ylabel ( "$\psi$" )
plt.gcf().subplots_adjust ( left = 0.2, bottom = 0.15 )
# set the x scale as logarithmic, base 2
ax.set_xscale ( 'log', base = 2 )
# use rotated labels (30 degrees) as ticks in the x axis
plt.xticks ( p, labels )
plt.xticks ( rotation = 30 )

# plot using a grid and green circle markers
plt.plot ( p, speedups, 'go' )
plt.grid ()

plt.savefig ( "MMA_Gspeedup_psweep.png" )
plt.show ()
