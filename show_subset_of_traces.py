import matplotlib.pyplot as plt
import numpy as np
import seismicutil

# load the seismic data ( a list of SeismicTrace objects)
data = seismicutil.load_seismic_data()
n = len(data)  # number of traces total
n_to_show = 3  # number of traces to visualize
random_set = np.random.permutation(n)[:n_to_show]

for k in random_set:
    seismicutil.plot_waveform(data[k].trace, data[k].arr)

plt.show()