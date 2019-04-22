import random
import pickle  # for loading data
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def get_psd(waveform, fs=40, nperseg=128):
    """
    Get power spectral density as funtion of time. (Totally a thing.)
    :param waveform:
    :param fs:
    :param nperseg: number of time-samples in each window
    :return:
    """

    # b, a = signal.butter(3, (0.1, .35), 'bandpass')
    # waveform = signal.filtfilt(b, a, waveform)

    # Compute the spectrogram
    f, t, psd = signal.spectrogram(
        waveform,
        fs=fs,
        nperseg=nperseg,
        noverlap=int(0.9*nperseg),
        return_onesided=True
    )
    psd = np.log10(np.abs(psd) + 1e-12)

    return f, t, psd


class SeismicTrace(object):
    """
    Class to encapsulate seismic trace and arrival times.
    """
    def __init__(self, trace, arr, target=None):
        """
        Seismic trace object stores the known waveform and labeled arrival
        times.

        :param trace: array of size (nsamples,) representing the seismic trace
        :param arr: array of size (narrivals,) representing the indices of
        trace when a seismic arrival was identified by expert.
        """
        self.trace = trace
        self.arr = arr
        self.target = target


    def get_window_data(self, win_sz, training=False):
        """
        Return disjoint windows extracted from self.trace.

        :param win_sz: integer number of samples per window
        :return:
        """
        fs = 40
        half_win = win_sz / 2
        wf = self.trace.copy()
        label = self.target.copy()

        f, t, data = get_psd(wf, nperseg=win_sz)

        target = []

        # for each datum,
        for k in range(len(t)):
            cntr = int(fs * t[k])
            target.append(label[cntr - half_win:cntr + half_win])

        target = np.array(target)
        data = data.transpose()

        signal_rows = set()

        # Collect windows near arrivals, and far from arrivals.
        if training:
            delta = 20  # plus/minus delta windows around each arrival
            time_index = t * fs
            test_output = time_index <= self.arr[:, None]
            for k_i in range(len(self.arr)):
                row = np.where(test_output[k_i])[0][-1]  # contains arrival
                a = max(0, row - delta)
                b = min(data.shape[0] - 1, row + delta)
                for j in range(a, b+1):
                    signal_rows.add(j)

            noise_rows = set(range(data.shape[0])) - signal_rows

            noise_rows = random.sample(
                noise_rows,
                min(len(signal_rows), len(noise_rows))
            )
            data = np.concatenate(
                (data[list(signal_rows)], data[list(noise_rows)]),
                axis=0
            )

            target = np.concatenate(
                (target[list(signal_rows)], target[list(noise_rows)]),
                axis=0
            )


        # # Center each coordinate
        # mu = np.mean(data, axis=0)  # mean of each window data
        # for i in range(data.shape[1]):
        #     data[:, i] -= mu[i]

        return data, target.mean(axis=1).reshape(-1,1)


def load_seismic_data():
    """
    Load a pickle file containing a list of SeismicTrace objects.

    :return: a list of SeismicTrace objects.
    """
    filename = './Data/seismicdata.dat'
    with open(filename, 'rb') as f:
        traces = pickle.load(f)

    return traces


def plot_waveform(waveform, arrivals=None, fs=40, zoom=True, label=None):
    """
    Visualize a waveform, its spectrogram, and any accompanying arrival times.

    :param waveform: an array of timesamples representing a seismic waveform
    :param arrivals: labeled arrival times, if available
    :param fs: sampling frequency (in Hertz)
    :param zoom: boolean. If true, we only show the region surrounding the
    :param label: array of same shape as waveform, representing the target
    sequence.
    """

    print('Waveform contains {} samples.'.format(len(waveform)))

    f, t, psd = get_psd(waveform, fs)

    print('{1} points in {0} dimensions.'.format(*psd.shape))

    # Plot the waveform (and label)
    fig, ax = plt.subplots(2, 1, figsize=(10, 4))
    time = np.arange(len(waveform)) / float(fs)
    ax[0].plot(time, waveform)
    if label is not None:
        ax[0].plot(time, label*waveform.max(), 'r.-', lw=2)
    ax[0].set_title('Original waveform')
    ax[0].set_xlabel('Time in seconds')
    ax[0].autoscale(enable=True, axis='both', tight=True)

    # Plot the waveform's spectrogram
    ax[1].pcolormesh(t, f, psd)
    ax[1].set_ylabel('Frequency [Hz]')
    ax[1].set_xlabel('Time [sec]')
    ax[1].autoscale(enable=True, axis='both', tight=True)

    if arrivals is not None:
        for i in (0,1):
            for k in arrivals:
                ylims = ax[i].get_ylim()
                ax[i].vlines(k / float(fs), ylims[0], ylims[1], lw=2)

    plt.show(block=False)


def estimate_dominant_frequency(trace, fs):
    """
    Estimate the dominant frequency present in trace, a seismic record obtained
    with sampling frequency fs.

    :param trace: is an array of size (n,)
    :param fs: an integer, representing the sampling frequency
    :return: an integer, representing the dominant frequency present in the
    input signal.
    """
    b, a = signal.butter(3, (0.01, .5), 'bandpass')
    fx = signal.filtfilt(b, a, trace)

    n = len(trace)

    fc = np.fft.fft(fx)  # Fourier coefficients
    fr = np.fft.fftfreq(n, 1.0/fs)  # Frequency bins
    dom_freq = abs(fr[np.argmax(abs(fc))])
    return dom_freq
