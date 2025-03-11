import numpy as np
import pandas as pd
import librosa
import librosa.display

from concurrent.futures import ThreadPoolExecutor
from scipy.stats import kurtosis as kurt, skew, entropy, mode 

def extract_features(dataset: list, hertz: int=16000, window_time: int=3, hop_time: int=1):
    """
    extracts the features from each segment of an audio signal
    """
    
    def helper(datum):
        # we access the SCR values via raw data column
        subject_name = datum[0]
        x_signals = datum[1]
        label = datum[2]

        print(subject_name)

        # get number of rows of 16000hz signals 
        n_rows = x_signals.shape[0]
        # print(n_rows)

        # we calculate the window size of each segment or the
        # amount of samples it has to have based on the frequency
        samples_per_win_size = int(window_time * hertz)
        samples_per_hop_size = int(hop_time * hertz)
        # print(samples_per_win_size)
        # print(samples_per_hop_size)

        # initialize segments to empty list as this will store our
        # segmented signals 
        subject_names = []
        segments = []
        labels = []

        

        # this segments our signals into overlapping segments
        for i in range(0, (n_rows - samples_per_win_size) + samples_per_hop_size, samples_per_hop_size):
            # # last segment would have start x: 464000 - end x: 512000
            # # and because 512000 plus our hop size of 16000 = 528000 
            # # already exceeding 521216 this then terminates the loop
            # i += samples_per_hop_size
            # start = i
            # end = i + samples_per_win_size
            start = i
            end = min((i + samples_per_win_size), n_rows)
            # print(f'start x: {start} - end x: {end}')

            # extract segment from calculated start and end
            # indeces
            segment = x_signals[start:end]

            # Calculate the spectrogram
            spectrogram = np.abs(librosa.stft(segment))

            # Get the frequencies corresponding to the spectrogram bins
            frequencies = librosa.fft_frequencies(sr=hertz)

            # Find the frequency bin with the highest average energy
            peak_frequency_bin = np.argmax(np.mean(spectrogram, axis=1))

            # Get the peak frequency in Hz
            peak_frequency = frequencies[peak_frequency_bin]
            
            # calculate statistical features
            # because the frequency for each segment is 16000hz we can divide
            # it by 1000 to instead to get its kilo hertz alternative
            mean_freq_kHz = np.mean(segment, axis=0)
            median_freq_kHz = np.median(segment, axis=0)
            std_freq = np.std(segment, axis=0)
            mode_freq = mode(segment, axis=0)

            # calculate also peak frequency
            # I think dito na gagamit ng fast fourier transform
            # to obtain the frequency, or use some sort of function
            # to convert the raw audio signals into a spectogram
            
            # max = np.max(segment, axis=0)
            # min = np.min(segment, axis=0)

            # calculate first quantile, third quantile, interquartile range
            first_quartile_kHz = np.percentile(segment, 25) / 1000,
            third_quartile_kHz = np.percentile(segment, 75) / 1000,
            inter_quartile_range_kHz = (np.percentile(segment, 75) - np.percentile(segment, 25)) / 1000,

            # compute morphological features
            skewness = skew(segment)
            kurtosis = kurt(segment)
            features = {
                "mean_freq_kHz": mean_freq_kHz,
                "median_freq_kHz": median_freq_kHz,
                "std_freq": std_freq,
                "mode_freq": mode_freq[0],
                "skewness": skewness,
                "kurtosis": kurtosis,
                "peak_frequency": peak_frequency,
                'first_quartile_kHz': first_quartile_kHz,
                'third_quartile_kHz': third_quartile_kHz,
                'inter_quartile_range_kHz': inter_quartile_range_kHz
            }
            
            segments.append(features)

        final = pd.DataFrame.from_records(segments)
        final["label"] = label  
        final["subject_name"] = subject_name
        return final

    with ThreadPoolExecutor() as exe: 
        subject_features = list(exe.map(helper, dataset))

    return subject_features