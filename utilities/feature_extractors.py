import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor
from scipy.stats import kurtosis as kurt, skew, entropy, mode 

def extract_features(dataset: list, hertz: int=16000, window_time: int=3, hop_time: int=1, config="trad"):
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
        print(f"samples per window size: {samples_per_win_size}")
        print(f"samples per hop size: {samples_per_hop_size}")

        # initialize segments to empty list as this will store our
        # segmented signals 
        # subject_names = []
        segments = []
        labels = []

        # fig = plt.figure(figsize=(17, 5))
        n_frames = 0

        if config == "trad":
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

                # calculate frequency domain features
                # get the spectrogram by calculating short time fourier transform
                spectrogram = np.abs(librosa.stft(segment))
                # print(f"spectrogram shape: {spectrogram.shape}")

                # Get the frequencies corresponding to the spectrogram bins
                frequencies = librosa.fft_frequencies(sr=hertz)
                # print(f"frequencies shape: {frequencies.shape}")

                # Find the frequency bin with the highest average energy
                peak_frequency_bin = np.argmax(np.mean(spectrogram, axis=1))

                # Get the peak frequency in Hz
                # calculate also peak frequency
                # I think dito na gagamit ng fast fourier transform
                # to obtain the frequency, or use some sort of function
                # to convert the raw audio signals into a spectogram
                peak_frequency = frequencies[peak_frequency_bin]

                # calculate the segments fast fourier transform
                ft = np.fft.fft(segment)

                # the fft vector can have negative or positive values
                # so to avoid negative values and just truly see the frequencies
                # of each segment we use its absolute values instead 
                magnitude = np.abs(ft)
                mag_len = magnitude.shape[0]
                frequency = np.linspace(0, hertz, mag_len)

                

                # calculate statistical features
                # because the frequency for each segment is 16000hz we can divide
                # it by 1000 to instead to get its kilo hertz alternative
                mean_freq_kHz = np.mean(segment, axis=0)
                median_freq_kHz = np.median(segment, axis=0)
                std_freq = np.std(segment, axis=0)
                mode_freq = mode(segment, axis=0)
                
                # min = np.min(segment, axis=0)

                # calculate first quantile, third quantile, interquartile range
                first_quartile_kHz = np.percentile(segment, 25) / 1000,
                third_quartile_kHz = np.percentile(segment, 75) / 1000,
                inter_quartile_range_kHz = (np.percentile(segment, 75) - np.percentile(segment, 25)) / 1000,

                # compute morphological features
                skewness = skew(segment)
                kurtosis = kurt(segment)

                # compute time domain features
                amp_env = np.max(segment, axis=0)
                rms = np.sqrt(np.sum(segment ** 2, axis=0) / samples_per_win_size)

                features = {
                    # statistical features
                    "mean_freq_kHz": mean_freq_kHz,
                    "median_freq_kHz": median_freq_kHz,
                    "std_freq": std_freq,
                    "mode_freq": mode_freq[0],
                    'first_quartile_kHz': first_quartile_kHz,
                    'third_quartile_kHz': third_quartile_kHz,
                    'inter_quartile_range_kHz': inter_quartile_range_kHz,

                    # morphological features
                    "skewness": skewness,
                    "kurtosis": kurtosis,

                    # time domain features
                    "amp_env":amp_env,
                    "rms": rms,
                    
                    # frequency features
                    "peak_frequency": peak_frequency,
                }
                
                segments.append(features)
                
                n_frames += 1

            frames = range(n_frames)
            print(f"number of frames resulting from window size of {samples_per_win_size} \
                and a hop size of {samples_per_hop_size} from audio signal frequency of {hertz}: {frames}")

            time = librosa.frames_to_time(frames, hop_length=samples_per_hop_size)
            print(f"shape of time calculated from number of frames: {time.shape[0]}")

            
            # calculate zero crossing features
            zcr = librosa.feature.zero_crossing_rate(y=x_signals, frame_length=samples_per_win_size, hop_length=samples_per_hop_size)
            
            # calculate the number of values we need to remove in the
            # feature vector librosa calculated for us compared to the
            # feature vectors we calculated on our own
            n_values_to_rem = np.abs(zcr.shape[1] - time.shape[0])
            zcr = zcr.reshape(-1)[:-n_values_to_rem]

            subject_data = pd.DataFrame.from_records(segments)
            subject_data["zcr"] = zcr
            subject_data["label"] = label
            subject_data["subject_name"] = subject_name

            

            # librosa.display.waveshow(x_signals, alpha=0.5, color="lightgreen")
            # plt.plot(time, subject_data["rms"], color="blue")
            # plt.tight_layout()
            # plt.show()

            return (subject_data, time)
        
        else:
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
                if segment.shape[0] < samples_per_win_size:
                    last_sample = segment[-1]

                    # (n_padding_we_want_for_the_front_of_the_array, 
                    # n_padding_we_want_for_the_back_of_the_array)
                    # 
                    n_pad_to_add = samples_per_win_size - segment.shape[0]
                    print(f"n padding to be added: {n_pad_to_add}") 

                    # we use the last value of the segment as padding to fill in
                    # the empty spots
                    segment = np.pad(segment, (0, n_pad_to_add), mode="constant", constant_values=last_sample)
                
                segments.append(segment)
                labels.append(label)

            # because x_window_list and y_window_list when converted to a numpy array will
            # be of dimensions (m, 640) and (m,) respectively we need to first and foremost
            # reshpae x_window_list into a 3D matrix such that it is able to be taken in
            # by an LSTM layer, m being the number of examples, 640 being the number of time steps
            # and 1 being the number of features which will be just our raw audio signals.    
            X = np.array(segments)
            subject_signals = np.reshape(X, (X.shape[0], X.shape[1], -1))

            Y = np.array(labels)
            subject_labels = np.reshape(Y, (Y.shape[0], -1))

            frames = range(n_frames)
            print(f"number of frames resulting from window size of {samples_per_win_size} \
            and a hop size of {samples_per_hop_size} from audio signal frequency of {hertz}: {frames}")

            time = librosa.frames_to_time(frames, hop_length=samples_per_hop_size)
            print(f"shape of time calculated from number of frames: {time.shape[0]}")

            return (subject_signals, subject_labels, time) 

    with ThreadPoolExecutor() as exe: 
        subjects_data = list(exe.map(helper, dataset))

        if config == "trad":
            subjects_features, time = zip(*subjects_data)
            return subjects_features, time
        else:
            subjects_signals, subjects_labels, time = zip(*subjects_data)
            return subjects_signals, subjects_labels, time