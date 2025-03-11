* after extractio next thing would be how to extract the files
* we need to delete all .tgz files after extraction to save space
* next is we need to load these .wav files individually 
* next is we need to preprocess these loaded wav files using some kind of library individually
* take not that we do not the individual .wav files, we need to look at each .wav files as single data points
* then we do feature extraction on these individual data points or rather single signals, based on some kind of window
* labels are placed inside readme.md file in 5th line starting from line 1

* calculate mel spectogram from each segment as a result of each window and calculate the necessary features from there
* calculate zero cross rate features
* but all these have and should be calculated strictly right after splitting the data first because this is to prevent data leakage
* use this sample command for running training of deep learning models `python tuning_dl.py -m lstm -c deep -lr 1e-3 --batch_size 32 --mode training --hyper_param_list window_time_1 hop_time_0.5 n_a_64 dense_drop_prob_0.2 rnn_drop_prob_0
.2 n_units_2`
* `python tuning_dl.py -m lstm -c deep -lr 1e-3 --batch_size 32 --mode training --hyper_param_list window_time_0.125 hop_time_0.0625 n_a_128 dense_drop_prob_0.2 rnn_drop_prob_0.2 n_units_2`
* `python tuning_dl.py -m lstm -c deep -lr 1e-3 --batch_size 32 --mode training --hyper_param_list hertz_8000 window_time_0.25 hop_time_0.125 n_a_128 dense_drop_prob_0.2 rnn_drop_prob_0.2` - this is if number of subjects is 10
* `python tuning_dl.py -m lstm -c deep -lr 1e-3 --batch_size 256 --mode training --hyper_param_list hertz_8000 window_time_0.25 hop_time_0.125 n_a_128 dense_drop_prob_0.2 rnn_drop_prob_0.2` - this is if number of subjects is 100