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
* 