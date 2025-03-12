import itertools
import json
import os
import pandas as pd
import numpy as np
import ast
import re

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalCrossentropy as cce_metric, CategoricalAccuracy, Precision, Recall, F1Score, AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from utilities.loaders import load_labels, load_audio, save_meta_data, save_model , concur_load_data
from utilities.preprocessors import encode_features
from models.voice_lstm import load_voice_lstm, load_voice_softmax

from argparse import ArgumentParser

def check_file_key(selector_config, estimator_name, hyper_param_config_key):
    # if json file exists read and use it
    os.makedirs("./results/", exist_ok=True)
    if os.path.exists(f'./results/{selector_config}_{estimator_name}_results.json'):
        # read json file as dictionary
        with open(f'./results/{selector_config}_{estimator_name}_results.json') as file:
            results = json.load(file)
        
            # also if hyper param already exists as a key then 
            # move on to next hyper param by returning from function
            if hyper_param_config_key in results[estimator_name]:
                return False
            
            file.close()
        return results

    # if json file does not exist create one and read as dictionary
    else:
        # will be populated later during loso cross validation
        results = {
            f'{estimator_name}': {
                # 'hyper_param_config_1': {
                #     'folds_train_acc': [<fold 1 train acc>, <fold 2 train acc>, ...],
                #     'folds_cross_acc': [<fold 1 cross acc>, <fold 2 cross acc>, ...],
                #     ...
                #     'folds_cross_roc_auc': [<fold 1 cross roc auc>, <fold 2 cross roc auc>, ...],
                # },

                # 'hyper_param_config_2': {
                #     'folds_train_acc': [<fold 1 train acc>, <fold 2 train acc>, ...],
                #     'folds_cross_acc': [<fold 1 cross acc>, <fold 2 cross acc>, ...],
                #     ...
                #     'folds_cross_roc_auc': [<fold 1 cross roc auc>, <fold 2 cross roc auc>, ...],
                # },

                # 'hyper_param_config_n': {
                #     'folds_train_acc': [<fold 1 train acc>, <fold 2 train acc>, ...],
                #     'folds_cross_acc': [<fold 1 cross acc>, <fold 2 cross acc>, ...],
                #     ...
                #     'folds_cross_roc_auc': [<fold 1 cross roc auc>, <fold 2 cross roc auc>, ...],
                # },
            }
        }

        return results
    
def k_fold_cross_validation(subjects_signals: list[np.ndarray],
    subjects_labels: list[np.ndarray],
    subject_to_id: dict,
    selector_config: str,
    alpha: float,
    opt: tf.keras.Optimizer,
    loss: tf.keras.Loss,
    metrics: list,
    estimator_name: str,
    estimator: tf.keras.Model,
    threshold_epochs: int,
    training_epochs: int,
    batch_size: int,
    **hyper_param_config: dict):
    """
    args:
        subjects_signals: pd.DataFrame - 
        subjects_labels: pd.DataFrame - 
        subject_to_id: dict - 
        model - 
        hyper_param_config: dict - 
    """

    # create key out of hyper_param_config
    hyper_param_config_key = "|".join([f"{hyper_param}_{value}" for hyper_param, value in hyper_param_config.items()])
    print(hyper_param_config_key)

    # if file exists or not return a dictionary but if hyper param 
    # config key already exists return from function
    if check_file_key(selector_config, estimator_name, hyper_param_config_key) != False:
        results = check_file_key(selector_config, estimator_name, hyper_param_config_key)
    else:
        return
    
    # define early stopping callback to stop if there is no improvement
    # of validation loss for 30 consecutive epochs
    stopper = EarlyStopping(
        monitor='val_auc',
        patience=threshold_epochs)
    callbacks = [stopper]

    # initialize empty lists to collect all metric values per fold
    folds_train_loss = []
    folds_train_acc = []
    folds_train_prec = []
    folds_train_rec = []
    folds_train_f1 = []
    folds_train_roc_auc = []

    folds_cross_loss = []
    folds_cross_acc = []
    folds_cross_prec = []
    folds_cross_rec = []
    folds_cross_f1 = []
    folds_cross_roc_auc = []

    # split signals and labels into train and cross by 
    # leaving 1 subject out for cross validatoin and the
    # rest for training, iterated for all subjects
    for subject_id in subject_to_id.values():
        # split data by leaving one subject out for testing
        # and the rest for training
        train_signals, train_labels, cross_signals, cross_labels = leave_one_subject_out(subjects_signals, subjects_labels, subject_id)
        train_labels[train_labels == 0] = -1 if estimator_name.lower() == "lstm-svm" else 0
        cross_labels[cross_labels == 0] = -1 if estimator_name.lower() == "lstm-svm" else 0

        # create/recreate model with specific hyper param configurations
        # in every hyper param config we must define/redefine an optimizer 
        model = estimator(**hyper_param_config)
        optimizer = opt(learning_rate=alpha)
        compile_args = {"optimizer": optimizer, "loss": loss, "metrics": metrics}# if len(metrics) > 0  else {"optimizer": optimizer, "loss": loss} 
        model.compile(**compile_args)

        # begin training model
        history = model.fit(train_signals, train_labels, 
        epochs=training_epochs,
        batch_size=batch_size, 
        callbacks=callbacks,
        validation_data=(cross_signals, cross_labels),
        verbose=1,)

        # # compare true cross and train labels to pred cross and train labels
        # pred_train = model.predict(train_signals)
        # pred_cross = model.predict(cross_signals)

        # if estimator_name.lower() == "lstm-svm":
        #     """still can't accomodate for sigmoid output layers
        #     because we do this sign and casting only if the dense
        #     layers are unactivated like the svm
        #     """
        #     print("converting output of lstm-svm to binary labels...")
        #     # signed_train_labels = tf.sign(pred_train_labels)

        #     # # using cast turns all negatives to -1, zeros to 0,
        #     # # and positives to 1
        #     # pred_train_labels = tf.cast(signed_train_labels >= 1, "float")
        #     # signed_cross_labels = tf.sign(pred_cross_labels)
        #     # pred_cross_labels = tf.cast(signed_cross_labels >= 1, "float")
        #     pred_train_probs = tf.nn.sigmoid(pred_train)
        #     pred_cross_probs = tf.nn.sigmoid(pred_cross)
        
        # accuracy takes in solid 1s and 0s
        # precision takes in solid 1s and 0s
        # recall takes in solid 1s and 0s

        # binary cross entropy takes in probability outputst
        # binary accuracy takes in probability outputs
        # f1 score takes in probability outputs
        # auc takes in probability outputs

        # now both models are able to output probability values
        # what I want is to also save DSC and SquaredHinge losses
        # in the dictionary of results
        fold_train_loss = history.history['loss'][-1]
        fold_cross_loss = history.history['val_loss'][-1]
        fold_train_acc = history.history['binary_accuracy'][-1]
        fold_cross_acc = history.history['val_binary_accuracy'][-1]
        fold_train_prec = history.history['precision'][-1]
        fold_cross_prec = history.history['val_precision'][-1]
        fold_train_rec = history.history['recall'][-1]
        fold_cross_rec = history.history['val_recall'][-1]
        fold_train_f1 = history.history['f1_score'][-1]
        fold_cross_f1 = history.history['val_f1_score'][-1]
        fold_train_roc_auc = history.history['auc'][-1]
        fold_cross_roc_auc = history.history['val_auc'][-1]
        
        # save append each metric value to each respective list
        folds_train_loss.append(fold_train_loss)
        folds_cross_loss.append(fold_cross_loss)
        folds_train_acc.append(fold_train_acc)
        folds_cross_acc.append(fold_cross_acc)
        folds_train_prec.append(fold_train_prec)
        folds_cross_prec.append(fold_cross_prec)
        folds_train_rec.append(fold_train_rec)
        folds_cross_rec.append(fold_cross_rec)
        folds_train_f1.append(fold_train_f1)
        folds_cross_f1.append(fold_cross_f1)
        folds_train_roc_auc.append(fold_train_roc_auc)
        folds_cross_roc_auc.append(fold_cross_roc_auc)

        print(f"fold: {subject_id} with hyper params: {hyper_param_config} \
              \ntrain loss: {fold_train_loss} cross loss: {fold_cross_loss} \
              \ntrain acc: {fold_train_acc} cross acc: {fold_cross_acc} \
              \ntrain prec: {fold_train_prec} cross prec: {fold_cross_prec} \
              \ntrain rec: {fold_train_rec} cross rec: {fold_cross_rec} \
              \ntrain f1: {fold_train_f1} cross f1: {fold_cross_f1} \
              \ntrain roc_auc: {fold_train_roc_auc} cross roc_auc: {fold_cross_roc_auc}")

    # once all fold train and cross metric values collected update read
    # dictionary with specific hyper param config as key and its recorded
    # metric values as value
    results[f'{estimator_name}'][hyper_param_config_key] = {
        'folds_train_loss': folds_train_loss,
        'folds_cross_loss': folds_cross_loss,
        'folds_train_acc':  folds_train_acc,
        'folds_cross_acc': folds_cross_acc,
        'folds_train_prec': folds_train_prec,
        'folds_cross_prec': folds_cross_prec,
        'folds_train_rec': folds_train_rec,
        'folds_cross_rec': folds_cross_rec,
        'folds_train_f1': folds_train_f1,
        'folds_cross_f1': folds_cross_f1,
        'folds_train_roc_auc': folds_train_roc_auc,
        'folds_cross_roc_auc': folds_cross_roc_auc
    }

    # if complete cross validation of all subjects is not finished 
    # this will not run
    with open(f'results/{selector_config}_{estimator_name}_results.json', 'w') as file:
        json.dump(results, file)

def grid_search_cv(subjects_signals: list[np.ndarray],
    subjects_labels: list[np.ndarray],
    subject_to_id: dict,
    selector_config: str,
    alpha: float,
    opt: tf.keras.Optimizer,
    loss: tf.keras.Loss,
    metrics: list,
    threshold_epochs: int,
    training_epochs: int,
    batch_size: int,
    estimator_name: str,
    estimator: tf.keras.Model,
    hyper_params: dict):

    """
    args:
        hyper_params - is a dictionary containing all the hyperparameters
        to be used in the model and the respective values to try

        e.g. >>> hyper_params = {'n_estimators': [10, 50, 100], 'max_depth': [3], 'gamma': [1, 10, 100, 1000]}
        >>> list(itertools.product(*list(hyper_params.values())))
        [(10, 3, 1), (10, 3, 10), (10, 3, 100), (10, 3, 1000), (50, 3, 1), (50, 3, 10), (50, 3, 100), (50, 3, 1000), (100, 3, 1), (100, 3, 10), (100, 3, 100), (100, 3, 1000)]
        >>>
        >>> keys, values = zip(*hyper_params.items())
        >>> perm_dicts = [dict(zip(keys, prod)) for prod in itertools.product(*values)]
        >>> perm_dicts
        [{'n_estimators': 10, 'max_depth': 3, 'gamma': 1}, {'n_estimators': 10, 'max_depth': 3, 'gamma': 10}, {'n_estimators': 10, 'max_depth': 3, 'gamma': 100}, {'n_estimators': 10, 'max_depth': 3, 'gamma': 1000}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 1}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 10}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 100}, {'n_estimators': 50, 'max_depth': 3, 'gamma': 1000}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 1}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 10}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 100}, {'n_estimators': 100, 'max_depth': 3, 'gamma': 1000}]
        >>>

        note in the passing of hyper param config dictionary to a function we can always unpack it by:
        >>> dict = {'a': 1, 'b': 2}
        >>> def myFunc(a=0, b=0, c=0):
        >>>     print(a, b, c)
        >>>
        >>> myFunc(**dict)
    """
    
    # unpack the dictionaries items and separate into list of keys and values
    # ('n_estimators', 'max_depth', 'gamma'), ([10, 50, 100], [3], [1, 10, 100, 1000])
    keys, values = zip(*hyper_params.items())

    # since values is an iterable and product receives variable arguments,
    # variable arg in product are essentially split thus above values
    # will be split into [10, 50, 100], [3], and [1, 10, 100, 1000]
    # this can be then easily used to produce all possible permutations of
    # these values e.g. (10, 3, 1), (10, 3, 10), (10, 3, 100) and so on...
    for prod in itertools.product(*values):
        # we use the possible permutations and create a dictionary
        # of the same keys as hyper params
        hyper_param_config = dict(zip(keys, prod))
        k_fold_cross_validation(
            subjects_signals, 
            subjects_labels, 
            subject_to_id, 
            selector_config,
            alpha,
            opt,
            loss,
            metrics,
            estimator_name,
            estimator,
            threshold_epochs, 
            training_epochs, 
            batch_size,
            **hyper_param_config)



# for training phase
def train_final_estimator(X,
    Y,
    alpha: float,
    opt: tf.keras.Optimizer,
    loss: tf.keras.Loss,
    metrics: list,
    threshold_epochs: int,
    training_epochs: int,
    batch_size: int,
    estimator_name: str,
    estimator,
    hyper_param_config: dict):

    """
    args:
        subjects_signals - 
        subjects_labels - 
        alpha - 
        loss - 
        metrics - 
        threshold_epochs - 
        training_epochs - 
        batch_size - 
        estimator_name - 
        estimator - 
        hyper_param_config - 
    """
    
    # # this would be appropriate if there was a larger ram
    # # min max scale training data and scale cross validation
    # # data based on scaler scaled on training data
    # scaler = MinMaxScaler()
    # X = scaler.fit_transform(X)
    # save_model(scaler, f'./saved/misc/{estimator_name}_scaler.pkl')

    # create model with specific hyper param configurations
    model = estimator(**hyper_param_config)

    # in every hyper param config we must define/redefine an optimizer 
    optimizer = opt(learning_rate=alpha)
    compile_args = {"optimizer": optimizer, "loss": loss, "metrics": metrics}# if len(metrics) > 0  else {"optimizer": optimizer, "loss": loss} 
    model.compile(**compile_args)

    # define checkpoint and early stopping callback to save
    # best weights at each epoch and to stop if there is no improvement
    # of validation loss for 10 consecutive epochs
    os.makedirs("./saved/weights/", exist_ok=True)
    path = f"./saved/weights/{estimator_name}"
    info = "_{epoch:02d}_{val_auc:.4f}.weights.keras"
    weights_path = path + info

    # create callbacks
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_auc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max')
    stopper = EarlyStopping(
        monitor='val_auc',
        patience=threshold_epochs)
        
    # append callbacks
    callbacks = [checkpoint, stopper]

    # save hyper params and other attributes of model 
    # for later model loading
    save_meta_data(f'./saved/misc/{estimator_name}_meta_data.json', hyper_param_config)

    # begin training final model, without validation data
    # as all data combining training and validation will all be used
    # in order to increase model generalization on test data
    history = model.fit(X, Y, 
    epochs=training_epochs,
    batch_size=batch_size, 
    callbacks=callbacks,

    # we set the validation split to all possible signal values not 
    # one subject so model can pull knowledge from more training subjects 
    validation_split=0.2,
    verbose=1,)

def create_hyper_param_config(hyper_param_list: list[str]):
    """
    will create a hyper param config dicitonary containing the specific
    values of each hyper param for a model to train with
    args:
        hyper_param_list - is a list of strings containing 
        the hyper param name and its respective values
        that will be parsed and extracted its key
        and value pairs to return as a dictionary
    """

    hyper_param_config = {}
    for hyper_param in hyper_param_list:
        try:
            print(hyper_param)
            # extract hyper param name and strip its last occuring underscore
            # extract hyper param value and convert to 
            # appropriate data type using literal evaluator
            key, value = hyper_param.rsplit('_', 1)
            hyper_param_config[key] = ast.literal_eval(value)
            
        except ValueError:
            hyper_param_config[key] = value

        finally:
            print(type(hyper_param_config[key]))

    if len(hyper_param_list) > 1:
        # add window size and hop size for lstm or softmax 
        # models to take in as kwargs
        hyper_param_config["window_size"] = int(hyper_param_config["window_time"] * hyper_param_config["hertz"])
        hyper_param_config["hop_size"] = int(hyper_param_config["hop_time"] * hyper_param_config["hertz"])
    
    return hyper_param_config


if __name__ == "__main__":
    # read and parse user arguments
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='lstm-cnn', help="model e.g. lstm-cnn for Llanes-Jurado et al. (2023) LSTM-CNN model, lstm-svm for Cueva et al. (2025), to train and validate ")
    parser.add_argument("-c", "--config", type=str, default="trad", 
        help="represents what pipeline which should be followed e.g. loading the \
        extracted features for ML models or loading and charging raw audio signals \
        for DL models")
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-5, help="what learning rate value should the optimizer of the model use")
    parser.add_argument("-the", "--threshold_epochs", type=int, default=10, help="represents how many epochs should the early stopping callback stop training the model tolerate metrics that aren't improving")
    parser.add_argument("-tre", "--training_epochs", type=int, default=30, help="represents how many epochs should at the maximum the model train regardless whether its metric values improve or not")
    parser.add_argument("-bs", "--batch_size", type=int, default=512, help="batch size during model training")
    parser.add_argument("--mode", type=str, default="tuning", help="tuning mode will not save weights during \
        fitting and while in training mode saves weights")
    parser.add_argument("--hyper_param_list", type=str, default=["window_time_0.5"], nargs="+", help="list of hyper parameters to be used as configuration during training")
    args = parser.parse_args()

    # # there are 16000 samples per second originally but
    # # if we let librosa interpolate our signals it would be 256hz
    # # which is frequency typical to that of a human voice
    # hertz = 256

    # # how many seconds we want our window to be
    # # e.g. if we want our signal segment to be 1 second
    # # then this would mean 16000 (or 22050) samples that we need to aggregate
    # # quarter of a second
    # window_time = 0.125

    # # how many seconds we want our signal segments to overlap
    # # one eighth of a second (1/8)
    # hop_time = 0.0625

    # build hyper param config dictionary from input
    hyper_param_config = create_hyper_param_config(hyper_param_list=args.hyper_param_list)
    print(hyper_param_config)

    # this is if user decides to use csv files of already extracted
    # features, there will be no need to concurrently load audios and labels
    if args.config == "trad":
        data_loader_kwargs = {
            "dataset": "train",
            "config": args.config
        }

        # load dataframes
        train_subjects_features, train_subjects_labels, _ = concur_load_data(**data_loader_kwargs)
        print(train_subjects_labels["0"].value_counts())
        
        # provide n_features kwarg to be used by softmax or any other ml model
        hyper_param_config["n_features"] = train_subjects_features.shape[1]

        X = train_subjects_features.to_numpy()
        Y_sparse = train_subjects_labels["0"].to_numpy()
        unique = np.unique(Y_sparse)

        # set value of n_units hyper param
        hyper_param_config["n_units"] = len(unique)

        # one hot encode train_subjects_labels
        Y = tf.one_hot(Y_sparse.reshape(-1), depth=len(unique))

    else:
        # read and load data as this will be converted to a list of 3D numpy arrays
        DIR = "./data/"
        folders = list(filter(lambda file: not file.endswith(".tgz"), os.listdir(DIR)))[:10]
        labels = load_labels(DIR, folders)
        signals = load_audio(DIR, folders, hyper_param_config["hertz"])

        # convert 2 latter variables to dataframe and merge the two
        signals_df = pd.DataFrame(signals, columns=["subject_name", "raw_signals"])
        labels_df = pd.DataFrame(labels, columns=["subject_name", "label"])
        dataset_df = signals_df.merge(labels_df, how="left", on=["subject_name"])

        dataset_df["label"], dataset_df_le = encode_features(dataset_df["label"])
        save_model(dataset_df_le, './saved/misc/audio_dataset_le.pkl')

        # split dataset
        train_dataset_df, test_dataset_df = train_test_split(dataset_df, test_size=0.2, random_state=0)

        # convert split data to lists
        train_dataset = list(train_dataset_df.itertuples(index=False, name=None))
        test_dataset = list(test_dataset_df.itertuples(index=False, name=None))

        data_loader_kwargs = {
            "dataset": train_dataset,
            "hertz": hyper_param_config["hertz"],
            "window_time": hyper_param_config["window_time"], 
            "hop_time": hyper_param_config["hop_time"], 
            "config": args.config
        }

        # load training data concurrently
        train_subjects_signals, train_subjects_labels, times  = concur_load_data(**data_loader_kwargs)
        print(train_subjects_signals[0].shape)

        # concatenate both train  lists of
        # signal and label numpy arrays into a single 
        # combined training dataset
        X = np.concatenate(train_subjects_signals, axis=0)
        Y_sparse = np.concatenate(train_subjects_labels, axis=0)
        unique = np.unique(Y_sparse)

        # set value of n_units hyper param
        hyper_param_config["n_units"] = len(unique)

        # one hot encode train_subjects_labels
        Y = tf.one_hot(Y_sparse.reshape(-1), depth=len(unique))

    



    # model hyper params
    models = {
        'lstm': {
            'model': load_voice_lstm, 
            'hyper_params': {
                'window_size': [hyper_param_config["window_time"] * hyper_param_config["hertz"]], 
                'n_a': [64, 128], 
                'dense_drop_prob': [0.05, 0.2, 0.75], 
                'rnn_drop_prob': [0.2],
                'units': [1]
            },
            'opt': Adam,
            'loss': cce_loss(),

            # following metrics would not work since all these require z to be activated by the sigmoid activation function
            # and naturally comparing Y_true which are 1's and 0's to unactivated values like 1.23, 0.28, 1.2, etc. will result
            # in 0 metric values being produced from Precision, Recall, etc.
            'metrics': [cce_metric(), CategoricalAccuracy(), Precision(), Recall(), F1Score(), AUC(name='auc'), ]
        },
        'softmax': {
            'model': load_voice_softmax,
            'hyper_params': {
                'n_features': [20],
            },
            'opt': SGD,
            'loss': cce_loss(),
            'metrics': [cce_metric(), CategoricalAccuracy(), Precision(), Recall(), F1Score(), AUC(name='auc')]
        },
    }

    if args.mode.lower() == "tuning":
        # do feature selection, hyperparameter tuning, 
        # loso cross validation across all subjects, and
        # save model & results
        grid_search_cv(
            subjects_signals, 
            subjects_labels, 
            subject_to_id,
            selector_config=args.pipeline,
            alpha=args.learning_rate,
            opt=models[args.model]['opt'],
            loss=models[args.model]['loss'],
            metrics=models[args.model]['metrics'],
            threshold_epochs=args.threshold_epochs,
            training_epochs=args.training_epochs,
            batch_size=args.batch_size,
            estimator_name=args.model,
            estimator=models[args.model]['model'],
            hyper_params=models[args.model]['hyper_params'],
        )

    elif args.mode.lower() == "training":
        
        # # we can just modify this script such that it doesn't loop through hyper param configs anymore and
        # # will just only now 1. load the preprocessed features, load the reduced feature set, 
        # # exclude use of grid serach loso cv, loso cross validation, and leave one subject out
        # # and instead use the best hyper param config obtained from summarization.ipynb and train the model
        # # not on a specific fold or set of subjects but all subjects
        # train_final_estimator(
        #     X,
        #     Y, 
        #     alpha=args.learning_rate,
        #     opt=models[args.model]['opt'],
        #     loss=models[args.model]['loss'],
        #     metrics=models[args.model]['metrics'],
        #     threshold_epochs=args.threshold_epochs,
        #     training_epochs=args.training_epochs,
        #     batch_size=args.batch_size,
        #     estimator_name=args.model,
        #     estimator=models[args.model]['model'],
        #     hyper_param_config=hyper_param_config
        # )
        pass