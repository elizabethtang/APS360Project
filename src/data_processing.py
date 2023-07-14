import numpy as np
import h5py
import torch
from torch.utils.data import TensorDataset
from globals import DATA
from utils import change_to_working_directory


def get_Xy_main_test_data(file_path, train_val_split=0.8):
    # Open the .mat file using h5py.File()
    with h5py.File(file_path, "r") as hdf_file:
        # List all the top-level keys (datasets, groups) in the HDF5 file
        ref_key = list(hdf_file.keys())[0]
        value_key = list(hdf_file.keys())[1]

        data = hdf_file[value_key]
        data = data[:200,0]

        numpy_data = []

        for i in range(data.shape[0]):
            person = data[i]
            obj = hdf_file[person]
            obj = obj[:]
            if obj.shape[0] > 5000:
                obj = obj[1000:4000]    #take 3000 data points for each subject
                numpy_data.append(obj)
            if len(numpy_data) >84:    #take the first 84 subjects
                break
    
    #convert numpy_data to numpy array
    numpy_data = np.array(numpy_data)
    #ppg_ecg is the 0th and 2nd channel
    ppg = numpy_data[:,:,0]
    ecg = numpy_data[:,:,2]
    ppg_ecg = np.stack((ppg, ecg), axis=2)
    abp = numpy_data[:,:,1]

    train_index = int(abp.shape[0] * train_val_split)

    X_main, X_test, y_main, y_test = ppg_ecg[:train_index], ppg_ecg[train_index:], abp[:train_index], abp[train_index:]
    return X_main, X_test, y_main, y_test

def save_train_test_data(dir_path, X_main, X_test, y_main, y_test):
    np.save(dir_path+'x_main.npy', X_main)
    np.save(dir_path+'y_main.npy', y_main)
    np.save(dir_path+'x_test.npy', X_test)
    np.save(dir_path+'y_test.npy', y_test)

def load_train_test_data(dir_path):
    X_train = np.load(dir_path+'x_main.npy',allow_pickle=True)
    y_train = np.load(dir_path+'y_main.npy',allow_pickle=True)
    X_test = np.load(dir_path+'x_test.npy',allow_pickle=True)
    y_test = np.load(dir_path+'y_test.npy',allow_pickle=True)

    return X_train, X_test, y_train, y_test

def convert_to_timeseries(arr, batch_size):
    array = []
    i = 0
    while i < arr.shape[0]-batch_size:
        if (i+batch_size+2) % 3000 == 0:
            i +=36
            continue
            #skip the next 34 rounds

        array.append(arr[i:i+batch_size])
        i+=1
    return np.array(array)

def convert_to_2d(arr, index=None):
    array = []
    if index != None:
        for subject_index in range(arr.shape[0]):
            array.extend(arr[subject_index][:,index])
    else:
        for subject_index in range(arr.shape[0]):
            array.extend(arr[subject_index])
    return np.array(array)

def convert_to_1d(arr):
    array = []
    for i in range(arr.shape[0]):
        if i == 0:
            array.extend(arr[i,:,0])
        else:
            array.extend(arr[i,-1:,0])
    return np.array(array)

def get_and_save_data(dir_path):
    change_to_working_directory()
    X_main, X_test, y_main, y_test = get_Xy_main_test_data(file_path=dir_path+'Part_1.mat')
    save_train_test_data(dir_path=dir_path, X_main=X_main, X_test=X_test, y_main=y_main, y_test=y_test)

def get_train_test_data(dir_path, timewindow=32):
    X_train, X_test, y_train, y_test = load_train_test_data(dir_path)

    ppg_train_conv = convert_to_2d(X_train, 0)
    ecg_train_conv = convert_to_2d(X_train, 1)
    abp_train_conv = convert_to_2d(y_train)
    ppg_test_conv = convert_to_2d(X_test, 0)
    ecg_test_conv = convert_to_2d(X_test, 1)
    abp_test_conv = convert_to_2d(y_test)

    ppg_train_conv_scaled = (ppg_train_conv - ppg_train_conv.min())/(ppg_train_conv.max()-ppg_train_conv.min())
    ecg_train_conv_scaled = (ecg_train_conv - ecg_train_conv.min())/(ecg_train_conv.max()-ecg_train_conv.min())
    abp_train_conv_scaled = (abp_train_conv - abp_train_conv.min())/(abp_train_conv.max()-abp_train_conv.min())
    ppg_test_conv_scaled = (ppg_test_conv - ppg_test_conv.min())/(ppg_test_conv.max()-ppg_test_conv.min())
    ecg_test_conv_scaled = (ecg_test_conv - ecg_test_conv.min())/(ecg_test_conv.max()-ecg_test_conv.min())
    abp_test_conv_scaled = (abp_test_conv - abp_test_conv.min())/(abp_test_conv.max()-abp_test_conv.min())

    ppg_train_timeseries_scaled = convert_to_timeseries(ppg_train_conv_scaled, timewindow)
    ecg_train_timeseries_scaled = convert_to_timeseries(ecg_train_conv_scaled, timewindow)
    abp_train_timeseries_scaled = convert_to_timeseries(abp_train_conv_scaled, timewindow)
    ppg_test_timeseries_scaled = convert_to_timeseries(ppg_test_conv_scaled, timewindow)
    ecg_test_timeseries_scaled = convert_to_timeseries(ecg_test_conv_scaled, timewindow)
    abp_test_timeseries_scaled = convert_to_timeseries(abp_test_conv_scaled, timewindow)

    #put ppg and ecg together to form the encoder input with shape (201554,32,2)
    encoder_input_train = np.stack((ppg_train_timeseries_scaled, ecg_train_timeseries_scaled), axis=2)
    encoder_input_test = np.stack((ppg_test_timeseries_scaled, ecg_test_timeseries_scaled), axis=2)


    #stack the abp as well to form the decoder output with shape (201554,32,1)
    decoder_output_train = np.expand_dims(abp_train_timeseries_scaled, axis=2)
    decoder_output_test = np.expand_dims(abp_test_timeseries_scaled, axis=2)

    train_data = TensorDataset(torch.from_numpy(encoder_input_train).float(), torch.from_numpy(decoder_output_train).float())
    test_data = TensorDataset(torch.from_numpy(encoder_input_test).float(), torch.from_numpy(decoder_output_test).float())
    
    return train_data, encoder_input_train, decoder_output_train, test_data, encoder_input_test, decoder_output_test

def main():
    get_and_save_data(dir_path=DATA)

if __name__ == "__main__":
    main()