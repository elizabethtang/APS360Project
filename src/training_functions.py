import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from lstm import lstm_model, train_lstm
from data_processing import get_and_save_data, get_train_test_data, convert_to_1d
from globals import DATA

def get_data_and_model(input_dim, hidden_dim, layer_dim, output_dim, batch_size = 128, time_window = 32):
    train_data, encoder_input_train, decoder_output_train, test_data, encoder_input_test, decoder_output_test = get_train_test_data(dir_path=DATA, timewindow=time_window)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    model = lstm_model(input_dim, hidden_dim, layer_dim, output_dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)



    return model, train_loader, encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test




def view_train_test_result(model, encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_trainset = model(torch.from_numpy(encoder_input_train[0:1280]).to(device).float())
    output_testset = model(torch.from_numpy(encoder_input_test[0:1280]).to(device).float())

    output_testset_1d = convert_to_1d(output_testset.cpu().detach().numpy())
    decoder_output_test_1d = convert_to_1d(decoder_output_test)
    output_trainset_1d = convert_to_1d(output_trainset.cpu().detach().numpy())
    decoder_output_train_1d = convert_to_1d(decoder_output_train)

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 1, 1)
    plt.plot(output_testset_1d[:800], label='test')
    plt.title('Validation Result')
    plt.ylabel('scaled ABP')
    plt.plot(decoder_output_test_1d[:800], label='ground truth')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(output_trainset_1d[:800], label='train')
    plt.title('Training Result')
    plt.ylabel('scaled ABP')
    plt.plot(decoder_output_train_1d[:800], label='ground truth')
    plt.legend()








