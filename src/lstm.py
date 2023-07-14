import torch
import torch.nn as nn
import torch.optim as optim
from data_processing import convert_to_1d
from sklearn.metrics import mean_squared_error

class lstm_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(lstm_model, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def get_mse(model, encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    output_trainset = model(torch.from_numpy(encoder_input_train[0:7000]).to(device).float())
    output_testset = model(torch.from_numpy(encoder_input_test[0:7000]).to(device).float())

    output_testset_1d = convert_to_1d(output_testset.cpu().detach().numpy())
    decoder_output_test_1d = convert_to_1d(decoder_output_test)
    output_trainset_1d = convert_to_1d(output_trainset.cpu().detach().numpy())
    decoder_output_train_1d = convert_to_1d(decoder_output_train)

    mse_train = mean_squared_error(decoder_output_train_1d[:output_trainset_1d.shape[0]], output_trainset_1d)
    mse_test = mean_squared_error(decoder_output_test_1d[:output_testset_1d.shape[0]], output_testset_1d)

    return mse_train, mse_test

def train_lstm(model, device, train_loader, num_epochs=40, lr=0.0003, encoder_input_test=None, decoder_output_test=None):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #train the model
    num_epochs = num_epochs
    loss_curve = []
    train_loss = []
    test_loss = []

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(inputs)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            #convert input to numpy
            encoder_input_train = inputs.cpu().detach().numpy()
            decoder_output_train = labels.cpu().detach().numpy()

            if (i+1) % 100 == 0:
                print('Epoch: {}/{}, Step: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                loss_curve.append(loss.item())
            if encoder_input_test is not None and decoder_output_test is not None:
                if (i + 1) % 100 == 0:
                    train_mse, test_mse = get_mse(model, encoder_input_train, decoder_output_train, encoder_input_test, decoder_output_test)
                    train_loss.append(train_mse)
                    test_loss.append(test_mse)

    if encoder_input_test is not None and decoder_output_test is not None:
        return loss_curve, train_loss, test_loss
    else:
        return loss_curve