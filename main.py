import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparams
input_size = 28 
sequence_length = 28
num_layers = 2
hidden_size = 128
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#readability stuff
showExampleData = True
if torch.cuda.is_available() == False:
    print("cuda unavailable, running on cpu")
#print(f"torch devices: {torch.cuda.device_count()}")
#print(f"current device: {torch.cuda.get_device_name(0)}")
    
#TODO: perfrom frame extraction from video, pass images through cnn feature extraction network, then pass those through lstm

#MNIST import
train_dataset = torchvision.datasets.MNIST(root = './data', train=True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root = './data', train=False, transform = transforms.ToTensor())

#data loaders
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

#looking at one batch of this data
examples = iter(train_loader)
samples, labels = next(examples)

if(showExampleData):
    print(samples.shape, labels.shape)
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.imshow(samples[i][0],cmap='gray')
    plt.show()

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        # x -> (batch_size, sequence length, input_size)
        self.fc = nn.Linear(hidden_size, num_classes) #only using the last time sequence frame to perform classification

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #initial tensor state

        out, _ =self.lstm(x, (h0, c0))
        # out shape: batch_size, seq_length, hidden_size
        # out (N, 128, 128)
        out = out[:, -1, :] #only taking last item of sequence
        # out (n, 128)
        out = self.fc(out)
        return out

#defining model
model = RNN(input_size, hidden_size, num_layers, num_classes)
#sending model to GPU
model.to(device)

#loss and optimizer
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # original shape: [100, 1, 28, 28]
        # resized shape: [100, 28, 28] 
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criteria(outputs, labels)

        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #ouputs
        if (i+1) % 100 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.5f}')
#test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)

        #this is value, index, but only care about index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    accuracy = 100 * n_correct / n_samples
    print(f"accuracy: {accuracy:.3f}")