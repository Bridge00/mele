import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def get_metrics(all_preds, all_labels):


    acc = np.mean(np.array(all_preds) == np.array(all_labels))

    return 0, 0,0, 0, acc

class CustomDataset(Dataset):
    def __init__(self, X, y):
        #print(type(X.to_numpy()))
        self.X = torch.tensor(X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
def train_val_nn(X_train, y_train, X_val, y_val):
#for trial in range(10):
    print('train nn')

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    # Convert data to PyTorch tensors


    input_dim = X_train.shape[1]
    output_dim = 1


    model = BinaryClassifier(input_dim)


    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5000

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            #print(inputs, labels)
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            #print(outputs.shape, labels.shape)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Validation step
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []#np.array([])
    all_preds = []#np.array([])
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()
            #print(outputs.data)
            #_, predicted = torch.max(outputs.data, 1)
            predicted = torch.round(outputs)
            #print(predicted)
            all_preds += [int(p.item()) for p in predicted]#list(predicted.numpy())
            all_labels += list(labels.numpy())
            
            #correct += (predicted == labels).sum().item()
    return all_preds
    # tpr, fpr, tnr, fnr, acc = get_metrics(all_preds, all_labels)

