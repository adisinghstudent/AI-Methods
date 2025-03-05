import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist():
    """
    Load and prepare the MNIST dataset.
    Returns train and test data as tensors.
    """
    # Define transformations: Convert to tensor
    transform = transforms.ToTensor()
    
    # Download and load training data
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return train_loader, test_loader


def build_model(cnn=True):
    if not cnn:
        # Fully connected neural network
        model = nn.Sequential(
            nn.Flatten(),  # Input is multidimensional, flattened to single dimension
            nn.Linear(28*28, 128),  # Hidden layer with 16 neurons
            nn.BatchNorm1d(128),       # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),           # Add dropout
            nn.Linear(128, 64),        # Additional hidden layer
             nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10),  # Output layer with 10 neurons (one per digit)
        )
    else:
        # Convolutional neural network
        model = nn.Sequential(
            # Conv layer with 8 filters and 2x2 kernel
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),       # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            #One more convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),       # Add batch normalization
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            
            #Pieced Together
            nn.Flatten(),  # Flatten dimensions before output
            nn.Linear(64 * 7 * 7, 128),  # Output layer
            nn.BatchNorm1d(128),       # Add batch normalization
            nn.ReLU(),
            nn.Dropout(0.2),           # Add dropout
            nn.Linear(128, 10),  # Output layer
        )
    
    return model


if __name__ == "__main__":
    # Hyperparameters
    epochs = 10
    learning_rate = 0.1

    # Load MNIST dataset
    train_loader, test_loader = load_mnist()
    
    # Build model (set cnn=True for convolutional network, False for MLP)
    model = build_model(cnn=False)
    
    # Print model architecture
    print(model)
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Train the model
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass, backward pass, optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.3f}')
    
    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    print(f"Test accuracy: {test_acc:.2f}%")