import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 

# Check for available device
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS for Apple Silicon
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced Model Architecture with BatchNorm and Dropout
class Adaptive_MLP(nn.Module):
    def __init__(self, embedding_dim=21952, num_classes=6, dropout_rate=0.2):
        super(Adaptive_MLP, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.fc4 = nn.Linear(256, num_classes)

    def forward(self, embedding):
        x = F.relu(self.bn1(self.fc1(embedding)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        return F.softmax(self.fc4(x), dim=1)

# Genetic Algorithm Components
def initialize_population(model, population_size=50, deviation=0.5):
    """Initialize population with adaptive deviation and proper BatchNorm handling."""
    population = []
    base_state_dict = model.state_dict()
    
    for _ in range(population_size):
        individual = {}
        for name, param in base_state_dict.items():
            if 'running_mean' in name or 'running_var' in name or 'num_batches_tracked' in name:
                individual[name] = param.clone()  # Copy BatchNorm stats directly
            else:
                layer_dev = deviation
                if 'fc4' in name:  # Output layer
                    layer_dev *= 0.5
                elif 'fc3' in name:  # Deep layer
                    layer_dev *= 0.7
                individual[name] = param.clone() + layer_dev * torch.randn_like(param)
        population.append(individual)
    return population

def evaluate_fitness(model, population, data_loader, criterion, device):
    """Evaluate fitness of each individual in the population."""
    fitness_scores = []
    for individual in population:
        current_state = model.state_dict()
        current_state.update(individual)
        model.load_state_dict(current_state)
        
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        fitness = -0.3 * total_loss + 0.7 * accuracy 
        fitness_scores.append((individual, fitness))
    
    return fitness_scores

def select_parents(fitness_scores, num_parents=10, tournament_size=5):
    """Select parents using elitism and tournament selection."""
    sorted_pop = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
    elite = [item[0] for item in sorted_pop[:int(0.15 * len(fitness_scores))]]  # 15% elitism
    
    parents = elite.copy()
    while len(parents) < num_parents:
        tournament = random.sample(fitness_scores, tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        parents.append(winner)
    
    return parents

def crossover(parents):
    """Create offspring using weighted average crossover."""
    offspring = []
    for i in range(len(parents)):
        for j in range(i + 1, min(i + 3, len(parents))):  # Each parent mates with up to 2 others
            parent1, parent2 = parents[i], parents[j]
            child = {}
            for key in parent1:
                if 'running_' in key or 'num_batches_tracked' in key:
                    child[key] = parent1[key].clone()  # Copy BatchNorm stats
                elif 'weight' in key:
                    alpha = random.uniform(0.3, 0.7)
                    child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
                else:
                    child[key] = parent1[key] if random.random() > 0.5 else parent2[key]
            offspring.append(child)
    return offspring

def mutate(offspring, generation, total_generations, initial_rate=0.3, final_rate=0.05):
    """
    Apply adaptive mutation with generation-based decay.
    
    Args:
        offspring: List of individuals to mutate.
        generation: Current generation number.
        total_generations: Total number of generations.
        initial_rate: Initial mutation rate.
        final_rate: Final mutation rate.
    
    Returns:
        Mutated offspring.
    """
    # Adaptive mutation rate
    mutation_rate = initial_rate - (initial_rate - final_rate) * (generation / total_generations)
    
    for individual in offspring:
        for key in individual:
            # Skip BatchNorm running stats
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                continue
                
            # Layer-specific mutation rate
            layer_rate = mutation_rate * (0.5 if 'fc4' in key else 1.0)  # Lower rate for output layer
            
            # Create a mask with the same shape as the tensor
            mask = torch.rand_like(individual[key]) < layer_rate
            
            # Apply mutation only to selected elements
            mutation_noise = 0.1 * torch.randn_like(individual[key])
            individual[key] = torch.where(mask, individual[key] + mutation_noise, individual[key])
    
    return offspring

# Genetic Algorithm Training with Early Stopping
def genetic_algorithm_training(model, train_loader, val_loader, num_generations=80, population_size=200, save_path="best_model.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    population = initialize_population(model, population_size)
    
    # Lists to store plotting metrics
    train_losses = []
    val_losses = []
    test_losses = []

    best_fitness = -float('inf')
    best_val_acc = 0
    no_improvement = 0
    best_model_state = None
    
    for generation in range(num_generations):
        print(f"Generation {generation + 1}/{num_generations}")
        
        # Evaluate population
        fitness_scores = evaluate_fitness(model, population, train_loader, criterion, device)
        current_best_fitness = max([fs[1] for fs in fitness_scores])
        print(f"Best fitness in generation: {current_best_fitness:.4f}")
        
        # Select parents and create offspring
        num_parents = max(10, int(population_size * 0.2))
        parents = select_parents(fitness_scores, num_parents)
        offspring = crossover(parents)
        offspring = mutate(offspring, generation, num_generations)
        population = parents + offspring
        
        # Validate current best model
        best_individual = max(fitness_scores, key=lambda x: x[1])[0]
        current_state = model.state_dict()
        current_state.update(best_individual)
        model.load_state_dict(current_state)
        
        # Calculate training loss
        train_loss = calculate_loss(model, train_loader, criterion, device)
        train_losses.append(train_loss)
        
        # Calculate validation loss
        val_loss = calculate_loss(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Calculate test loss
        test_loss = calculate_loss(model, test_loader, criterion, device)
        test_losses.append(test_loss)

        val_acc, _, _ = evaluate_model(model, val_loader, device)
        print(f"Validation accuracy: {val_acc:.4f}")
        
        # Early stopping and save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)  # Save the best model to disk
            print(f"New best model saved to {save_path}")
            no_improvement = 0
        else:
            no_improvement += 1
        
        if no_improvement >= 15:
            print("Early stopping triggered.")
            break
    
    # Load best model
    # After training, plot and save the losses
    plot_losses(train_losses, val_losses, test_losses)
    model.load_state_dict(torch.load(save_path))
    return model

def calculate_loss(model, data_loader, criterion, device):
    """
    Calculate the loss of the model on a given dataset.
    
    Args:
        model: The neural network model.
        data_loader: DataLoader for the dataset.
        criterion: Loss function.
        device: The device to run the evaluation on.
    
    Returns:
        loss: The average loss over the dataset.
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    
    return total_loss / total_samples

def plot_losses(train_losses, val_losses, test_losses):
    """
    Plot the training, validation, and test losses over generations.
    
    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        test_losses: List of test losses.
        num_generations: Total number of generations.
    """
    generations = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, train_losses, label="Training Loss", marker="o")
    plt.plot(generations, val_losses, label="Validation Loss", marker="o")
    plt.plot(generations, test_losses, label="Test Loss", marker="o")
    # plt.plot(test_losses, linestyle="--", color="red", label=f"Test Loss({test_losses:.4f})")

    plt.xlabel("Generation")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Generations")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save the plot to disk
    # plt.savefig(save_path)
    # print(f"Plot saved to {save_path}")

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model's performance on a given dataset.
    
    Args:
        model: The neural network model.
        data_loader: DataLoader for the dataset to evaluate.
        device: The device to run the evaluation on (e.g., 'cuda', 'cpu').
    
    Returns:
        accuracy: The accuracy of the model on the dataset.
        all_predictions: List of all predicted labels.
        all_targets: List of all ground truth labels.
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            outputs = model(inputs)  # Forward pass
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Calculate accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions and targets for further analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    accuracy = correct / total  # Compute accuracy
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy, all_predictions, all_targets

# Custom Dataset
class CustomMLPDataset(Dataset):
    def __init__(self, data_file, label_file):
        data = torch.load(data_file)
        labels = torch.load(label_file)
        
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data).float()
        else:
            self.data = data.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Main Execution
if __name__ == "__main__":
    # File paths
    train_data_file = 'Embedding/flatten_train_data.pt'
    train_label_file = 'Embedding/flatten_train_labels.pt'
    val_data_file = 'Embedding/flatten_val_data.pt'
    val_label_file = 'Embedding/flatten_val_labels.pt'
    
    test_data_file = 'Embedding/flatten_test_data.pt'  # Add test data
    test_label_file = 'Embedding/flatten_test_labels.pt'  # Add test labels

    # Create datasets and data loaders
    train_dataset = CustomMLPDataset(train_data_file, train_label_file)
    val_dataset = CustomMLPDataset(val_data_file, val_label_file)
    test_dataset = CustomMLPDataset(test_data_file, test_label_file)  # Test dataset
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)  # Test loader

    # Initialize model
    model = Adaptive_MLP(embedding_dim=21952, num_classes=6).to(device)
    
    # Train with genetic algorithm
    trained_model = genetic_algorithm_training(
        model, 
        train_loader, 
        val_loader, 
        num_generations=80, 
        population_size=50, 
        save_path="models/best_model_flatten.pth"  # Specify the save path here
    )
    
    # Evaluate final model
    val_acc, _, _ = evaluate_model(trained_model, val_loader, device)
    print(f"Final validation accuracy: {val_acc:.4f}")
