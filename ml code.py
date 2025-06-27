import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Set random seed for reproducibility
np.random.seed(42)

# Load the Cleveland heart disease dataset
print("Loading data...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=column_names, na_values='?')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nData overview:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Convert target to binary (0 for no disease, 1 for disease)
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values by dropping rows with missing values
data = data.dropna()
print(f"\nDataset shape after removing rows with missing values: {data.shape}")

# Display class distribution
print("\nClass distribution:")
print(data['target'].value_counts())

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Original Jellyfish Optimization Algorithm for feature selection - SCENARIO 1
class JellyfishOptimization:
    def __init__(self, n_jellyfishes=30, max_iter=100, beta=3, gamma=0.1):
        self.n_jellyfishes = n_jellyfishes
        self.max_iter = max_iter
        self.beta = beta
        self.gamma = gamma
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.title = "Original JOA"
       
    def initialize_population(self, n_features):
        """Initialize jellyfish population with binary vectors"""
        population = np.random.randint(0, 2, size=(self.n_jellyfishes, n_features))
        # Ensure diversity in initial solutions
        for i in range(5):
            num_features = np.random.randint(4, 8)  # Try solutions with 4-7 features
            features = np.random.choice(n_features, num_features, replace=False)
            population[i] = np.zeros(n_features)
            population[i][features] = 1
           
        # Also add a solution with all features selected
        population[5] = np.ones(n_features)
        return population
   
    def fitness_function(self, solution, X, y, classifier='svm'):
        """Calculate fitness using 5-fold cross-validation with specified classifier"""
        selected_features = np.where(solution == 1)[0]
       
        # If no feature or too few features are selected, return high fitness
        if len(selected_features) < 3:
            return 0.8
       
        X_selected = X[:, selected_features]
       
        # Choose classifier
        if classifier == 'svm':
            model = SVC(kernel='rbf', probability=True, C=10)
        elif classifier == 'dt':
            model = DecisionTreeClassifier()
        elif classifier == 'ann':
            model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
        elif classifier == 'adaboost':
            model = AdaBoostClassifier()
        else:
            model = SVC(kernel='rbf', probability=True, C=10)
       
        # Use 5-fold stratified cross-validation for better handling of imbalanced data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_selected, y, cv=skf, scoring='roc_auc')
       
        # Fitness combines error and feature count (weighted)
        error_rate = 1 - np.mean(scores)
       
        # Add feature penalty based on number of features
        if len(selected_features) > 8:
            feature_penalty = 0.03 * (len(selected_features) - 8) / X.shape[1]
        else:
            feature_penalty = 0.01 * len(selected_features) / X.shape[1]
       
        # Heavily weight accuracy over feature count
        fitness = 0.9 * error_rate + 0.1 * feature_penalty
       
        return fitness
   
    def update_position(self, current_position, X_mean, best_jellyfish, t):
        """Update jellyfish position following the paper's equations"""
        n_features = len(current_position)
       
        # Ocean current direction (Eq. 3)
        trend = best_jellyfish - self.beta * np.random.random() * X_mean
       
        # Time control parameter (Eq. 14)
        c_t = abs((1 - t/self.max_iter) * (2 * np.random.random() - 1))
       
        new_position_continuous = current_position.copy().astype(float)
       
        if c_t > 0.5:  # Follow ocean current
            # Motion by ocean current (Eq. 9)
            new_position_continuous = current_position + np.random.random() * trend
        else:  # Group movement
            # Select random jellyfish
            j = np.random.randint(0, self.n_jellyfishes)
            other_jellyfish = self.population[j]
           
            # Active motion based on fitness comparison (Eq. 12 & 13)
            if self.fitness_values[self.current_index] < self.fitness_values[j]:
                new_position_continuous = current_position + np.random.random() * (other_jellyfish - current_position)
            else:
                new_position_continuous = current_position + np.random.random() * (current_position - other_jellyfish)
               
            # Passive motion - small random movement (Eq. 11)
            if np.random.random() < 0.2:  # 20% chance for passive motion
                new_position_continuous += self.gamma * np.random.random() * np.random.choice([-1, 1], n_features)
       
        # Convert to binary using sigmoid function - steeper sigmoid for more decisive selection
        sigmoid = 1 / (1 + np.exp(-12 * (new_position_continuous - 0.5)))
        new_position = np.where(np.random.random(n_features) < sigmoid, 1, 0)
       
        # Ensure at least 3 features are selected
        if np.sum(new_position) < 3:
            random_indices = np.random.choice(n_features, 3, replace=False)
            new_position[random_indices] = 1
       
        return new_position
   
    def optimize(self, X, y, classifier='svm'):
        """Main optimization function"""
        n_features = X.shape[1]
       
        # Initialize population
        self.population = self.initialize_population(n_features)
       
        # Evaluate initial population
        self.fitness_values = np.array([self.fitness_function(solution, X, y, classifier) for solution in self.population])
       
        # Find the best jellyfish
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
       
        print(f"Initial best fitness: {self.best_fitness:.4f}, Selected features: {np.sum(self.best_solution)}/{n_features}")
       
        # Main optimization loop
        for t in range(self.max_iter):
            X_mean = np.mean(self.population, axis=0)
           
            for i in range(self.n_jellyfishes):
                self.current_index = i
               
                # Update position
                new_position = self.update_position(self.population[i], X_mean, self.best_solution, t)
               
                # Evaluate new position
                new_fitness = self.fitness_function(new_position, X, y, classifier)
               
                # Update if better
                if new_fitness < self.fitness_values[i]:
                    self.population[i] = new_position
                    self.fitness_values[i] = new_fitness
                   
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = new_fitness
           
            self.fitness_history.append(self.best_fitness)
           
            # Print progress every 10 iterations
            if (t+1) % 10 == 0 or t == 0:
                selected_count = np.sum(self.best_solution)
                print(f"Iteration {t+1}/{self.max_iter}, Best fitness: {self.best_fitness:.4f}, Selected features: {selected_count}/{n_features}")
       
        # Return selected feature indices
        selected_indices = np.where(self.best_solution == 1)[0]
        return selected_indices, self.best_fitness

# Enhanced Jellyfish Optimization Algorithm for feature selection - SCENARIO 2 (Hard Cap at 7 features with Modified Sigmoid)
class JellyfishOptimizationScenario2:
    def __init__(self, n_jellyfishes=30, max_iter=100, beta=3, gamma=0.1):
        self.n_jellyfishes = n_jellyfishes
        self.max_iter = max_iter
        self.beta = beta
        self.gamma = gamma
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness_history = []
        self.title = "JOA with Hard Cap at 7 Features"
       
    def initialize_population(self, n_features):
        """Initialize jellyfish population with binary vectors"""
        population = np.random.randint(0, 2, size=(self.n_jellyfishes, n_features))
        
        # Create more diverse initial population with focus on 5-7 features
        for i in range(10):
            # Generate solutions with exactly 5, 6, or 7 features
            feature_count = np.random.choice([5, 6, 7])
            features = np.random.choice(n_features, feature_count, replace=False)
            population[i] = np.zeros(n_features)
            population[i][features] = 1
           
        return population
   
    def fitness_function(self, solution, X, y, classifier='svm'):
        """Calculate fitness using 5-fold cross-validation with specified classifier"""
        selected_features = np.where(solution == 1)[0]
       
        # If less than 3 features are selected, return high fitness
        if len(selected_features) < 3:
            return 0.8
        
        # Hard cap at 7 features - if more features, heavily penalize
        if len(selected_features) > 7:
            return 0.7  # High fitness (bad) for solutions with more than 7 features
       
        X_selected = X[:, selected_features]
       
        # Choose classifier
        if classifier == 'svm':
            model = SVC(kernel='rbf', probability=True, C=10)
        elif classifier == 'dt':
            model = DecisionTreeClassifier()
        elif classifier == 'ann':
            model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
        elif classifier == 'adaboost':
            model = AdaBoostClassifier()
        else:
            model = SVC(kernel='rbf', probability=True, C=10)
       
        # Use 5-fold stratified cross-validation for better handling of imbalanced data
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_selected, y, cv=skf, scoring='roc_auc')
       
        # Fitness combines error and feature count (weighted)
        error_rate = 1 - np.mean(scores)
       
        # Add feature penalty - progressive penalty to encourage fewer features
        # Encourage solutions with 6 or fewer features with mild penalty
        if len(selected_features) <= 6:
            feature_penalty = 0.005 * len(selected_features) / X.shape[1]
        else:  # Stronger penalty for exactly 7 features
            feature_penalty = 0.02 * len(selected_features) / X.shape[1]
       
        # Balance accuracy vs feature count
        fitness = 0.9 * error_rate + 0.1 * feature_penalty
       
        return fitness
   
    def update_position(self, current_position, X_mean, best_jellyfish, t):
        """Update jellyfish position with modified sigmoid function"""
        n_features = len(current_position)
       
        # Ocean current direction (Eq. 3)
        trend = best_jellyfish - self.beta * np.random.random() * X_mean
       
        # Time control parameter (Eq. 14)
        c_t = abs((1 - t/self.max_iter) * (2 * np.random.random() - 1))
       
        new_position_continuous = current_position.copy().astype(float)
       
        if c_t > 0.5:  # Follow ocean current
            # Motion by ocean current (Eq. 9)
            new_position_continuous = current_position + np.random.random() * trend
        else:  # Group movement
            # Select random jellyfish
            j = np.random.randint(0, self.n_jellyfishes)
            other_jellyfish = self.population[j]
           
            # Active motion based on fitness comparison (Eq. 12 & 13)
            if self.fitness_values[self.current_index] < self.fitness_values[j]:
                new_position_continuous = current_position + np.random.random() * (other_jellyfish - current_position)
            else:
                new_position_continuous = current_position + np.random.random() * (current_position - other_jellyfish)
               
            # Passive motion - small random movement (Eq. 11)
            if np.random.random() < 0.2:  # 20% chance for passive motion
                new_position_continuous += self.gamma * np.random.random() * np.random.choice([-1, 1], n_features)
       
        # Modified sigmoid function - use softer sigmoid (6 instead of 12) for less decisive selection
        sigmoid = 1 / (1 + np.exp(-6 * (new_position_continuous - 0.5)))
        new_position = np.where(np.random.random(n_features) < sigmoid, 1, 0)
       
        # Ensure at least 3 features are selected and no more than 7
        feature_count = np.sum(new_position)
        if feature_count < 3:
            # Add random features to reach minimum 3
            zeros_indices = np.where(new_position == 0)[0]
            add_count = 3 - feature_count
            if len(zeros_indices) >= add_count:
                random_indices = np.random.choice(zeros_indices, add_count, replace=False)
                new_position[random_indices] = 1
        elif feature_count > 7:
            # Remove random features to meet maximum 7
            ones_indices = np.where(new_position == 1)[0]
            remove_count = feature_count - 7
            if len(ones_indices) >= remove_count:
                random_indices = np.random.choice(ones_indices, remove_count, replace=False)
                new_position[random_indices] = 0
       
        return new_position
   
    def optimize(self, X, y, classifier='svm'):
        """Main optimization function"""
        n_features = X.shape[1]
       
        # Initialize population
        self.population = self.initialize_population(n_features)
       
        # Evaluate initial population
        self.fitness_values = np.array([self.fitness_function(solution, X, y, classifier) for solution in self.population])
       
        # Find the best jellyfish
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
       
        print(f"Initial best fitness: {self.best_fitness:.4f}, Selected features: {np.sum(self.best_solution)}/{n_features}")
       
        # Main optimization loop
        for t in range(self.max_iter):
            X_mean = np.mean(self.population, axis=0)
           
            for i in range(self.n_jellyfishes):
                self.current_index = i
               
                # Update position
                new_position = self.update_position(self.population[i], X_mean, self.best_solution, t)
               
                # Evaluate new position
                new_fitness = self.fitness_function(new_position, X, y, classifier)
               
                # Update if better
                if new_fitness < self.fitness_values[i]:
                    self.population[i] = new_position
                    self.fitness_values[i] = new_fitness
                   
                    # Update global best
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = new_fitness
           
            self.fitness_history.append(self.best_fitness)
           
            # Print progress every 10 iterations
            if (t+1) % 10 == 0 or t == 0:
                selected_count = np.sum(self.best_solution)
                print(f"Iteration {t+1}/{self.max_iter}, Best fitness: {self.best_fitness:.4f}, Selected features: {selected_count}/{n_features}")
       
        # Return selected feature indices
        selected_indices = np.where(self.best_solution == 1)[0]
        return selected_indices, self.best_fitness

# Function to evaluate classifier and get ROC data
def evaluate_classifier_with_roc(X, y, classifier, feature_indices=None, n_folds=10, label=""):
    if feature_indices is not None:
        X_selected = X[:, feature_indices]
    else:
        X_selected = X
   
    # Performance metrics
    accuracies = []
    sensitivities = []  # Recall for positive class
    specificities = []  # Recall for negative class
    f1_scores = []
    aucs = []
    
    # For ROC curve plotting
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    # Create stratified KFold object for better handling of imbalanced data
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
   
    # Perform cross-validation
    fold = 1
    for train_index, test_index in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        # Train model
        classifier.fit(X_train, y_train)
       
        # Predict
        y_pred = classifier.predict(X_test)
       
        # Get prediction probabilities if available
        try:
            y_prob = classifier.predict_proba(X_test)[:, 1]
            auc_value = roc_auc_score(y_test, y_prob)
            
            # ROC curve for this fold
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
        except:
            auc_value = 0  # In case the classifier doesn't support predict_proba
       
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        f1 = f1_score(y_test, y_pred)
       
        # Store metrics
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1)
        aucs.append(auc_value)
       
        fold += 1
   
    # Calculate average metrics
    return {
        'Label': label,
        'Accuracy': np.mean(accuracies) * 100,
        'Sensitivity': np.mean(sensitivities) * 100,
        'Specificity': np.mean(specificities) * 100,
        'F1_Score': np.mean(f1_scores),
        'AUC': np.mean(aucs) * 100,
        'tprs': tprs,
        'mean_fpr': mean_fpr
    }

# Main execution
print("\nApplying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Use RobustScaler for better handling of outliers
print("\nApplying RobustScaler for feature scaling...")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Create different classifiers
classifiers = {
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
    'DT': DecisionTreeClassifier(random_state=42),
    'ANN': MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

# --------------------------------------
# SCENARIO 1: Original JOA
# --------------------------------------
print("\nRunning Jellyfish Optimization Algorithm for feature selection (Scenario 1 - Original)...")
joa1 = JellyfishOptimization(n_jellyfishes=30, max_iter=50, beta=3, gamma=0.1)  # Reduced iterations for demo
selected_features1, best_fitness1 = joa1.optimize(X_scaled, y_resampled, classifier='svm')

# Get original feature names
feature_names = X.columns
selected_feature_names1 = [feature_names[i] for i in selected_features1]

print(f"\nSelected features ({len(selected_features1)}): {selected_feature_names1}")
print(f"Best fitness: {best_fitness1:.4f}")

# --------------------------------------
# SCENARIO 2: JOA with Hard Cap at 7 Features
# --------------------------------------
print("\nRunning Jellyfish Optimization Algorithm for feature selection (Scenario 2 - Hard Cap at 7 Features)...")
joa2 = JellyfishOptimizationScenario2(n_jellyfishes=30, max_iter=50, beta=3, gamma=0.1)  # Reduced iterations for demo
selected_features2, best_fitness2 = joa2.optimize(X_scaled, y_resampled, classifier='svm')

# Get original feature names for scenario 2
selected_feature_names2 = [feature_names[i] for i in selected_features2]

print(f"\nSelected features ({len(selected_features2)}): {selected_feature_names2}")
print(f"Best fitness: {best_fitness2:.4f}")

# --------------------------------------
# SCENARIO 3: Using all features (no feature selection)
# --------------------------------------
print("\nUsing all features for classification (no feature selection)...")
print(f"Total features: {X_scaled.shape[1]}")
print(f"Feature names: {list(X.columns)}")

# --------------------------------------
# EVALUATE ALL SCENARIOS
# --------------------------------------
# Results dictionary for each scenario
all_results = {
    'Scenario 1': {},
    'Scenario 2': {},
    'Scenario 3': {}
}

# For ROC curve comparison
plt.figure(figsize=(18, 14))

# For each classifier, evaluate all scenarios
for i, (name, classifier) in enumerate(classifiers.items()):
    plt.subplot(2, 2, i+1)
    
    # Evaluate Scenario 1: Original JOA
    print(f"\nEvaluating {name} with Scenario 1 (Original JOA)...")
    metrics1 = evaluate_classifier_with_roc(X_scaled, y_resampled, classifier, selected_features1, 
                                           label=f"{name} - Scenario 1")
    all_results['Scenario 1'][name] = metrics1
    
    # Plot ROC for Scenario 1
    mean_tpr = np.mean(metrics1['tprs'], axis=0)
    mean_auc = metrics1['AUC']/100
    plt.plot(metrics1['mean_fpr'], mean_tpr, 
             label=f'S1: Original JOA (AUC = {mean_auc:.3f})',
             color='blue', lw=2)
    
    # Evaluate Scenario 2: JOA with Hard Cap at 7 Features
    print(f"\nEvaluating {name} with Scenario 2 (JOA with Hard Cap at 7)...")
    metrics2 = evaluate_classifier_with_roc(X_scaled, y_resampled, classifier, selected_features2, 
                                           label=f"{name} - Scenario 2")
    all_results['Scenario 2'][name] = metrics2
    
    # Plot ROC for Scenario 2
    mean_tpr = np.mean(metrics2['tprs'], axis=0)
    mean_auc = metrics2['AUC']/100
    plt.plot(metrics2['mean_fpr'], mean_tpr, 
             label=f'S2: JOA with Hard Cap (AUC = {mean_auc:.3f})',
             color='green', lw=2)
    
    # Evaluate Scenario 3: All Features
    print(f"\nEvaluating {name} with Scenario 3 (All Features)...")
    metrics3 = evaluate_classifier_with_roc(X_scaled, y_resampled, classifier, None, 
                                           label=f"{name} - Scenario 3")
    all_results['Scenario 3'][name] = metrics3
    
    # Plot ROC for Scenario 3
    mean_tpr = np.mean(metrics3['tprs'], axis=0)
    mean_auc = metrics3['AUC']/100
    plt.plot(metrics3['mean_fpr'], mean_tpr, 
             label=f'S3: All Features (AUC = {mean_auc:.3f})',
             color='red', lw=2)
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    
    # Set plot details
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {name}')
    plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

# Compare features selected by JOA algorithms
plt.figure(figsize=(12, 6))

# Get all unique features from both scenarios
all_features = list(set(selected_feature_names1 + selected_feature_names2))
all_features.sort()  # Sort for better visualization

# Create values for each scenario (1 if feature was selected, 0 otherwise)
scenario1_values = [1 if feature in selected_feature_names1 else 0 for feature in all_features]
scenario2_values = [1 if feature in selected_feature_names2 else 0 for feature in all_features]

# Plotting
x = np.arange(len(all_features))
width = 0.35

plt.bar(x - width/2, scenario1_values, width, label='Scenario 1: Original JOA', color='blue')
plt.bar(x + width/2, scenario2_values, width, label='Scenario 2: JOA with Hard Cap at 7', color='green')

plt.xlabel('Features')
plt.ylabel('Selected (1) / Not Selected (0)')
plt.title('Feature Selection Comparison Between JOA Scenarios')
plt.xticks(x, all_features, rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot JOA convergence comparison
plt.figure(figsize=(12, 6))
plt.plot(joa1.fitness_history, label='Scenario 1: Original JOA', color='blue')
plt.plot(joa2.fitness_history, label='Scenario 2: JOA with Hard Cap at 7', color='green')
plt.title('Fitness Convergence Comparison Between JOA Scenarios')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value (Lower is Better)')
plt.legend()
plt.grid(True)
plt.show()

# Create and display summary tables
models = list(classifiers.keys())
metrics = ['Accuracy (%)', 'Sensitivity (%)', 'Specificity (%)', 'AUC (%)']

# Comparison of performance metrics between scenarios
plt.figure(figsize=(15, 10))

for i, metric_name in enumerate(metrics):
    metric = metric_name.split()[0]
    plt.subplot(2, 2, i+1)
    
    # Extract metric values for each model and scenario
    scenario1_values = [all_results['Scenario 1'][model][metric] for model in models]
    scenario2_values = [all_results['Scenario 2'][model][metric] for model in models]
    scenario3_values = [all_results['Scenario 3'][model][metric] for model in models]
    
    # Set up bar positions
    x = np.arange(len(models))
    width = 0.25
    
    # Plot bars
    plt.bar(x - width, scenario1_values, width, label='Scenario 1: Original JOA', color='blue')
    plt.bar(x, scenario2_values, width, label='Scenario 2: JOA with Hard Cap at 7', color='green')
    plt.bar(x + width, scenario3_values, width, label='Scenario 3: All Features', color='red')
    
    # Add value labels on bars
    for j, v in enumerate(scenario1_values):
        plt.text(j - width, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    for j, v in enumerate(scenario2_values):
        plt.text(j, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    for j, v in enumerate(scenario3_values):
        plt.text(j + width, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.title(f'Comparison of {metric_name} Between Scenarios')
    plt.xticks(x, models)
    plt.ylim(70, 100)  # Set y-axis from 70% to 100% for better visualization
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Create a summary metrics comparison table
summary_data = {
    'Model': [],
    'S1 Accuracy (%)': [],
    'S2 Accuracy (%)': [],
    'S3 Accuracy (%)': [],
    'S1 AUC (%)': [],
    'S2 AUC (%)': [],
    'S3 AUC (%)': [],
    'S1 Feature Count': [],
    'S2 Feature Count': [],
    'S3 Feature Count': []
}

for model in models:
    summary_data['Model'].append(model)
    summary_data['S1 Accuracy (%)'].append(all_results['Scenario 1'][model]['Accuracy'])
    summary_data['S2 Accuracy (%)'].append(all_results['Scenario 2'][model]['Accuracy'])
    summary_data['S3 Accuracy (%)'].append(all_results['Scenario 3'][model]['Accuracy'])
    summary_data['S1 AUC (%)'].append(all_results['Scenario 1'][model]['AUC'])
    summary_data['S2 AUC (%)'].append(all_results['Scenario 2'][model]['AUC'])
    summary_data['S3 AUC (%)'].append(all_results['Scenario 3'][model]['AUC'])
    summary_data['S1 Feature Count'].append(len(selected_features1))
    summary_data['S2 Feature Count'].append(len(selected_features2))
    summary_data['S3 Feature Count'].append(X_scaled.shape[1])

summary_df = pd.DataFrame(summary_data)

print("\nScenario Comparison Summary:")
display(summary_df)

print("\nFeature comparison between scenarios:")
print(f"Scenario 1 features ({len(selected_features1)}): {selected_feature_names1}")
print(f"Scenario 2 features ({len(selected_features2)}): {selected_feature_names2}")
print(f"Scenario 3 features ({X_scaled.shape[1]}): {list(X.columns)}")

# Find the best overall model and scenario based on AUC
best_model = None
best_scenario = None
best_auc = 0

for scenario in all_results:
    for model in all_results[scenario]:
        if all_results[scenario][model]['AUC'] > best_auc:
            best_auc = all_results[scenario][model]['AUC']
            best_model = model
            best_scenario = scenario

print(f"\nBest overall performance: {best_model} with {best_scenario}")
print(f"AUC: {best_auc:.2f}%")
print(f"Accuracy: {all_results[best_scenario][best_model]['Accuracy']:.2f}%")
print(f"Sensitivity: {all_results[best_scenario][best_model]['Sensitivity']:.2f}%")
print(f"Specificity: {all_results[best_scenario][best_model]['Specificity']:.2f}%")