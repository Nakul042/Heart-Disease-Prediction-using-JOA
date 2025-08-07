import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import time
import io

# Set page config
st.set_page_config(
    page_title="Jellyfish Optimization Algorithm - Feature Selection",
    page_icon="🐙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🐙 Jellyfish Optimization Algorithm</h1>', unsafe_allow_html=True)
st.markdown("### Feature Selection for Machine Learning Classification")

# Sidebar for configuration
st.sidebar.header("🔧 Configuration")

# Dataset selection
dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Heart Disease (Cleveland)", "Breast Cancer", "Diabetes", "Credit Card Fraud"]
)

# Algorithm parameters
st.sidebar.subheader("🐙 JOA Parameters")
n_jellyfishes = st.sidebar.slider("Number of Jellyfishes", 10, 100, 30)
max_iterations = st.sidebar.slider("Maximum Iterations", 20, 200, 50)
beta = st.sidebar.slider("Beta Parameter", 1.0, 5.0, 3.0, 0.1)
gamma = st.sidebar.slider("Gamma Parameter", 0.01, 0.5, 0.1, 0.01)

# Classifier selection
classifier_option = st.sidebar.selectbox(
    "Select Classifier",
    ["SVM", "Decision Tree", "Neural Network", "AdaBoost", "Random Forest"]
)

# Feature selection scenario
scenario_option = st.sidebar.selectbox(
    "Feature Selection Scenario",
    ["Original JOA", "JOA with Hard Cap (7 features)", "All Features"]
)

# Load datasets function
def load_dataset(dataset_name):
    """Load different datasets"""
    if dataset_name == "Heart Disease (Cleveland)":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
                       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        data = pd.read_csv(url, names=column_names, na_values='?')
        data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
        
    elif dataset_name == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        data['target'] = cancer.target
        
    elif dataset_name == "Diabetes":
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        # Convert to binary classification (above/below median)
        median_target = np.median(diabetes.target)
        data['target'] = (diabetes.target > median_target).astype(int)
        
    elif dataset_name == "Credit Card Fraud":
        # Using a synthetic dataset for demonstration
        np.random.seed(42)
        n_samples = 1000
        n_features = 20
        
        # Generate synthetic credit card data
        features = np.random.randn(n_samples, n_features)
        # Create target with some correlation to features
        target = (features[:, 0] + features[:, 5] + features[:, 10] + np.random.randn(n_samples) * 0.5 > 0).astype(int)
        
        feature_names = [f'feature_{i}' for i in range(n_features)]
        data = pd.DataFrame(features, columns=feature_names)
        data['target'] = target
    
    # Handle missing values
    data = data.dropna()
    return data

# Jellyfish Optimization Algorithm classes (same as your original code)
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
        population = np.random.randint(0, 2, size=(self.n_jellyfishes, n_features))
        for i in range(5):
            num_features = np.random.randint(4, 8)
            features = np.random.choice(n_features, num_features, replace=False)
            population[i] = np.zeros(n_features)
            population[i][features] = 1
        population[5] = np.ones(n_features)
        return population
   
    def fitness_function(self, solution, X, y, classifier='svm'):
        selected_features = np.where(solution == 1)[0]
       
        if len(selected_features) < 3:
            return 0.8
       
        X_selected = X[:, selected_features]
       
        if classifier == 'svm':
            model = SVC(kernel='rbf', probability=True, C=10)
        elif classifier == 'dt':
            model = DecisionTreeClassifier()
        elif classifier == 'ann':
            model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
        elif classifier == 'adaboost':
            model = AdaBoostClassifier()
        elif classifier == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = SVC(kernel='rbf', probability=True, C=10)
       
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_selected, y, cv=skf, scoring='roc_auc')
       
        error_rate = 1 - np.mean(scores)
       
        if len(selected_features) > 8:
            feature_penalty = 0.03 * (len(selected_features) - 8) / X.shape[1]
        else:
            feature_penalty = 0.01 * len(selected_features) / X.shape[1]
       
        fitness = 0.9 * error_rate + 0.1 * feature_penalty
       
        return fitness
   
    def update_position(self, current_position, X_mean, best_jellyfish, t):
        n_features = len(current_position)
       
        trend = best_jellyfish - self.beta * np.random.random() * X_mean
        c_t = abs((1 - t/self.max_iter) * (2 * np.random.random() - 1))
       
        new_position_continuous = current_position.copy().astype(float)
       
        if c_t > 0.5:
            new_position_continuous = current_position + np.random.random() * trend
        else:
            j = np.random.randint(0, self.n_jellyfishes)
            other_jellyfish = self.population[j]
           
            if self.fitness_values[self.current_index] < self.fitness_values[j]:
                new_position_continuous = current_position + np.random.random() * (other_jellyfish - current_position)
            else:
                new_position_continuous = current_position + np.random.random() * (current_position - other_jellyfish)
               
            if np.random.random() < 0.2:
                new_position_continuous += self.gamma * np.random.random() * np.random.choice([-1, 1], n_features)
       
        sigmoid = 1 / (1 + np.exp(-12 * (new_position_continuous - 0.5)))
        new_position = np.where(np.random.random(n_features) < sigmoid, 1, 0)
       
        if np.sum(new_position) < 3:
            random_indices = np.random.choice(n_features, 3, replace=False)
            new_position[random_indices] = 1
       
        return new_position
   
    def optimize(self, X, y, classifier='svm'):
        n_features = X.shape[1]
       
        self.population = self.initialize_population(n_features)
        self.fitness_values = np.array([self.fitness_function(solution, X, y, classifier) for solution in self.population])
       
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
       
        for t in range(self.max_iter):
            X_mean = np.mean(self.population, axis=0)
           
            for i in range(self.n_jellyfishes):
                self.current_index = i
               
                new_position = self.update_position(self.population[i], X_mean, self.best_solution, t)
                new_fitness = self.fitness_function(new_position, X, y, classifier)
               
                if new_fitness < self.fitness_values[i]:
                    self.population[i] = new_position
                    self.fitness_values[i] = new_fitness
                   
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = new_fitness
           
            self.fitness_history.append(self.best_fitness)
       
        selected_indices = np.where(self.best_solution == 1)[0]
        return selected_indices, self.best_fitness

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
        population = np.random.randint(0, 2, size=(self.n_jellyfishes, n_features))
        
        for i in range(10):
            feature_count = np.random.choice([5, 6, 7])
            features = np.random.choice(n_features, feature_count, replace=False)
            population[i] = np.zeros(n_features)
            population[i][features] = 1
           
        return population
   
    def fitness_function(self, solution, X, y, classifier='svm'):
        selected_features = np.where(solution == 1)[0]
       
        if len(selected_features) < 3:
            return 0.8
        
        if len(selected_features) > 7:
            return 0.7
       
        X_selected = X[:, selected_features]
       
        if classifier == 'svm':
            model = SVC(kernel='rbf', probability=True, C=10)
        elif classifier == 'dt':
            model = DecisionTreeClassifier()
        elif classifier == 'ann':
            model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)
        elif classifier == 'adaboost':
            model = AdaBoostClassifier()
        elif classifier == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = SVC(kernel='rbf', probability=True, C=10)
       
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_selected, y, cv=skf, scoring='roc_auc')
       
        error_rate = 1 - np.mean(scores)
       
        if len(selected_features) <= 6:
            feature_penalty = 0.005 * len(selected_features) / X.shape[1]
        else:
            feature_penalty = 0.02 * len(selected_features) / X.shape[1]
       
        fitness = 0.9 * error_rate + 0.1 * feature_penalty
       
        return fitness
   
    def update_position(self, current_position, X_mean, best_jellyfish, t):
        n_features = len(current_position)
       
        trend = best_jellyfish - self.beta * np.random.random() * X_mean
        c_t = abs((1 - t/self.max_iter) * (2 * np.random.random() - 1))
       
        new_position_continuous = current_position.copy().astype(float)
       
        if c_t > 0.5:
            new_position_continuous = current_position + np.random.random() * trend
        else:
            j = np.random.randint(0, self.n_jellyfishes)
            other_jellyfish = self.population[j]
           
            if self.fitness_values[self.current_index] < self.fitness_values[j]:
                new_position_continuous = current_position + np.random.random() * (other_jellyfish - current_position)
            else:
                new_position_continuous = current_position + np.random.random() * (current_position - other_jellyfish)
               
            if np.random.random() < 0.2:
                new_position_continuous += self.gamma * np.random.random() * np.random.choice([-1, 1], n_features)
       
        sigmoid = 1 / (1 + np.exp(-6 * (new_position_continuous - 0.5)))
        new_position = np.where(np.random.random(n_features) < sigmoid, 1, 0)
       
        feature_count = np.sum(new_position)
        if feature_count < 3:
            zeros_indices = np.where(new_position == 0)[0]
            add_count = 3 - feature_count
            if len(zeros_indices) >= add_count:
                random_indices = np.random.choice(zeros_indices, add_count, replace=False)
                new_position[random_indices] = 1
        elif feature_count > 7:
            ones_indices = np.where(new_position == 1)[0]
            remove_count = feature_count - 7
            if len(ones_indices) >= remove_count:
                random_indices = np.random.choice(ones_indices, remove_count, replace=False)
                new_position[random_indices] = 0
       
        return new_position
   
    def optimize(self, X, y, classifier='svm'):
        n_features = X.shape[1]
       
        self.population = self.initialize_population(n_features)
        self.fitness_values = np.array([self.fitness_function(solution, X, y, classifier) for solution in self.population])
       
        best_idx = np.argmin(self.fitness_values)
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = self.fitness_values[best_idx]
        self.fitness_history.append(self.best_fitness)
       
        for t in range(self.max_iter):
            X_mean = np.mean(self.population, axis=0)
           
            for i in range(self.n_jellyfishes):
                self.current_index = i
               
                new_position = self.update_position(self.population[i], X_mean, self.best_solution, t)
                new_fitness = self.fitness_function(new_position, X, y, classifier)
               
                if new_fitness < self.fitness_values[i]:
                    self.population[i] = new_position
                    self.fitness_values[i] = new_fitness
                   
                    if new_fitness < self.best_fitness:
                        self.best_solution = new_position.copy()
                        self.best_fitness = new_fitness
           
            self.fitness_history.append(self.best_fitness)
       
        selected_indices = np.where(self.best_solution == 1)[0]
        return selected_indices, self.best_fitness

# Evaluation function
def evaluate_classifier_with_roc(X, y, classifier, feature_indices=None, n_folds=5, label=""):
    if feature_indices is not None:
        X_selected = X[:, feature_indices]
    else:
        X_selected = X
   
    accuracies = []
    sensitivities = []
    specificities = []
    f1_scores = []
    aucs = []
    
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
   
    for train_index, test_index in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_index], X_selected[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
       
        try:
            y_prob = classifier.predict_proba(X_test)[:, 1]
            auc_value = roc_auc_score(y_test, y_prob)
            
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            
        except:
            auc_value = 0
       
        accuracy = accuracy_score(y_test, y_pred)
        sensitivity = recall_score(y_test, y_pred)
        specificity = recall_score(y_test, y_pred, pos_label=0)
        f1 = f1_score(y_test, y_pred)
       
        accuracies.append(accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        f1_scores.append(f1)
        aucs.append(auc_value)
   
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
if st.sidebar.button("🚀 Run Optimization", type="primary"):
    # Load data
    with st.spinner("Loading dataset..."):
        data = load_dataset(dataset_option)
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Shape", f"{data.shape[0]} × {data.shape[1]}")
    with col2:
        st.metric("Features", data.shape[1] - 1)
    with col3:
        st.metric("Target Classes", len(data['target'].unique()))
    
    # Data preprocessing
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Apply SMOTE
    with st.spinner("Applying SMOTE for class balancing..."):
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Scale features
    with st.spinner("Scaling features..."):
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_resampled)
    
    # Classifier mapping
    classifier_map = {
        "SVM": SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    classifier = classifier_map[classifier_option]
    
    # Progress bar for optimization
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run optimization based on scenario
    if scenario_option == "Original JOA":
        with st.spinner("Running Original JOA..."):
            joa = JellyfishOptimization(n_jellyfishes=n_jellyfishes, max_iter=max_iterations, beta=beta, gamma=gamma)
            selected_features, best_fitness = joa.optimize(X_scaled, y_resampled, classifier=classifier_option.lower().replace(" ", ""))
            fitness_history = joa.fitness_history
    elif scenario_option == "JOA with Hard Cap (7 features)":
        with st.spinner("Running JOA with Hard Cap..."):
            joa = JellyfishOptimizationScenario2(n_jellyfishes=n_jellyfishes, max_iter=max_iterations, beta=beta, gamma=gamma)
            selected_features, best_fitness = joa.optimize(X_scaled, y_resampled, classifier=classifier_option.lower().replace(" ", ""))
            fitness_history = joa.fitness_history
    else:  # All Features
        selected_features = None
        best_fitness = 0
        fitness_history = []
    
    # Get feature names
    feature_names = X.columns.tolist()
    if selected_features is not None:
        selected_feature_names = [feature_names[i] for i in selected_features]
    else:
        selected_feature_names = feature_names
    
    # Evaluate model
    with st.spinner("Evaluating model performance..."):
        metrics = evaluate_classifier_with_roc(X_scaled, y_resampled, classifier, selected_features, label=scenario_option)
    
    # Display results
    st.success("✅ Optimization completed!")
    
    # Results section
    st.header("📊 Results")
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.2f}%")
    with col2:
        st.metric("AUC", f"{metrics['AUC']:.2f}%")
    with col3:
        st.metric("Sensitivity", f"{metrics['Sensitivity']:.2f}%")
    with col4:
        st.metric("Specificity", f"{metrics['Specificity']:.2f}%")
    
    # Feature selection results
    if selected_features is not None:
        st.subheader("🎯 Selected Features")
        st.write(f"**Number of selected features:** {len(selected_features)} out of {len(feature_names)}")
        st.write(f"**Selected features:** {', '.join(selected_feature_names)}")
        st.write(f"**Best fitness:** {best_fitness:.4f}")
    
    # Visualizations
    st.header("📈 Visualizations")
    
    # Create tabs for different plots
    tab1, tab2, tab3, tab4 = st.tabs(["ROC Curve", "Feature Selection", "Convergence", "Dataset Analysis"])
    
    with tab1:
        # ROC Curve
        if metrics['tprs']:
            fig = go.Figure()
            
            mean_tpr = np.mean(metrics['tprs'], axis=0)
            mean_auc = metrics['AUC']/100
            
            fig.add_trace(go.Scatter(
                x=metrics['mean_fpr'],
                y=mean_tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {mean_auc:.3f})',
                line=dict(color='blue', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random Chance',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title=f'ROC Curve - {classifier_option} with {scenario_option}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                width=800,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Feature selection visualization
        if selected_features is not None:
            # Create feature importance plot
            feature_importance = np.zeros(len(feature_names))
            feature_importance[selected_features] = 1
            
            fig = go.Figure(data=[
                go.Bar(
                    x=feature_names,
                    y=feature_importance,
                    marker_color=['green' if i in selected_features else 'lightgray' for i in range(len(feature_names))],
                    text=[f'Selected' if i in selected_features else 'Not Selected' for i in range(len(feature_names))],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f'Feature Selection Results - {scenario_option}',
                xaxis_title='Features',
                yaxis_title='Selected (1) / Not Selected (0)',
                width=800,
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Convergence plot
        if fitness_history:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(range(len(fitness_history))),
                y=fitness_history,
                mode='lines',
                name='Fitness Value',
                line=dict(color='purple', width=2)
            ))
            
            fig.update_layout(
                title=f'JOA Convergence - {scenario_option}',
                xaxis_title='Iteration',
                yaxis_title='Fitness Value (Lower is Better)',
                width=800,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Dataset analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution
            fig = px.pie(
                values=data['target'].value_counts().values,
                names=['Class 0', 'Class 1'],
                title='Target Class Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature correlation heatmap (if not too many features)
            if len(feature_names) <= 15:
                corr_matrix = data.corr()
                fig = px.imshow(
                    corr_matrix,
                    title='Feature Correlation Matrix',
                    color_continuous_scale='RdBu'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature correlation heatmap skipped (too many features)")
    
    # Download results
    st.header("💾 Download Results")
    
    # Create results summary
    results_summary = {
        'Dataset': dataset_option,
        'Classifier': classifier_option,
        'Scenario': scenario_option,
        'Accuracy (%)': metrics['Accuracy'],
        'AUC (%)': metrics['AUC'],
        'Sensitivity (%)': metrics['Sensitivity'],
        'Specificity (%)': metrics['Specificity'],
        'F1 Score': metrics['F1_Score'],
        'Selected Features': ', '.join(selected_feature_names) if selected_features is not None else 'All Features',
        'Number of Features': len(selected_features) if selected_features is not None else len(feature_names),
        'Best Fitness': best_fitness if selected_features is not None else 'N/A'
    }
    
    results_df = pd.DataFrame([results_summary])
    
    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f'joa_results_{dataset_option.replace(" ", "_")}_{classifier_option.replace(" ", "_")}.csv',
        mime='text/csv'
    )

# Information section
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ About")
st.sidebar.markdown("""
This application demonstrates the Jellyfish Optimization Algorithm (JOA) for feature selection in machine learning classification tasks.

**Key Features:**
- Multiple datasets support
- Interactive parameter tuning
- Real-time optimization visualization
- Comprehensive performance metrics
- Feature selection analysis

**Algorithms:**
- Original JOA
- JOA with hard feature cap
- Multiple classifier support
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🐙 Jellyfish Optimization Algorithm - Feature Selection Tool</p>
    <p>Built with Streamlit | Machine Learning Project</p>
</div>
""", unsafe_allow_html=True) 