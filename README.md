# 🐙 Jellyfish Optimization Algorithm - Feature Selection Tool

A comprehensive web application built with Streamlit that demonstrates the Jellyfish Optimization Algorithm (JOA) for feature selection in machine learning classification tasks.

## 🚀 Features

### **Interactive Web Interface**
- **Multiple Datasets**: Heart Disease, Breast Cancer, Diabetes, Credit Card Fraud
- **Parameter Tuning**: Adjust JOA parameters in real-time
- **Multiple Classifiers**: SVM, Decision Tree, Neural Network, AdaBoost, Random Forest
- **Feature Selection Scenarios**: Original JOA, JOA with Hard Cap, All Features

### **Advanced Visualizations**
- **ROC Curves**: Interactive plots with AUC metrics
- **Feature Selection Analysis**: Visual representation of selected features
- **Convergence Plots**: Real-time optimization progress
- **Dataset Analysis**: Class distribution and correlation matrices

### **Comprehensive Metrics**
- Accuracy, Sensitivity, Specificity
- AUC-ROC scores
- F1-Score
- Feature reduction statistics

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd jellyfish-optimization-app
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run streamlit_app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📊 How to Use

### **1. Configuration (Sidebar)**
- **Select Dataset**: Choose from 4 different datasets
- **JOA Parameters**: 
  - Number of Jellyfishes (10-100)
  - Maximum Iterations (20-200)
  - Beta Parameter (1.0-5.0)
  - Gamma Parameter (0.01-0.5)
- **Classifier**: Choose your preferred ML algorithm
- **Scenario**: Select feature selection approach

### **2. Run Optimization**
- Click the **"🚀 Run Optimization"** button
- Watch real-time progress indicators
- View comprehensive results and visualizations

### **3. Analyze Results**
- **Metrics Dashboard**: Key performance indicators
- **Feature Selection**: See which features were selected
- **Visualizations**: Interactive plots in organized tabs
- **Download Results**: Export results as CSV

## 🧠 Algorithm Details

### **Jellyfish Optimization Algorithm (JOA)**
The JOA is a metaheuristic optimization algorithm inspired by the behavior of jellyfish in the ocean:

1. **Ocean Current**: Jellyfish follow ocean currents (trend direction)
2. **Group Movement**: Jellyfish move in groups based on fitness
3. **Passive Motion**: Random movement for exploration
4. **Time Control**: Balances exploration vs exploitation

### **Feature Selection Implementation**
- **Binary Encoding**: Features represented as 0/1 (selected/not selected)
- **Fitness Function**: Combines classification accuracy with feature count penalty
- **Sigmoid Function**: Converts continuous positions to binary feature selection
- **Cross-Validation**: 5-fold stratified CV for robust evaluation

## 📈 Performance Metrics

### **Classification Metrics**
- **Accuracy**: Overall classification accuracy
- **Sensitivity**: True Positive Rate (Recall)
- **Specificity**: True Negative Rate
- **AUC**: Area Under ROC Curve
- **F1-Score**: Harmonic mean of precision and recall

### **Feature Selection Metrics**
- **Feature Reduction**: Percentage of features removed
- **Fitness Value**: Optimization objective function value
- **Convergence**: Iteration-wise fitness improvement

## 🎯 Use Cases

### **Medical Diagnosis**
- Heart disease prediction
- Breast cancer detection
- Diabetes classification

### **Financial Applications**
- Credit card fraud detection
- Risk assessment
- Customer churn prediction

### **Research Applications**
- Feature importance analysis
- Model interpretability
- Algorithm comparison

## 🔧 Technical Implementation

### **Data Preprocessing**
- **SMOTE**: Synthetic Minority Over-sampling Technique for class balancing
- **RobustScaler**: Robust feature scaling for outlier handling
- **Missing Value Handling**: Automatic removal of incomplete records

### **Model Evaluation**
- **Stratified K-Fold CV**: Ensures balanced class distribution in folds
- **ROC Analysis**: Comprehensive receiver operating characteristic curves
- **Statistical Significance**: Confidence intervals for performance metrics

### **Visualization Framework**
- **Plotly**: Interactive, publication-quality plots
- **Streamlit**: Modern, responsive web interface
- **Real-time Updates**: Dynamic visualization during optimization

## 📁 Project Structure

```
jellyfish-optimization-app/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md               # Project documentation
└── ml code.py             # Original implementation
```

## 🚀 Deployment Options

### **Local Development**
```bash
streamlit run streamlit_app.py
```

### **Cloud Deployment**
- **Streamlit Cloud**: Free hosting for Streamlit apps
- **Heroku**: Container-based deployment
- **AWS/GCP**: Scalable cloud deployment

### **Docker Deployment**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Jellyfish Optimization Algorithm**: Original research paper implementation
- **Streamlit**: For the amazing web framework
- **Scikit-learn**: For machine learning algorithms
- **Plotly**: For interactive visualizations

## 📞 Contact

For questions, suggestions, or collaborations:
- **Email**: [your-email@example.com]
- **GitHub**: [your-github-profile]
- **LinkedIn**: [your-linkedin-profile]

---

**🐙 Made with ❤️ using Streamlit and Python**
