# Heart-Disease-Prediction-using-JOA

 
 
ABSTRACT 
 
Heart disease continues to be a leading global cause of mortality, necessitating early and 
accurate diagnostic tools. In this study, the authors present a novel heart disease prediction 
framework by integrating the Jellyfish Optimization Algorithm (JOA) with various Machine 
Learning (ML) techniques to enhance predictive performance. Using the Cleveland heart 
disease dataset, the authors first applied the JFO to reduce dimensionality by selecting the 
most relevant features, thereby mitigating the curse of dimensionality and improving 
model generalizability. The Jellyfish algorithm, inspired by the swarm movement of jellyfish 
in ocean currents, was employed due to its high convergence speed, adaptability, and 
effectiveness in finding global optima without being trapped in local minima. The reduced 
feature set was then used to train and evaluate four supervised ML classifiers: Artificial 
Neural Networks (ANN), Decision Trees (DT), AdaBoost, and Support Vector Machines 
(SVM). 
 
Objective 
This study aims to evaluate the effectiveness of the Jellyfish Optimization Algorithm (JOA) 
for feature selection in heart disease prediction. We implement two variants of JOA, the 
original algorithm and a modified version with a hard cap at 7 features, and compare their 
performance against using all features. By applying four different machine learning 
classifiers (SVM, Decision Tree, ANN, and AdaBoost) to the Cleveland heart disease dataset, 
we assess whether optimized feature subsets can improve prediction accuracy while 
reducing model complexity. The research seeks to identify the most significant predictive 
features for heart disease and determine which combination of feature selection approach 
and classification algorithm yields the best performance in terms of accuracy, sensitivity, 
specificity, and AUC. 
 
 

 
How JOA Works (The Intuition) 
JOA mimics the behaviour of jellyfish drifting in the ocean. It combines: 
1. Ocean Current Following — exploring globally by drifting with the current toward better solutions. 
2. Group Movement — locally adjusting position based on other jellyfish (can be passive or active). 
3. Time-based Switching — alternates between current-following and group movement as optimization 
progresses. 
Jellyfish Optimization Algorithm: Key Steps 
1. Initialize a population of jellyfish (potential solutions). 
2. Evaluate each jellyfish using a fitness function. 
3. Identify the best jellyfish with lowest fitness value. 
4. For each iteration: Calculate mean position of all jellyfish. 
For each jellyfish:  
 Either follow ocean current (move toward best solution) OR 
 Perform group movement (interact with random jellyfish). 
 Convert continuous position to binary using sigmoid function. 
 Update population if new position is better. 
5. Return the best solution (selected features). 
 
1. Dataset Overview 
The analysis was performed on the Cleveland heart disease dataset, which contains 303 records with 14 attributes 
including the target class. After removing 6 records with missing values, the dataset contained 297 samples. The 
class distribution showed an imbalance with 160 records for non-disease (class 0) and 137 records for disease (class 
1). 
To address this class imbalance, the SMOTE (Synthetic Minority Oversampling Technique) was applied, resulting in a 
balanced dataset with 160 samples for each class (320 total). RobustScaler was used for feature scaling to handle 
potential outliers in the data. 
2. Feature Selection Experiments 
The study compared three scenarios: 
1. Scenario 1: Original Jellyfish Optimization Algorithm (JOA). Used standard JOA implementation with 
weighted fitness function (90% accuracy, 10% feature count). Employed steep sigmoid function (parameter 
12) for decisive feature selection. Achieved 9-feature selection with best fitness score of 0.0843 
2. Scenario 2: Modified JOA with Hard Cap at 7 Features (proposed change to get a more aggressive feature 
selection). Implemented strict feature count limitation mechanism. Used softer sigmoid function (parameter 
6) for more exploratory feature selection. Applied progressive penalty system (0.005 for ≤6 features, 0.02 for 
7 features). Enforced hard rejection of solutions with >7 features. Resulted in 6-feature selection with 
slightly better fitness score of 0.0832 
5 
 
3. Scenario 3: All Features. Baseline approach using all 13 features from the dataset for comparative 
performance assessment. 
3. Feature Selection Results 
The Jellyfish Optimization Algorithm proved to be an effective tool for selecting meaningful features in both tested 
scenarios. In Scenario 1 (Original JOA), the algorithm selected 9 features: sex, cp (chest pain type), chol (cholesterol), 
restecg (resting ECG), thalach (maximum heart rate), exang (exercise-induced angina), slope, ca (number of major 
vessels), and thal. It reached a best fitness score of 0.0843 after 50 iterations. In contrast, Scenario 2 (Modified JOA 
with a Hard Cap) produced a more compact feature set of just 6: age, sex, thalach, exang, ca, and thal, with a slightly 
better fitness score of 0.0832. 
Notably, five features — sex, thalach, exang, ca, and thal — were common to both versions, highlighting their strong 
and consistent predictive value for heart disease. Convergence plots showed that both JOA variants were able to 
reach near-optimal solutions efficiently, with Scenario 1 stabilizing by iteration 10 and Scenario 2 by iteration 20 
(Figure 3). The quicker convergence in Scenario 1 is likely due to the broader flexibility in selecting features, as it 
wasn’t restricted by a feature cap. 
When comparing feature importance, attributes tied to heart performance (thalach), blood vessel health (ca), and 
stress-induced symptoms (exang) stood out as particularly significant. These results align well with medical 
knowledge about key risk indicators, reinforcing the clinical relevance of the features selected by the algorithm. 

 
4. Classification Performance Analysis 
The classification performance across all scenarios and models is illustrated in the figures above. Support Vector 
Machine (SVM) performed best in Scenario 2, achieving 82.19% accuracy and 90.86% AUC, while AdaBoost delivered 
the strongest results in Scenario 1, with 85.31% accuracy and 91.39% AUC. When using all available features 
(Scenario 3), AdaBoost achieved the highest overall AUC of 91.80%. 
The ROC curve analysis showed that even with fewer features, the models retained strong classification ability. In 
fact, SVM with Scenario 2’s compact feature set outperformed its full-feature version, highlighting how targeted 
feature selection can enhance model efficiency without sacrificing accuracy. 
Looking at sensitivity and specificity, some interesting patterns emerged. Scenario 1 (9 features) provided better 
sensitivity with AdaBoost (80.00%), while Scenario 2 (6 features) led to higher specificity with SVM (98.37%). This 
shows how different feature subsets can optimize different parts of a model’s performance, making it possible to 
tailor model selection depending on whether false positives or false negatives are more important to reduce — a 
crucial consideration in clinical decision-making. 
5. Discussion of Results 
The performance differences across scenarios and classifiers offer meaningful insights into how feature selection 
methods interact with different types of machine learning models. AdaBoost achieved its best results with Scenario 
1, likely because it benefits from having access to a wider range of features that work well together, strengthening 
its ensemble learning process. On the other hand, SVM performed better with Scenario 2’s smaller feature set, 
which aligns with its sensitivity to irrelevant or noisy features that can interfere with finding the optimal decision 
boundary. 
When we compared our results to those in the original paper, we noticed a clear gap in accuracy. The paper 
reported much higher results (97–98%) compared to ours (73–85%). This difference can mostly be explained by 
dataset size. The paper used 1172 samples, while we worked with just 320 after applying SMOTE. A smaller dataset 
often means more variability, a higher chance of overfitting, and less stable feature selection. In addition, differences 
in how the Jellyfish Optimization Algorithm was implemented, such as the sigmoid function used, fitness penalty 
values, and how the initial population was created, also contributed to differences in the selected features. 
Interestingly, the baseline model using all features (Scenario 3) sometimes outperformed the optimized feature sets, 
especially with AdaBoost. While that might seem surprising, it makes sense in context. First, with a small dataset, the 
benefits of reducing features are less noticeable. Second, AdaBoost is naturally good at filtering out unhelpful 
features as it learns. And third, when selecting features from a small dataset, it’s easy to overfit during the selection 
phase. 
Overall, the results emphasize the need to carefully align feature selection methods with the classifier being used 
and the nature of the dataset. While reducing features usually helps simplify the model and improve performance, 
the best approach ultimately depends on the algorithm, how much data you have, and which performance metrics 
matter most for your use case. 
Innovations Relative to Original Paper 
Our study enhances the original research in several meaningful ways: 
1. Smarter Feature Limits: We created a version of JOA that caps the number of features at 7, 
making the resulting models more practical for real-world clinical use where simpler models are 
easier to understand and implement. 

 
2. Gentler Decision-Making: By modifying how the algorithm selects features, we allowed it to be less 
rigid in its choices, helping it find better combinations of features even with limited data. 
3. Side-by-Side Comparison: Unlike the original paper that tested just one approach, we 
systematically compared different versions of JOA across multiple classifiers, providing clearer 
guidance on which combinations work best. 
4. Realistic Performance Assessment: Through rigorous cross-validation, our approach achieved 
more realistic accuracy metrics (73-85%) compared to the potentially inflated results in the original 
paper (97-98%), providing a more reliable foundation for clinical application. 
5. Comparison with other Common Feature Selection Methods: We compared our jellyfish
inspired approach against six widely used feature selection techniques that researchers rely on.  
These improvements make our approach more suitable for developing heart disease prediction tools that 
doctors could actually use and trust in everyday practice. 
 
CONCLUSION 
In this study, we explored how the Jellyfish Optimization Algorithm (JOA) can improve heart disease 
prediction. We tested two versions: the original, which picked 9 features, and a modified version that 
limited selection to 7. Surprisingly, the simpler version performed slightly better, showing that less can 
sometimes be more. 
Some features — like sex, maximum heart rate, angina during exercise, number of vessels, and thallium 
test results — consistently stood out as key indicators. 
We found that different models responded differently to the selected features. AdaBoost gave the best 
results with 9 features, while SVM worked best with just 6. By using careful cross-validation, we avoided 
overly optimistic results and got a more realistic picture of how these models would perform in practice. 
In short, our modified JOA helps create faster, simpler, and still accurate models — which is exactly what 
we need in real-world healthcare tools. 
GROUP DETAILS 
NAME ID TASK 
Shardul Bhope 2022A4PS1484H Implementation of baseline model 
(without Jellyfish Algo) 
Niranj Gaurav 2022A4PS0849H Implementation of SVM and AdaBoost. 
Silla Nakul 2022A3PS1322H Implementation of the Jellyfish 
Optimization Algorithm 
Souvik Sattwik Agasti 2022A3PS1659H Implementation of ANN and Decision 
Trees. 
 
 
