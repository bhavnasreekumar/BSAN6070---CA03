# CA-03 Decision Tree Income Classification

## Assignment Overview

This project is part of BSAN 6070 – Intro to Machine Learning and focuses on building a Decision Tree classification model to predict an individual’s income category (<=50K or >50K).

The assignment includes data quality analysis (DQA), discretization (binning), model tuning, visualization of the best tree, and prediction of a new individual using the trained model.

---

## Technologies & Packages Used

This analysis was conducted using **Python** in a Jupyter Notebook environment. The following libraries were used:

- `pandas` – Data manipulation and preprocessing  
- `numpy` – Numerical computations  
- `matplotlib` – Tree visualization  
- `sklearn` – Machine learning modeling  
  - `DecisionTreeClassifier`  
  - `train_test_split`  
  - `LabelEncoder`  
  - `accuracy_score`  

---

## Steps in the Analysis

### **1: Data Quality Analysis (DQA)**

- Checked for missing values and null entries  
- Verified data types of each column  
- Reviewed distributions of continuous variables  
- Examined categorical consistency  
- Identified potential outliers  

This step ensured the dataset was clean and suitable for modeling.

---

### **2: Discretization (Binning)**

Continuous variables such as:

- Age  
- Hours per week  
- Capital gains  
- Education years  

were transformed into grouped categories (bins).

Binning helped:
- Reduce sensitivity to very small numeric differences  
- Improve interpretability of tree splits  
- Limit excessive tree growth  
- Reduce risk of overfitting  

Each bin was labeled (e.g., `"a. Low"`, `"b. Mid - Low"`, etc.) and later encoded for modeling.

---

### **3: Encoding**

Categorical bin variables were converted into numeric format using `LabelEncoder`.

No one-hot encoding was applied in this assignment.

---

### **4: Train-Test Split**

The dataset was split into training and testing sets to evaluate performance on unseen data.

---

### **5: Model Building & Hyperparameter Tuning**

A Decision Tree Classifier was trained using selected hyperparameters:

- `criterion = "gini"`  
- `max_depth = 16`  
- `min_samples_leaf = 40`  
- `max_features = 0.6`  

Multiple combinations were evaluated and the best-performing tree was selected based on test accuracy.

**Final Test Accuracy:** ~82%  
**Training Runtime:** ~0.021 seconds  

---

### **6: Decision Tree Visualization**

The best-performing tree was visualized to:

- Identify the root split (MSR bin)  
- Analyze feature importance  
- Observe tree depth and branching  
- Examine Gini impurity values at terminal nodes  

The tree was not fully grown due to depth and leaf constraints, helping control overfitting.

---

### **7: Prediction of a New Individual (Q8)**

A new single-person record was created using the assignment-provided attributes. The same binning and encoding logic used during training was applied before prediction.

**Prediction Result:** >50K  
**Probability of predicted class:** 0.8235  

The probability was obtained using `model.predict_proba()`.

---

## Key Insights

- Marriage Status & Relationship (MSR) was the most informative feature at the root node.  
- Occupation and capital gain were strong predictors of income classification.  
- Age bins refined predictions deeper in the tree.  
- The model achieved strong but not perfect performance (~82%), indicating potential for further tuning.

---

## Authors

This project was completed by:

- **Bhavna Sreekumar**  
- **Jessica Shono Thai**
