# Decision Trees and Random Forests

The objective of this task is to understand and implement **Decision Tree** and **Random Forest** models for classification. The project also focuses on tree visualization, overfitting analysis, feature importance, model comparison, and cross-validation.

## Objectives

- Train a Decision Tree Classifier
- Visualize the Decision Tree
- Analyze overfitting using tree depth
- Control tree complexity using `max_depth`
- Train a Random Forest Classifier
- Compare model accuracy
- Interpret feature importance
- Evaluate models using cross-validation

## Dataset

**Dataset:** Heart Disease Dataset

The dataset contains patient-related medical attributes used to predict the presence of heart disease.

The target variable is:

- `0` — No heart disease
- `1` — Heart disease

### Dataset Location

```text
dataset/heart.csv
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Graphviz
- Jupyter Notebook

## Project Structure

```text
Day-5-task-Elevate-Lab---Nishita-kanojiya/
│
├── dataset/
│   └── heart.csv
│
├── notebooks/
│   └── decision_tree_random_forest.ipynb
│
├── src/
│   └── tree_models.py
│
├── outputs/
│   ├── target_distribution.png
│   ├── decision_tree.png
│   ├── decision_tree_confusion_matrix.png
│   ├── depth_vs_accuracy.png
│   ├── random_forest_confusion_matrix.png
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── cross_validation.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Project Workflow

```text
Load Dataset
      ↓
Data Inspection
      ↓
Data Preprocessing
      ↓
Train-Test Split
      ↓
Decision Tree Classifier
      ↓
Tree Visualization
      ↓
Overfitting Analysis
      ↓
Random Forest Classifier
      ↓
Accuracy Comparison
      ↓
Feature Importance
      ↓
Cross-Validation
      ↓
Final Evaluation
```

## Decision Tree

A Decision Tree Classifier is trained on the dataset and evaluated using the test set.

The tree is also visualized to understand how the model makes classification decisions.

## Overfitting Analysis

Different values of `max_depth` are tested to study the relationship between:

- Training accuracy
- Testing accuracy
- Tree complexity

The results are visualized using a **Depth vs Accuracy** graph.

This helps identify when increasing tree depth starts causing overfitting.

## Random Forest

A Random Forest Classifier is trained using multiple decision trees.

The Random Forest model is then evaluated and compared with the Decision Tree based on classification accuracy.

## Feature Importance

Random Forest feature importance is calculated to identify which features contribute most to the model's predictions.

The top features are visualized using a bar chart.

## Cross-Validation

Both models are evaluated using **5-fold Stratified Cross-Validation**.

This provides a more reliable estimate of model performance across different subsets of the dataset.

## Model Evaluation

The project evaluates the models using:

- Accuracy
- Classification Report
- Confusion Matrix
- Cross-Validation Accuracy

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Day-5-task-Elevate-Lab---Nishita-kanojiya
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows PowerShell:**

```bash
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/decision_tree_random_forest.ipynb
```

### 6. Run all cells

Make sure the dataset is available at:

```text
dataset/heart.csv
```

The generated graphs will be saved inside:

```text
outputs/
```

## Key Learnings

Through this task, I learned:

- How Decision Trees perform classification
- How tree depth affects model complexity
- How overfitting can occur in Decision Trees
- How Random Forest improves robustness by combining multiple trees
- How to interpret feature importance
- How to compare different machine learning models
- How cross-validation can be used for model evaluation
