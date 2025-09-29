# Iris Models Demo (scikit-learn)

This repository contains demo machine learning models trained on the classic **Iris dataset**.  
They are lightweight examples meant for educational use and Kaggle badge experiments.

## Models Included
1. **Logistic Regression Pipeline**  
   - Preprocessing: `StandardScaler`  
   - Estimator: `LogisticRegression` (max_iter=200, random_state=42)  
   - Accuracy: ~95% on test split  
   - File: `iris_logreg.pkl`

2. **RandomForestClassifier**  
   - Estimator: `RandomForestClassifier` (100 trees, random_state=42)  
   - Accuracy: ~97% on test split  
   - File: `iris_rf.pkl`

---

## Usage

### Logistic Regression
```python
import joblib, numpy as np

model = joblib.load("iris_logreg.pkl")
X_new = np.array([[5.1, 3.5, 1.4, 0.2]])
pred = model.predict(X_new)
print("Predicted class:", int(pred[0]))  # 0=setosa, 1=versicolor, 2=virginica
```

### Random Forest
```python
import joblib, numpy as np

model = joblib.load("iris_rf.pkl")
X_new = np.array([[6.0, 3.0, 5.0, 1.8]])
pred = model.predict(X_new)

species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
print("Predicted species:", species_map[int(pred[0])])
```

---

## Environment
- Python 3.11  
- scikit-learn 1.2.2  
- joblib 1.5.1  
- Numpy 1.26.x  

---

## Provenance
- Dataset: Iris dataset (Fisher, 1936), available via `sklearn.datasets.load_iris`.  
- Training: Performed in Kaggle Notebooks (CPU, negligible runtime).  

## Citation
Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems*.  
Annals of Eugenics, 7(2), 179–188.
