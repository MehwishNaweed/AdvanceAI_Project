# Integrated Healthcare AI -- Google Colab

# --- 1) Imports ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample
import joblib
import os
from google.colab import files
%matplotlib inline
print("Libraries loaded")

# --- 2) Load dataset (CSV upload fallback to sklearn) ---

csv_path = 'synthetic_dataset_B.csv'

if os.path.exists(csv_path):
  print("Loading", csv_path)
  df = pd.read_csv(csv_path)
else:
  print("CSV not found. Please upload it or fallback to sklearn breast cancer dataset.")
  uploaded = files.upload()
  if uploaded:
    csv_file = list(uploaded.keys())[0]
    df = pd.read_csv(csv_file)
  else:
    print("No file uploaded. Using sklearn breast cancer dataset.")
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    df = pd.DataFrame(X, columns=load_breast_cancer().feature_names)
    df['target'] = y

df.head()

# --- 3) Prepare train/test split ---

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y)
print('Train/Test shapes:', X_train.shape, X_test.shape)

# --- 4) CENTRALIZED MODEL ---

central = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=200, random_state=0)
central.fit(X_train, y_train)
acc_central = accuracy_score(y_test, central.predict(X_test))
print('Centralized accuracy:', acc_central)

# --- 5) FEDERATED SIMULATION (2 nodes) ---

n_nodes = 2
fed_models = []
fed_accs = []
for i in range(n_nodes):
  Xp, yp = resample(X_train, y_train, n_samples=len(X_train)//n_nodes, random_state=i, stratify=y_train)
  m = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=150, random_state=1+i)
  m.fit(Xp, yp)
  fed_models.append(m)
  fed_accs.append(accuracy_score(y_test, m.predict(X_test)))

acc_fed = float(np.mean(fed_accs))
print('Federated node accuracies:', fed_accs)
print('Federated (mean) accuracy:', acc_fed)

# --- 6) DEFENSE MODEL (noise-augmented) ---

noise = np.random.normal(0, 0.05, X_train.shape)
Xd = X_train + noise
def_model = MLPClassifier(hidden_layer_sizes=(16,8), max_iter=200, random_state=10)
def_model.fit(Xd, y_train)
acc_def = accuracy_score(y_test, def_model.predict(X_test))
print('Defense model accuracy:', acc_def)

# --- 7) INTEGRATED MODEL (majority vote) ---

class IntegratedModel:
  def __init__(self, models):
    self.models = models
  def predict(self, X):
    preds = np.vstack([m.predict(X) for m in self.models])
    return (preds.mean(axis=0) > 0.5).astype(int)

integrated = IntegratedModel([central] + fed_models + [def_model])
y_pred_int = integrated.predict(X_test)
acc_integrated = accuracy_score(y_test, y_pred_int)
acc_weighted = 0.5*acc_central + 0.3*acc_fed + 0.2*acc_def
print('Integrated (majority vote) accuracy:', acc_integrated)
print('Weighted integrated score:', acc_weighted)

# --- 8) Save models and results ---

os.makedirs('Models', exist_ok=True)
joblib.dump(central, 'Models/central_model.pkl')
joblib.dump(def_model, 'Models/defense_model.pkl')
for i,m in enumerate(fed_models,start=1):
  joblib.dump(m,f'Models/federated_node_{i}.pkl')
joblib.dump(integrated,'Models/integrated_model.pkl')

results = {
'Centralized': float(acc_central),
'Federated': float(acc_fed),
'Defense': float(acc_def),
'Integrated': float(acc_integrated),
'Weighted_Integrated': float(acc_weighted)
}
import json
with open('Models/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved models and results to Models/')
print('Results summary:', results)

# --- Download Models (optional) ---

for f_name in os.listdir('Models'):
  files.download(os.path.join('Models', f_name))

# --- 9) Plots ---

plt.figure(figsize=(8,5))
names = ['Centralized','Federated(mean)','Defense','Integrated']
vals = [results['Centralized'], results['Federated'], results['Defense'], results['Integrated']]
plt.bar(names, vals)
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.grid(axis='y', linestyle=':')
plt.show()

print('\nConfusion Matrix: Centralized')
ConfusionMatrixDisplay.from_estimator(central, X_test, y_test)
plt.show()

print('\nConfusion Matrix: Defense')
ConfusionMatrixDisplay.from_estimator(def_model, X_test, y_test)
plt.show()

# --- 10) Classification reports and sample predictions ---

#Centralized

print("Classification report (Centralized):")
print(classification_report(y_test, central.predict(X_test)))

#Federated nodes

for i, m in enumerate(fed_models, start=1):
  print(f"\nClassification report (Federated Node {i}):")
  print(classification_report(y_test, m.predict(X_test)))

#Federated majority vote

fed_preds = np.vstack([m.predict(X_test) for m in fed_models])
fed_vote = (fed_preds.mean(axis=0) > 0.5).astype(int)
print("\nClassification report (Federated Integrated / Majority Vote):")
print(classification_report(y_test, fed_vote))

#Defense model

print("\nClassification report (Defense model):")
print(classification_report(y_test, def_model.predict(X_test)))

#Integrated model (all models)

print("\nClassification report (Integrated model):")
print(classification_report(y_test, integrated.predict(X_test)))

label_map = {0: "Malignant", 1: "Benign"}
sample = X_test[0].reshape(1,-1)
print("\nSingle-sample predictions:")
print("Central:", label_map[central.predict(sample)[0]])
print("Defense:", label_map[def_model.predict(sample)[0]])
for i, m in enumerate(fed_models, start=1):
  print(f"Fed{i}:", label_map[m.predict(sample)[0]])
print("Integrated:", label_map[integrated.predict(sample)[0]])

# --- 11) show predictions for all test samples in dataframe ---

all_preds = {
'Central': [label_map[x] for x in central.predict(X_test)],
'Defense': [label_map[x] for x in def_model.predict(X_test)]
}
for i, m in enumerate(fed_models, start=1):
  all_preds[f'Fed{i}'] = [label_map[x] for x in m.predict(X_test)]
all_preds['Integrated'] = [label_map[x] for x in integrated.predict(X_test)]

summary_df = pd.DataFrame(all_preds)

#--- 12) Count Malignant and Benign predictions ----

for col in summary_df.columns:
  counts = summary_df[col].value_counts()
  print(f"\n{col} predictions summary:")
  print(counts)

# --- 13) Visualize predictions -----

plt.figure(figsize=(10,6))
for col in summary_df.columns:
  counts = summary_df[col].value_counts()
  plt.bar([f"{col} - {k}" for k in counts.index], counts.values, alpha=0.7)

plt.ylabel("Number of samples")
plt.title("Malignant/Benign predictions per model")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle=':')
plt.tight_layout()
plt.show()

# --- 14) End of notebook ---

print("Models and results saved in 'Models/' folder.")
