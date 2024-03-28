from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score, matthews_corrcoef, \
    f1_score
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial

# importing the dataset
features_df = pd.read_csv('/home/mi/tsteuerwald/ctd_hypertension_20240223_scores.tsv', sep='\t')
ground_truth_df = pd.read_csv('merged_hyperhyper.tsv', sep='\t')
print("the data has been imported")

# merging the dataset
merged_df = pd.merge(features_df, ground_truth_df, on=['drugA', 'drugB'])
merged_df.set_index(['drugA', 'drugB'], inplace=True)

X = merged_df.drop(['adv/app', 'drugcomb', 'sA', 'sB', 'opA', 'opB'], axis=1)
Y = merged_df['adv/app']

print("the data has been merged")

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("the data has been split")

# StandardScaler initialisieren
scaler = StandardScaler()
# Feature Scaling auf Trainingsdaten anwenden
X_train_scaled = scaler.fit_transform(X_train)
# Feature Scaling auf Testdaten anwenden
X_test_scaled = scaler.transform(X_test)

print("scaling complete")

# oversampling
adasyn = ADASYN(sampling_strategy='auto', random_state=42)
X_resampled, Y_resampled = adasyn.fit_resample(X_train_scaled, Y_train)
print("oversampling complete")

# Define the parameter space for Bayesian Optimization
param_space = [Real(0.1, 10.0, name='C'),
               Real(0.1, 10.0, name='gamma'),
               Integer(2, 4, name='degree')]


# Define the objective function for Bayesian Optimization
def objective_function(params, X, Y):
    # Instantiate and train the SVM model with given hyperparameters
    svm_model = SVC(C=params[0], gamma=params[1], degree=params[2], kernel='rbf')
    svm_model.fit(X, Y)

    # Evaluate the model on validation data
    accuracy = svm_model.score(X_test_scaled, Y_test)
    return -accuracy  # Minimize negative accuracy


# Perform Bayesian Optimization
result = gp_minimize(partial(objective_function, X=X_resampled, Y=Y_resampled), param_space, n_calls=20,
                     random_state=42)

# Get the best hyperparameters
best_params = result.x

print("Best Hyperparameters:", best_params)

# Train the final model with the best hyperparameters
final_model = SVC(C=best_params[0], gamma=best_params[1], degree=best_params[2], kernel='rbf')
final_model.fit(X_resampled, Y_resampled)

# Evaluating the final model
accuracy = final_model.score(X_test_scaled, Y_test)
print("Accuracy:", accuracy)

predictions = final_model.predict(X_test_scaled)
report = classification_report(Y_test, predictions)
print(report)

# Assuming 'final_model' is your trained SVM classifier
y_pred_probs = final_model.decision_function(X_test_scaled)

# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(Y_test, y_pred_probs)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(Y_test, y_pred_probs)

# Calculate ROC AUC score
roc_auc = roc_auc_score(Y_test, y_pred_probs)

# Calculate Matthews correlation coefficient
mcc = matthews_corrcoef(Y_test, predictions)

# Calculate weighted F1 score
weighted_f1 = f1_score(Y_test, predictions, average='weighted')

# Print ROC AUC score
print(f'ROC AUC: {roc_auc}')

# Print Matthews correlation coefficient
print(f'Matthews Correlation Coefficient: {mcc}')

# Print weighted F1 score
print(f'Weighted F1 Score: {weighted_f1}')

'''
# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid(True)
plt.show()
'''

'''
 -> kernel 'linear'     
         precision    recall  f1-score   support

           0       0.00      0.00      0.00        11
           1       0.99      1.00      1.00      1376

    accuracy                           0.99      1387
   macro avg       0.50      0.50      0.50      1387
weighted avg       0.98      0.99      0.99      1387

ROC AUC: 0.7287922832980973
Matthews Correlation Coefficient: 0.0
Weighted F1 Score: 0.9881196081393822

-> kernel 'poly'
              precision    recall  f1-score   support

           0       0.00      0.00      0.00        11
           1       0.99      1.00      1.00      1376

    accuracy                           0.99      1387
   macro avg       0.50      0.50      0.50      1387
weighted avg       0.98      0.99      0.99      1387

ROC AUC: 0.8382663847780126
Matthews Correlation Coefficient: 0.0
Weighted F1 Score: 0.9881196081393822

-> kernel 'rbf'
      precision    recall  f1-score   support

           0       0.00      0.00      0.00        11
           1       0.99      1.00      1.00      1376

    accuracy                           0.99      1387
   macro avg       0.50      0.50      0.50      1387
weighted avg       0.98      0.99      0.99      1387

ROC AUC: 0.8592098308668076
Matthews Correlation Coefficient: 0.0
Weighted F1 Score: 0.9881196081393822

-> kernel 'rbf', ADASYN
Accuracy: 0.886085075702956
              precision    recall  f1-score   support

           0       0.05      0.82      0.10        11
           1       1.00      0.89      0.94      1376

    accuracy                           0.89      1387
   macro avg       0.53      0.85      0.52      1387
weighted avg       0.99      0.89      0.93      1387

ROC AUC: 0.898718287526427
Matthews Correlation Coefficient: 0.19310782503160498
Weighted F1 Score: 0.9325466236853482

-> kernel 'rbf', ADASYN, grid search
Accuracy: 0.9718817591925017
              precision    recall  f1-score   support

           0       0.11      0.36      0.17        11
           1       0.99      0.98      0.99      1376

    accuracy                           0.97      1387
   macro avg       0.55      0.67      0.58      1387
weighted avg       0.99      0.97      0.98      1387

ROC AUC: 0.8531316067653277
Matthews Correlation Coefficient: 0.18988531087151542
Weighted F1 Score: 0.9792311265672272

-> kernel 'rbf', ADASYN, grid search, scaling
Accuracy: 0.9718817591925017
              precision    recall  f1-score   support

           0       0.11      0.36      0.17        11
           1       0.99      0.98      0.99      1376

    accuracy                           0.97      1387
   macro avg       0.55      0.67      0.58      1387
weighted avg       0.99      0.97      0.98      1387

ROC AUC: 0.8531316067653277
Matthews Correlation Coefficient: 0.18988531087151542
Weighted F1 Score: 0.9792311265672272

-> kernel 'rbf', ADASYN, bayes optimization, scaling
Best Hyperparameters: [7.985775569916307, 1.9160044196750219, 4]
Accuracy: 0.9935111751982696
              precision    recall  f1-score   support

           0       1.00      0.18      0.31        11
           1       0.99      1.00      1.00      1376

    accuracy                           0.99      1387
   macro avg       1.00      0.59      0.65      1387
weighted avg       0.99      0.99      0.99      1387

ROC AUC: 0.9046643763213531
Matthews Correlation Coefficient: 0.4250137548692131
Weighted F1 Score: 0.9912756193099853
'''
