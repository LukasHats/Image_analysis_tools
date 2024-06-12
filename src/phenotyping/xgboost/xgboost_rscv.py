# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna
import xgboost as xgb

core_path = '/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/phenotyping/xgboost/standard/'

data = pd.read_csv('/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/phenotyping/manual_phenotypes_standard.csv')

data.drop(columns=["index", "Y_centroid", "X_centroid"], inplace=True)

transformed = np.arcsinh(data.iloc[:, 0:32, ])
data.drop(columns=data.columns[0:32], inplace=True)
phenotypes = pd.concat([transformed, data], axis=1)

phenotypes['distance_to_bone'] = phenotypes['distance_to_bone'].replace(-999, np.nan)

X = phenotypes.iloc[:, :-1]
y = phenotypes.iloc[:, -1]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_mapping = pd.DataFrame({
    'Phenotype': y,
    'EncodedLabel': y_encoded
})

label_mapping.drop_duplicates(inplace=True)

label_mapping.to_csv(core_path + '/phenotype_label_mapping.csv', index=False)

classes = np.unique(y_encoded)
weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
class_weights = dict(zip(classes, weights))

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=20240610)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=20240610)

stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=20240610)

model = XGBClassifier(
    objective='multi:softmax',
    eval_metric='mlogloss',
    use_label_encoder=False,
    num_class=len(np.unique(y_encoded)),
    tree_method='gpu_hist' 
)

param_dist = {
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.5 ,0.6, 0.7, 0.8, 0.9, 1.0],
    'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    'n_estimators': [200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'lambda': [0.1, 0.2, 0.5, 1, 1.5, 2],
    'alpha': [0, 0.5, 1, 2]
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, scoring='f1_weighted', cv=stratified_kfold, verbose=2, random_state=20240610, n_jobs=-1)

sample_weights = np.array([class_weights[class_label] for class_label in y_train])

random_search.fit(X_train, y_train, sample_weight=sample_weights)

best_model = random_search.best_estimator_
print("Best parameters found: ", random_search.best_params_)
with open(core_path + '/best_model_params.txt', 'w') as file:
    file.write("Best Model Parameters:\n")
    for param, value in random_search.best_params_.items():
        file.write(f"{param}: {value}\n")

y_pred = best_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score (weighted):", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

plt.style.use('ggplot')

feature_importances = best_model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Features': features,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=True)


plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Features', data=importance_df, palette='viridis')

plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances Visualized')
plt.savefig(core_path + 'feature_importances.png', dpi=300)


cm = confusion_matrix(y_test, y_pred)

# Calculate the percentage of each value in the confusion matrix
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Create a heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentage)')
plt.tight_layout()
plt.savefig(core_path + 'confusion_matrix_percentage.png', dpi=300)
plt.show()


accuracy = accuracy_score(y_test, y_pred)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

report_df = pd.DataFrame(report).transpose()


report_df.to_csv(core_path + 'classification_report_wb.csv', index=True)

dump(best_model, core_path + 'best_xgb_model_wb.joblib')




