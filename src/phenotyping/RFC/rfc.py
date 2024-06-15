import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import dump

base_path = '/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/github/myeloma_standal/src/Phenotyping/ensemble'

data = pd.read_csv('/Users/lukashat/Documents/PhD_Schapiro/Projects/Myeloma_Standal/github/myeloma_standal/phenotyping/manual_phenotypes_standard.csv')
data.drop(columns=["index", "Y_centroid", "X_centroid"], inplace=True)
transformed = np.arcsinh(data.iloc[:, 0:32])
data.drop(columns=data.columns[0:32], inplace=True)
data = pd.concat([transformed, data], axis=1)

y = data.iloc[:, -1]  # assuming the last column is the target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_mapping = pd.DataFrame({'Phenotype': y, 'EncodedLabel': y_encoded}).drop_duplicates()
label_mapping.to_csv(os.path.join(base_path, 'mapping_path.csv'), index=False)

X = data.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=20240613)

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': 'balanced'
    }

    classifier = RandomForestClassifier(**params, n_jobs=-1)
    f1_weighted_scorer = make_scorer(f1_score, average='weighted')
    score = cross_val_score(classifier, X_train, y_train, cv=5, scoring=f1_weighted_scorer)
    return score.mean()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)

with open(os.path.join(base_path,"best_trial.txt"), "w") as file:
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write("  Value: {}\n".format(trial.value))
    file.write("  Params:\n")
    for key, value in trial.params.items():
        file.write("    {}: {}\n".format(key, value))

best_params = study.best_params
best_rf = RandomForestClassifier(**best_params, n_jobs=-1)
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

plt.style.use('ggplot')
feature_importances = best_rf.feature_importances_
features = X_train.columns

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
plt.savefig(os.path.join(base_path, 'feature_importances_opt.png'), dpi=300)
plt.close()


cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentage)')
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'confusion_matrix_percentage_opt.png'), dpi=300)
plt.close()

# Classification Report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df = report_df[report_df.index != 'accuracy']

accuracies = {}
for label_str in label_encoder.classes_:
    label_int = label_encoder.transform([label_str])[0]
    mask = y_test == label_int
    accuracies[label_str] = accuracy_score(y_test[mask], y_pred[mask]) if mask.sum() > 0 else np.nan

report_df['accuracy'] = pd.Series(accuracies)
macro_avg_accuracy = report_df.loc[label_encoder.classes_, 'accuracy'].mean()
weights = report_df.loc[label_encoder.classes_, 'support'] / report_df.loc[label_encoder.classes_, 'support'].sum()
weighted_avg_accuracy = np.sum(report_df.loc[label_encoder.classes_, 'accuracy'] * weights)
report_df.loc['macro avg', 'accuracy'] = macro_avg_accuracy
report_df.loc['weighted avg', 'accuracy'] = weighted_avg_accuracy
report_df = report_df[['precision', 'recall', 'f1-score', 'accuracy', 'support']]
report_df.to_csv(os.path.join(base_path, 'classification_report_wb_opt.csv'), index=True)

# Save the Model
dump(best_rf, os.path.join(base_path, 'best_rf_model.joblib'))