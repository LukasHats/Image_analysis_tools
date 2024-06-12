

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from optuna.integration import XGBoostPruningCallback

core_path = '/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/phenotyping/xgboost/standard/optuna_cv'

data = pd.read_csv('/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/phenotyping/manual_phenotypes_standard.csv')

data.drop(columns=["index", "Y_centroid", "X_centroid"], inplace=True)

transformed = np.arcsinh(data.iloc[:, 0:32])
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


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.10, random_state=20240610)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=20240610)

weights = [class_weights[i] for i in y_train]
dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
weights_val = [class_weights[i] for i in y_val]
dval = xgb.DMatrix(X_val, label=y_val, weight=weights_val)
evals = [(dtrain, 'train'), (dval, 'eval')]

def objective(trial):
    param = {
        'objective': 'multi:softmax',
        'tree_method': 'gpu_hist',
        'lambda': trial.suggest_loguniform('lambda', 1e-4, 1.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-4, 1.0),
        'gamma': trial.suggest_loguniform('gamma', 1e-4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.5),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'num_class': len(classes),
        'eval_metric': 'mlogloss'
    }
    num_boost_round = trial.suggest_int('num_boost_round', 100, 1000)
    pruning_callback = XGBoostPruningCallback(trial, 'test-mlogloss')
    # Perform cross-validation
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=10,
        stratified=True,
        early_stopping_rounds=50,
        as_pandas=True,
        verbose_eval=True,
        seed=20240610,
        callbacks=[pruning_callback]
    )

    # Minimization problem (negative mean of cross-validation scores)
    return cv_results['test-mlogloss-mean'].min()


pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=200, interval_steps=20)
study = optuna.create_study(direction='minimize', pruner=pruner)
study.optimize(objective, n_trials=100, n_jobs=-1)

print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open(core_path + "best_trial.txt", "w") as file:
    file.write("Best trial:\n")
    trial = study.best_trial
    file.write("  Value: {}\n".format(trial.value))
    file.write("  Params:\n")
    for key, value in trial.params.items():
        file.write("    {}: {}\n".format(key, value))

best_params = trial.params
best_num_boost_round = best_params.pop('num_boost_round', 600)
best_params.update({
    'objective': 'multi:softmax',
    'num_class': len(classes),
    'eval_metric': 'mlogloss'
})

best_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    evals=evals,
    num_boost_round=best_num_boost_round,
    early_stopping_rounds=60,

)

evals_result = best_model.evals_result()

train_loss = evals_result['validation_0']['mlogloss']
val_loss = evals_result['validation_1']['mlogloss']

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
ax.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')

ax.set_title('Training and Validation Loss')
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

ax.legend()

plt.savefig(core_path + 'training_validation_loss_opt.png', dpi=300)
plt.close()

feature_importances = best_model.get_score(importance_type='gain')
features = X_train.columns
sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

importance_df = pd.DataFrame({
    'Features': [item[0] for item in sorted_feature_importances],
    'Importance': [item[1] for item in sorted_feature_importances]
})
importance_df = importance_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Features', data=importance_df, palette='viridis')
plt.xlabel('Feature Importance (Gain)')
plt.ylabel('Features')
plt.title('Feature Importances Visualized (Gain)')
plt.savefig(core_path + 'feature_importances_gain.png', dpi=300)
plt.close()

dtest = xgb.DMatrix(X_test)
y_pred = best_model.predict(dtest)

cm = confusion_matrix(y_test, y_pred)
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentage)')
plt.tight_layout()
plt.savefig(core_path + 'confusion_matrix_percentage_opt.png', dpi=300)
plt.close()

report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

report_df = pd.DataFrame(report).transpose()
report_df = report_df[report_df.index != 'accuracy']

accuracies = {}
for label_str in label_encoder.classes_:
    label_int = label_encoder.transform([label_str])[0]
    mask = y_test == label_int
    if mask.sum() > 0:
        accuracies[label_str] = accuracy_score(y_test[mask], y_pred[mask])
    else:
        accuracies[label_str] = np.nan


report_df['accuracy'] = pd.Series(accuracies)
macro_avg_accuracy = report_df.loc[label_encoder.classes_, 'accuracy'].mean()
weights = report_df.loc[label_encoder.classes_, 'support'] / report_df.loc[label_encoder.classes_, 'support'].sum()
weighted_avg_accuracy = np.sum(report_df.loc[label_encoder.classes_, 'accuracy'] * weights)
report_df.loc['macro avg', 'accuracy'] = macro_avg_accuracy
report_df.loc['weighted avg', 'accuracy'] = weighted_avg_accuracy
report_df = report_df[['precision', 'recall', 'f1-score', 'accuracy', 'support']]
report_df.to_csv(core_path + 'classification_report_wb_opt.csv', index=True)

best_model.save_model(os.path.join(core_path, 'best_xgb_model_wb_opt.json'))