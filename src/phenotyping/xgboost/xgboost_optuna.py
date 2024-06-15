
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import optuna
from joblib import dump

base_path = '/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/phenotyping/xgboost/standard'
data_path = ('/gpfs/bwfor/work/ws/hd_bm327-myeloma_standal_ws/phenotyping/manual_phenotypes_standard.csv')
mapping_path = os.path.join(base_path, 'phenotype_label_mapping.csv')

def preprocess_data(filepath):
    """
    XGBoost2 can handle missing values. I substituted them in the column called sitance_to_bone with -999, that is why I am replacing them with NaN again.
    Data here is transformed with arcsinh (IMC data), adjust to your needs.
    """

    data = pd.read_csv(data_path)
    data.drop(columns=["index", "Y_centroid", "X_centroid"], inplace=True)
    transformed = np.arcsinh(data.iloc[:, 0:32])
    data.drop(columns=data.columns[0:32], inplace=True)
    phenotypes = pd.concat([transformed, data], axis=1)
    phenotypes['distance_to_bone'] = phenotypes['distance_to_bone'].replace(-999, np.nan)
    return phenotypes


def encode_labels(y):
    """
    Encoding the string phenotypes. A translation is saved
    """

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_mapping = pd.DataFrame({'Phenotype': y, 'EncodedLabel': y_encoded}).drop_duplicates()
    label_mapping.to_csv(mapping_path, index=False)
    return y_encoded, label_encoder

def main():
    """
    Main function with Optuna tuning, IMPORTANT: Adjust the n_trials, 100 trials might already run several hours on a cluster GPU if cv=10 is used.
    Data is split 80/10/10 and only Train+Validation is used for Hyperparameter tuning in the Optuna Study, while Test data is held out completely.
    Custom Cross validation is included and sample weights are ysed to adjust class imbalances.
    The model will be saved twice, using joblib and xgboost built-in option
    """

    phenotypes = preprocess_data(data_path)
    X = phenotypes.iloc[:, :-1]
    y = phenotypes.iloc[:, -1]
    y_encoded, label_encoder = encode_labels(y)

    classes = np.unique(y_encoded)
    weights = compute_class_weight('balanced', classes=classes, y=y_encoded)
    class_weights = dict(zip(classes, weights))

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded, test_size=0.10, random_state=20240610)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=20240610)

    sample_weights = np.array([class_weights[class_label] for class_label in y_train])
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=20240610)

    def objective(trial):
        """
        There are many parameters to tune. Adjust them according to your data. These should however include most of the range of parameters.
        """
        param = {
            'verbosity': 0,
            'objective': 'multi:softmax',
            'tree_method': 'gpu_hist',
            'lambda': trial.suggest_float('lambda', 1e-4, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-4, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1200, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'num_class': len(np.unique(y_encoded)),
            'eval_metric': 'mlogloss'
        }

        f1_scores = []
        for train_index, test_index in kf.split(X_train_val, y_train_val):
            X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[test_index]
            y_train, y_val = y_train_val[train_index], y_train_val[test_index]
            sample_weights = np.array([class_weights[class_label] for class_label in y_train])

            model = XGBClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=2, sample_weight=sample_weights)

            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds, average='macro')
            f1_scores.append(f1)
        return np.mean(f1_scores)


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200, n_jobs=-1) # CONSIDER ADJUSTING

    with open(os.path.join(base_path,"best_trial.txt"), "w") as file:
        file.write("Best trial:\n")
        trial = study.best_trial
        file.write("  Value: {}\n".format(trial.value))
        file.write("  Params:\n")
        for key, value in trial.params.items():
            file.write("    {}: {}\n".format(key, value))

    best_params = trial.params
    best_model = XGBClassifier(**best_params)
    eval_s = [(X_train, y_train), (X_val, y_val)]
    best_model.fit(
        X_train, 
        y_train, 
        eval_set=eval_s,  
        eval_metric=["mlogloss", "auc"] 
        verbose=True, 
        early_stopping_rounds=50,
        sample_weight=sample_weights,
    )

    evals_result = best_model.evals_result_

    train_loss = evals_result['validation_0']['mlogloss']
    val_loss = evals_result['validation_1']['mlogloss']
    train_auc = evals_result['validation_0']['auc']
    val_auc = evals_result['validation_1']['auc']

    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    ax1.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(range(1, len(train_auc) + 1), train_auc, label='Training AUC')
    ax2.plot(range(1, len(val_auc) + 1), val_auc, label='Validation AUC')
    ax2.set_title('Training and Validation AUC')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('AUC')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'training_validation_loss_auc_opt.png'), dpi=300)
    plt.close()

    y_pred = best_model.predict(X_test)

    plt.style.use('ggplot')
    feature_importances = best_model.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({
        'Features': features,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Features', data=importance_df, palette='viridis')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importances Visualized')
    plt.savefig(os.path.join(base_path, 'feature_importances_opt.png'), dpi=300)
    plt.close()

    cm = confusion_matrix(y_test, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Percentage)')
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, 'confusion_matrix_percentage_opt.png'), dpi=300)
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
    report_df.to_csv(os.path.join(base_path, 'classification_report_wb_opt.csv'), index=True)

    dump(best_model, os.path.join(base_path, 'best_xgb_model.joblib'))
    best_model.save_model(os.path.join(base_path, 'best_xgb_model.json'))
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.show()
    fig.savefig(os.path.join(base_path, 'optuna_parameter_importance.png'), dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
