"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


DURATION_COLUMN = 1
AMOUNT_COLUMN = 4
MAX_CATEGORIES = 10


def add_credit_features(features):
    features = np.asarray(features, dtype=float)
    amount = features[:, AMOUNT_COLUMN]
    duration = features[:, DURATION_COLUMN]
    per_month = amount / np.maximum(duration, 1.0)
    return np.column_stack([features, per_month, np.log1p(amount), np.log1p(duration)])


def find_categorical_columns(train_features):
    columns = []
    for i in range(train_features.shape[1]):
        values = train_features[:, i]
        if len(np.unique(values)) <= MAX_CATEGORIES and np.all(values == np.round(values)):
            columns.append(i)
    return columns


def build_model(train_features, random_seed):
    n_original = train_features.shape[1]
    categorical = find_categorical_columns(train_features)
    numeric = [i for i in range(n_original) if i not in categorical]
    numeric += [n_original, n_original + 1, n_original + 2]
    adder = FunctionTransformer(add_credit_features)

    logistic = make_pipeline(
        adder,
        ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]),
        LogisticRegression(C=0.1, max_iter=2000),
    )
    forest = make_pipeline(
        adder,
        RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=random_seed),
    )
    extra_trees = make_pipeline(
        adder,
        ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, random_state=random_seed),
    )
    return VotingClassifier(
        [("logistic", logistic), ("forest", forest), ("extra_trees", extra_trees)],
        voting="soft",
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    train_features = np.asarray(data.train_features, dtype=float)
    train_target = np.asarray(data.train_target).ravel()
    test_features = np.asarray(data.test_features, dtype=float)
    test_target = np.asarray(data.test_target).ravel()

    model = build_model(train_features, data.random_seed)
    model.fit(train_features, train_target)

    positive_index = list(model.classes_).index(1) if 1 in model.classes_ else len(model.classes_) - 1
    positive_class = model.classes_[positive_index]
    probabilities = model.predict_proba(test_features)[:, positive_index]
    predictions = (probabilities >= 0.5).astype(int) if positive_class == 1 else model.predict(test_features)

    return EnsembleResult(
        model_name="Soft Voting: LogisticRegression + RandomForest + ExtraTrees",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(test_target, probabilities)),
        accuracy=float(accuracy_score(test_target, predictions)),
        precision=float(precision_score(test_target, predictions, zero_division=0)),
        recall=float(recall_score(test_target, predictions, zero_division=0)),
        f1=float(f1_score(test_target, predictions, zero_division=0)),
    )