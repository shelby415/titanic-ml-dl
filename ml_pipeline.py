"""Классические модели: проверка на фолдах и обучение на всём train для submission."""

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from common import make_cv, make_preprocessor, submission_frame, summarize


def build_model(name, config):
    params = dict(config["ml"]["models"][name])
    seed = config["seed"]
    if name in {"voting", "stacking"}:
        members = params.pop("members")
        if any(member in {"voting", "stacking"} for member in members):
            raise ValueError("В ансамбль можно включать отдельные модели, но не другие ансамбли.")
        estimators = [(member, build_pipeline(member, config)) for member in members]
        if name == "voting":
            return VotingClassifier(estimators=estimators, voting="soft", **params)
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=make_cv(config), stack_method="predict_proba", **params,
        )
    constructors = {
        "dummy": DummyClassifier,
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "decision_tree": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
        "gradient_boosting": GradientBoostingClassifier,
    }
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, **params)
    if name not in constructors:
        raise ValueError(f"Неизвестная модель: {name}")
    if name not in {"knn", "dummy"}:
        params.setdefault("random_state", seed)
    return constructors[name](**params)


def build_pipeline(name, config):
    model = build_model(name, config)
    if name in {"voting", "stacking"}:
        return model
    scaled = name in config["ml"]["scale_numeric_for"]
    return Pipeline([
        ("preprocessor", make_preprocessor(config, scale=scaled)),
        ("classifier", model),
    ])


def run_ml(config, X, y, X_test, ids):
    summaries, folds, submissions = [], [], {}
    for name in config["ml"]["selected_models"]:
        estimator = build_pipeline(name, config)
        scores = cross_validate(
            estimator, X, y, cv=make_cv(config), scoring="accuracy",
            return_train_score=True, error_score="raise",
        )
        for fold, (train_score, valid_score) in enumerate(zip(scores["train_score"], scores["test_score"]), 1):
            folds.append({"task": "ml", "model": name, "fold": fold, "train_accuracy": float(train_score), "valid_accuracy": float(valid_score)})
        summary = summarize("ml", name, scores["test_score"])
        summaries.append(summary)
        print(f"ML {name}: accuracy={summary['accuracy_mean']:.4f}, std={summary['accuracy_std']:.4f}", flush=True)
        fitted = clone(estimator).fit(X, y)
        submissions[f"ml_{name}"] = submission_frame(config, ids, fitted.predict(X_test))
    return summaries, folds, submissions
