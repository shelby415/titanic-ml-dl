

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(config, base_dir):
    
    data = config["data"]
    train = pd.read_csv(base_dir / data["train_path"])
    test = pd.read_csv(base_dir / data["test_path"])
    columns = config["features"]["numeric"] + config["features"]["categorical"]
    if len(columns) != len(set(columns)):
        raise ValueError("В списках признаков не должно быть повторяющихся столбцов.")
    target = data["target"]
    if target in columns or data["id_column"] in columns:
        raise ValueError("Целевая переменная и идентификатор пассажира не должны входить в признаки.")
    if train[target].isna().any() or not set(train[target].unique()) <= {0, 1}:
        raise ValueError("В целевой переменной должны быть только 0 и 1, без пропусков.")
    return train[columns].copy(), train[target].copy(), test[columns].copy(), test[data["id_column"]].copy()


def make_preprocessor(config, scale=False):
    """Собираем обработку признаков. Обучать её нужно только на train части."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numbers", Pipeline(steps), config["features"]["numeric"]),
        ("categories", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), config["features"]["categorical"]),
    ])


def make_cv(config):
    """Задаём одинаковые фолды для всех моделей с сохранением долей классов."""
    return StratifiedKFold(
        n_splits=config["validation"]["n_splits"],
        shuffle=True,
        random_state=config["seed"],
    )


def summarize(task, model, scores):
    """Считаем среднюю accuracy и её разброс между фолдами."""
    return {
        "task": task, "model": model,
        "accuracy_mean": float(np.mean(scores)),
        "accuracy_std": float(np.std(scores)),
    }


def create_run_dir(config, base_dir):
    """Создаём отдельную папку запуска, чтобы не затереть прошлые результаты."""
    path = base_dir / config["output_dir"] / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_run(config, run_dir, summaries, folds, submissions):
    """Сохраняем метрики, настройки и предсказания каждой модели."""
    table = pd.DataFrame(summaries).sort_values("accuracy_mean", ascending=False)
    table.to_csv(run_dir / "results.csv", index=False)
    pd.DataFrame(folds).to_csv(run_dir / "fold_scores.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, frame in submissions.items():
        frame.to_csv(run_dir / f"submission_{name}.csv", index=False)
    best = table.iloc[0]
    best_name = f"{best['task']}_{best['model']}"
    submissions[best_name].to_csv(run_dir / "submission.csv", index=False)
    return table


def submission_frame(config, ids, predictions):
    """Собираем submission в исходном порядке пассажиров и проверяем ответы 0/1."""
    predictions = np.asarray(predictions).reshape(-1).astype(int)
    if len(predictions) != len(ids) or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Для каждого пассажира из test нужен один ответ: 0 или 1.")
    return pd.DataFrame({config["data"]["id_column"]: ids.to_numpy(), config["data"]["target"]: predictions})
