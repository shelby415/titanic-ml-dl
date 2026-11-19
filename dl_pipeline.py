"""Обучение MLP: в каждом фолде заново создаём обработку данных и сеть."""

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import make_cv, make_preprocessor, submission_frame, summarize


def set_seed(seed):
    """Фиксируем случайность перед обучением, чтобы можно было повторить запуск."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_network(input_size, settings):
    """Собираем слои сети. На выходе один логит, sigmoid отдельно не добавляем."""
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
    activation = activations[settings["activation"]]
    layers = []
    for width in settings["hidden_sizes"]:
        layers.append(nn.Linear(input_size, width))
        if settings["batchnorm"]:
            layers.append(nn.BatchNorm1d(width))
        layers.append(activation())
        if settings["dropout"]:
            layers.append(nn.Dropout(settings["dropout"]))
        input_size = width
    layers.append(nn.Linear(input_size, 1))
    return nn.Sequential(*layers)


def train_network(X, y, config):
    """Обучаем новую сеть на CPU заданное число эпох без валидационных строк."""
    settings = config["dl"]
    set_seed(config["seed"])
    X_tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_tensor = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
    batch_size = settings["batch_size"] or len(X_tensor)
    if settings["batchnorm"] and (batch_size < 2 or len(X_tensor) < 2):
        raise ValueError("Для BatchNorm нужно хотя бы два обучающих объекта в батче.")
    # Если в последнем батче один пассажир, пропускаем его: BatchNorm нужно хотя бы два.
    drop_last = settings["batchnorm"] and len(X_tensor) % batch_size == 1
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor), batch_size=batch_size,
        shuffle=True, drop_last=drop_last,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    model = build_network(X_tensor.shape[1], settings)
    optimizer_args = {"lr": settings["learning_rate"], "weight_decay": settings["weight_decay"]}
    if settings["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), momentum=settings["momentum"], **optimizer_args)
    elif settings["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    else:
        raise ValueError("Для нейросети выберите оптимизатор 'adam' или 'sgd'.")
    scheduler = None
    if settings["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=settings["epochs"], eta_min=settings["min_lr"],
        )
    elif settings["scheduler"] is not None:
        raise ValueError("Для scheduler укажите null (без него) или 'cosine'.")
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(settings["epochs"]):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model


def predict(model, X):
    """Получаем классы 0/1: логит от нуля соответствует вероятности от 50%."""
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(np.asarray(X), dtype=torch.float32)
        return (model(tensor).reshape(-1) >= 0).numpy().astype(int)


def run_dl(config, X, y, X_test, ids):
    """Проверяем сеть на фолдах, затем обучаем новую на всём train для submission."""
    torch.set_num_threads(config["dl"]["num_threads"])
    scores, folds = [], []
    for fold, (train_idx, valid_idx) in enumerate(make_cv(config).split(X, y), 1):
        preprocessor = make_preprocessor(config, scale=True)
        train_ready = preprocessor.fit_transform(X.iloc[train_idx])
        valid_ready = preprocessor.transform(X.iloc[valid_idx])
        model = train_network(train_ready, y.iloc[train_idx], config)
        train_accuracy = float(np.mean(predict(model, train_ready) == y.iloc[train_idx].to_numpy()))
        accuracy = float(np.mean(predict(model, valid_ready) == y.iloc[valid_idx].to_numpy()))
        scores.append(accuracy)
        folds.append({"task": "dl", "model": "mlp", "fold": fold, "train_accuracy": train_accuracy, "valid_accuracy": accuracy})
        print(f"DL fold {fold}: accuracy={accuracy:.4f}", flush=True)
    summary = summarize("dl", "mlp", scores)
    preprocessor = make_preprocessor(config, scale=True)
    train_ready = preprocessor.fit_transform(X)
    test_ready = preprocessor.transform(X_test)
    final_model = train_network(train_ready, y, config)
    submission = submission_frame(config, ids, predict(final_model, test_ready))
    return [summary], folds, {"dl_mlp": submission}
