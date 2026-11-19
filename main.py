

import argparse
import json
from pathlib import Path

from common import create_run_dir, load_data, save_run


def main():
    
    parser = argparse.ArgumentParser(description="Titanic cross-validation and submission")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--task", choices=["ml", "dl", "all"])
    parser.add_argument("--model", help="Configured ML name, comma-separated names, or 'all'")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.task:
        config["task"] = args.task
    if args.model:
        if config["task"] == "dl":
            parser.error("Параметр --model выбирает классические модели; используйте --task ml или all.")
        config["ml"]["selected_models"] = list(config["ml"]["models"]) if args.model == "all" else args.model.split(",")
    if config["task"] not in {"ml", "dl", "all"}:
        parser.error("В конфиге поле task должно быть ml, dl или all.")
    names = config["ml"]["selected_models"]
    if config["task"] in {"ml", "all"} and (not names or any(name not in config["ml"]["models"] for name in names)):
        parser.error("Выберите хотя бы одну ML-модель из раздела models в config.json.")
    base_dir = config_path.parent
    X, y, X_test, ids = load_data(config, base_dir)
    summaries, folds, submissions = [], [], {}
    if config["task"] in {"ml", "all"}:
        from ml_pipeline import run_ml
        rows, fold_rows, frames = run_ml(config, X, y, X_test, ids)
        summaries.extend(rows)
        folds.extend(fold_rows)
        submissions.update(frames)
    if config["task"] in {"dl", "all"}:
        from dl_pipeline import run_dl
        rows, fold_rows, frames = run_dl(config, X, y, X_test, ids)
        summaries.extend(rows)
        folds.extend(fold_rows)
        submissions.update(frames)
    run_dir = create_run_dir(config, base_dir)
    table = save_run(config, run_dir, summaries, folds, submissions)
    print(table.to_string(index=False))
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
