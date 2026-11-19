# Titanic: ML и MLP

Учебный проект бинарной классификации: предсказать `Survived` для пассажиров из Kaggle test. Сравниваю модели по accuracy на пяти одинаковых стратифицированных фолдах.

## Установка

Проверено на Python 3.13. Команды выполняются из папки проекта. Исходные `train.csv` и `test.csv` находятся рядом с `main.py`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
## Запуск

После установки одна команда запускает обработку, CV, финальное обучение на всём train и формирование submission:

```powershell
.\.venv\Scripts\python.exe main.py
```

По умолчанию запускается LightGBM. Если нужное окружение уже активно, достаточно `python main.py`.

```powershell
python main.py --task ml --model random_forest
python main.py --task ml --model logistic_regression,knn,lightgbm
python main.py --task ml --model all
python main.py --task dl
python main.py --task all
python main.py --config config.json
```

`--task all` запускает выбранные в конфиге ML-модели и одну MLP. `--model all` запускает все ML-модели, включая voting и stacking; этот вариант дольше.

Для каждого запуска создаётся новая папка `outputs/<дата_время>/`:

- `results.csv` — средняя accuracy и стандартное отклонение между фолдами;
- `fold_scores.csv` — train и validation accuracy каждого фолда;
- `config.json` — настройки с учётом аргументов командной строки;
- `submission_<task>_<model>.csv` — предсказания каждой модели;
- `submission.csv` — предсказания модели с максимальной средней CV в этом запуске.


## Как менять эксперимент

Все настройки находятся в `config.json`:

- `task`: `ml`, `dl` или `all`;
- `seed`, `validation.n_splits`: случайность и число фолдов;
- `data`: пути к CSV, имя цели и идентификатора;
- `features.numeric`, `features.categorical`: исходные признаки;
- `ml.selected_models`: список ML-моделей для запуска;
- `ml.models.<имя>`: параметры конструктора соответствующей модели;
- `ml.scale_numeric_for`: модели, которым добавляется StandardScaler;
- `dl`: архитектура, оптимизатор и режим обучения MLP.


## Структура и ход выполнения

```text
main.py              аргументы запуска и вызов нужного пайплайна
config.json          данные, модели и гиперпараметры
common.py            загрузка, препроцессинг, CV, сохранение
ml_pipeline.py       классические модели и ансамбли
dl_pipeline.py       MLP и цикл обучения PyTorch
EDA.ipynb            анализ исходных данных с выводами и графиками
experiments.ipynb    прежние ручные эксперименты и блок продолжения DL
requirements.txt    окружение
```

## Результаты и ограничения сравнения

В самостоятельной серии экспериментов использовались семь исходных признаков, без FamilySize и HasCabin. Зафиксированы, в частности:

| Эксперимент                                   | Средняя CV accuracy |
| --------------------------------------------- | ------------------: |
| Логрегрессия, исторический вариант без scaler |              79,80% |
| Random Forest, глубина 5                      |              82,94% |
| CatBoost, 500 деревьев, depth 4               |              83,72% |
| LightGBM, 300 деревьев, 8 листьев             |              83,72% |
| MLP, SGD, momentum 0,9, batch 32, 100 эпох    |              82,49% |
| MLP, тот же режим с batch 64                  |              81,93% |

