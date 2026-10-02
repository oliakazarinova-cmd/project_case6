# Проверка подготовленного варианта

Основа: https://github.com/MariaKhudiakova/cirrhosis-prediction

Исходный commit: `b6fea086dbd282f99d86b9731c839d6b5c8e1460`.

Проверено в Linux, Python 3.12.14, Poetry 2.2.1, scikit-learn 1.7.2.
Windows отдельно не проверялся; инструкция использует команды для PowerShell.

| Проверка | Результат |
| --- | --- |
| `poetry lock` | Lock-файл сформирован Poetry |
| `poetry install --with dev` | Зависимости и пакет установлены в `.venv` |
| `poetry check --lock` | All set! |
| `poetry run pytest -q` | 4 passed |
| `poetry run pre-commit install` | Git-хук установлен в тестовом клоне |
| `poetry run pre-commit run --all-files` | black, isort, flake8: Passed |
| `poetry run cirrhosis-train --config configs/train.toml` | Pipeline и метрики сохранены |
| `poetry run cirrhosis-predict --input data/raw_data.csv` | CSV с метками и вероятностями создан |

## Результаты нового сценария

Данные: `data/raw_data.csv`, 7905 строк. Обучающая часть: 6324,
тестовая: 1581. Разделение стратифицированное, random_state=8.
Предобработка включена в Pipeline; кросс-валидация — 5 фолдов.

| Метрика | Значение |
| --- | ---: |
| CV log loss, среднее | 0.525595 |
| CV log loss, стандартное отклонение | 0.011951 |
| Test log loss | 0.524183 |
| Test accuracy | 0.789374 |
| Test F1 macro | 0.517384 |

Это результаты фиксированной базовой модели, а не результат нового выбора
лучшего алгоритма. Метрики исходных ноутбуков не переиспользуются.
Качество не доказывает клиническую пригодность модели.

SHA-256 исходного CSV:
`cdd70050ef50682c8ee5f17779d9076e5ac03858f89b10f44c37a82eb1e2e952`.

На GitHub файлы автоматически не отправлялись. Для сдачи их нужно добавить
в свой клон, проверить на своём компьютере, закоммитить и отправить в ветку/PR.
