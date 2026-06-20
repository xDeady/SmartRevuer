# SmartRevuer — Умный проверяльщик цифр: Design Spec

**Date:** 2026-06-20  
**Status:** Approved

---

## Overview

Учебный ИИ-проект для демонстрации школьникам 1–3 классов (6–9 лет), как компьютер
учится распознавать рукописные цифры от 0 до 9. Реализован как полноценный
Python-пакет с правильной архитектурой, тестами и Makefile.

**Запуск:** `make run` или `python -m smartrevuer`

---

## Architecture

### File Structure

```
SmartRevuer/
├── src/
│   └── smartrevuer/
│       ├── __init__.py
│       ├── __main__.py       # точка входа: python -m smartrevuer
│       ├── config.py         # все константы (слои, пороги, пути, цвета)
│       ├── data.py           # загрузка, StandardScaler, train/test split
│       ├── model.py          # MLPClassifier: обучение, предсказание, метрики
│       ├── visualization.py  # ASCII-рендер цифр, ANSI-утилиты, progress bars
│       ├── plots.py          # 4 PNG через matplotlib (без plt.show())
│       └── cli.py            # оркестрация 6 шагов + интерактивное демо
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_visualization.py
├── outputs/                  # .gitignored, сюда сохраняются PNG
├── pyproject.toml            # зависимости, entry point, конфиг ruff/pytest
├── requirements.txt
├── Makefile                  # make run / make test / make lint
├── .gitignore
└── README.md
```

### Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Константы: `HIDDEN_LAYERS = (128, 64)`, `TEST_SIZE = 0.2`, пути вывода, ANSI-коды цветов |
| `data.py` | `load_and_split()` → `DataBundle` (X_train, X_test, y_train, y_test, scaler, X_orig, y_orig); X_orig/y_orig нужны для интерактивного демо |
| `model.py` | `train(data)` → `ModelResult` (model, y_pred, train_acc, test_acc) |
| `visualization.py` | `ascii_digit()`, `color()`, `progress_bar()`, `section_header()` |
| `plots.py` | `save_all_plots(data, result, output_dir)` → список из 4 путей PNG |
| `cli.py` | `run_pipeline()` — шаги 1–6, затем `interactive_demo()` при `sys.stdout.isatty()` |
| `__main__.py` | Вызывает `cli.run_pipeline()` |

---

## Neural Network

- **Вход:** 64 признака (8×8 пикселей, значения 0–16)
- **Нормализация:** `StandardScaler` (mean=0, std=1) — fit только на train
- **Скрытые слои:** 128 → 64 нейрона, активация ReLU
- **Выход:** 10 классов (цифры 0–9), softmax
- **Разбивка:** 80% train / 20% test, `stratify=y`, `random_state=42`
- **Датасет:** `sklearn.datasets.load_digits` (1797 образцов)

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=500,
    random_state=42,
)
```

---

## 6 Обязательных Шагов

### Шаг 1: Загрузка данных
- Вывод статистики: кол-во образцов, размер изображения, распределение по классам
- Прогресс-бары из `█░` для каждого класса

### Шаг 2: ASCII-визуализация
- Показать 5 случайных примеров из датасета
- Символы `░▒▓█` по 4 порогам яркости (0/5/10/14 из диапазона 0–16)
- Сетка 8×8, над каждой — подпись «Это цифра N»

### Шаг 3: Обучение
- Вывод точности на train и test
- Прогресс-бар точности, зелёный ≥80%, красный <80%
- Педагогический текст: «Мы показали ИИ X примеров…»

### Шаг 4: Анализ ошибок
- Матрица ошибок 10×10 в терминале (цифры выровнены)
- Топ-5 самых частых путаниц: «ИИ принял N за M (K раз)»
- ASCII-примеры 3 неправильно распознанных цифр
- Педагогический текст: «Ошибки — это нормально»

### Шаг 5: Сохранение 4 PNG
1. `01_digit_examples.png` — 2 ряда × 10 цифр, по одному примеру каждого класса
2. `02_confusion_matrix.png` — цветная heatmap с числами внутри
3. `03_accuracy_per_digit.png` — горизонтальный bar chart (зелёный/жёлтый/красный)
4. `04_network_diagram.png` — схема сети: блоки слоёв со стрелками через matplotlib patches

**Выходная папка:** пробуем `/mnt/user-data/outputs/`, при отсутствии прав — `./outputs/`

### Шаг 6: Итоговый чек-лист
- ✅/❌ по каждому критерию успеха
- Точность ≥ 80% на тесте
- Обучение завершено без ошибок
- Примеры ошибок показаны
- 4 PNG сохранены
- Этический блок: нет персональных данных, ошибки показаны как норма

---

## Интерактивное Демо (TTY only)

Запускается только при `sys.stdout.isatty() == True`.

- Пользователь вводит цифру 0–9 или 'q' для выхода
- Скрипт берёт случайный пример с `np.random.choice(np.where(y == digit)[0])`
- Показывает ASCII-изображение, предсказание ИИ, топ-3 вероятности через `predict_proba()`
- Ведёт счёт «X из Y угадано»

---

## Terminal Output Style

- **ANSI escape-коды:** зелёный для успеха, красный для ошибок, серый для метаданных
- **Заголовки секций:** `─── 📦 ШАГ 1: Название ───`
- **Прогресс-бары:** `████████░░ 80%`
- **Педагогические вставки:** простой язык без терминов в основном потоке

---

## Tests

| Файл | Что проверяет |
|---|---|
| `test_data.py` | Формы массивов, стратификация, нормализация (mean≈0, std≈1), DataBundle поля |
| `test_model.py` | Smoke-тест обучения, точность ≥ 80%, форма `y_pred`, ModelResult поля |
| `test_visualization.py` | ASCII содержит `░▒▓█`, длина каждой строки = 8 символов, ANSI-коды применяются |

---

## Dependencies

```
scikit-learn>=1.3
matplotlib>=3.7
numpy>=1.24
seaborn>=0.12
```

**Dev:** `ruff`, `pytest`

---

## Success Criteria

- [ ] Точность ≥ 80% на тесте
- [ ] Обучение завершено без ошибок
- [ ] Примеры ошибок ИИ наглядно показаны (ASCII)
- [ ] 4 файла PNG сохранены
- [ ] Нет персональных данных (только sklearn.datasets.load_digits)
- [ ] Ошибки ИИ показаны как норма (педагогический этический блок)
- [ ] `pytest tests/` проходит без ошибок
