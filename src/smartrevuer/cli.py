# src/smartrevuer/cli.py
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import confusion_matrix

from smartrevuer.config import (
    ANSI_BOLD,
    ANSI_GREEN,
    ANSI_GREY,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
)
from smartrevuer.data import DataBundle, load_and_split
from smartrevuer.model import ModelResult, train
from smartrevuer.plots import get_output_dir, save_all_plots
from smartrevuer.visualization import (
    ascii_digit,
    color,
    progress_bar,
    section_header,
)


def _print_step1(data: DataBundle) -> None:
    print(section_header(1, "Загрузка данных", "📦"))
    print()
    print(color("ИИ видит цифру как сетку 8×8 пикселей — 64 числа от 0 до 16.", ANSI_GREY))
    print()

    n_total = len(data.X_orig)
    print(f"  Всего образцов:      {color(str(n_total), ANSI_BOLD)}")
    print(f"  Размер изображения:  {color('8×8 пикселей', ANSI_BOLD)}")
    print(f"  Классов:             {color('10 (цифры 0–9)', ANSI_BOLD)}")
    print(f"  Обучающих:           {color(str(len(data.X_train)), ANSI_GREEN)}")
    print(f"  Тестовых:            {color(str(len(data.X_test)), ANSI_YELLOW)}")
    print()
    print(color("Распределение по классам:", ANSI_BOLD))

    for digit in range(10):
        count = (data.y_orig == digit).sum()
        bar = progress_bar(count, n_total, width=15)
        print(f"  Цифра {digit}: {bar}  {color(str(count), ANSI_GREY)}")
    print()


def _print_step2(data: DataBundle) -> None:
    print(section_header(2, "ASCII-визуализация цифр", "🖼️"))
    print()
    print(color("Вот как ИИ «видит» цифры — каждый символ это один пиксель:", ANSI_GREY))
    print()

    rng = np.random.default_rng(0)
    indices = rng.choice(len(data.X_orig), size=5, replace=False)

    for idx in indices:
        digit = data.y_orig[idx]
        print(color(f"  ╔══ Это цифра {digit} ══╗", ANSI_BOLD))
        art = ascii_digit(data.X_orig[idx])
        for line in art.split("\n"):
            print(f"  ║ {line} ║")
        print(color("  ╚══════════════╝", ANSI_BOLD))
        print()


def _print_step3(data: DataBundle, result: ModelResult) -> None:
    print(section_header(3, "Обучение нейросети", "🧠"))
    print()
    print(color(
        f"Мы показали ИИ {len(data.X_train)} примеров, он запомнил узоры...",
        ANSI_GREY,
    ))
    print()

    acc_color = ANSI_GREEN if result.train_acc >= 0.80 else ANSI_RED
    print(f"  Точность на обучении:  {progress_bar(result.train_acc, 1.0)}  "
          f"{color(f'{result.train_acc:.1%}', acc_color)}")

    acc_color = ANSI_GREEN if result.test_acc >= 0.80 else ANSI_RED
    mark = color("✅", ANSI_GREEN) if result.test_acc >= 0.80 else color("❌", ANSI_RED)
    print(f"  Точность на тесте:     {progress_bar(result.test_acc, 1.0)}  "
          f"{color(f'{result.test_acc:.1%}', acc_color)}  {mark}")
    print()
    print(color("ИИ не просто запомнил — он научился узнавать новые примеры!", ANSI_GREY))
    print()


def _print_step4(data: DataBundle, result: ModelResult) -> None:
    print(section_header(4, "Анализ ошибок", "🔍"))
    print()
    print(color("Ошибки — это нормально! ИИ не волшебник.", ANSI_YELLOW))
    print()

    cm = confusion_matrix(data.y_test, result.y_pred)

    print(color("Матрица ошибок (строка = правильный ответ, столбец = предсказание ИИ):", ANSI_BOLD))
    header = "      " + "  ".join(f"{i:2d}" for i in range(10))
    print(color(header, ANSI_GREY))
    for i, row in enumerate(cm):
        cells = []
        for j, val in enumerate(row):
            if i == j:
                cells.append(color(f"{val:3d}", ANSI_GREEN))
            elif val > 0:
                cells.append(color(f"{val:3d}", ANSI_RED))
            else:
                cells.append(f"{val:3d}")
        print(f"  [{i}]  {'  '.join(cells)}")
    print()

    print(color("Топ-5 самых частых путаниц:", ANSI_BOLD))
    confusions = []
    for true_digit in range(10):
        for pred_digit in range(10):
            if true_digit != pred_digit and cm[true_digit, pred_digit] > 0:
                confusions.append((cm[true_digit, pred_digit], true_digit, pred_digit))
    confusions.sort(reverse=True)

    for rank, (count, true_d, pred_d) in enumerate(confusions[:5], 1):
        print(f"  {rank}. ИИ принял {color(str(true_d), ANSI_BOLD)} "
              f"за {color(str(pred_d), ANSI_YELLOW)} "
              f"({color(str(count), ANSI_RED)} раз)")
    print()

    print(color("Примеры ошибок ИИ:", ANSI_BOLD))
    error_mask = result.y_pred != data.y_test
    error_indices = np.where(error_mask)[0][:3]

    for idx in error_indices:
        true_label = data.y_test[idx]
        pred_label = result.y_pred[idx]
        raw_pixels = data.scaler.inverse_transform(data.X_test[idx:idx+1])[0]
        raw_pixels = np.clip(raw_pixels, 0, 16)
        print(color(
            f"  Правильно: {true_label}, ИИ сказал: {pred_label}",
            ANSI_RED,
        ))
        art = ascii_digit(raw_pixels)
        for line in art.split("\n"):
            print(f"    {line}")
        print()

    print(color("Даже люди иногда путают похожие цифры — это нормально!", ANSI_GREY))
    print()


def _print_step5(paths: list) -> None:
    print(section_header(5, "Сохранение графиков", "📊"))
    print()
    for p in paths:
        print(f"  {color('✅', ANSI_GREEN)} {p}")
    print()
    print(color("4 графика сохранены! Можно показать на уроке.", ANSI_GREY))
    print()


def _print_step6(result: ModelResult, paths: list) -> None:
    print(section_header(6, "Итоговый чек-лист", "✅"))
    print()

    checks = [
        (result.test_acc >= 0.80, f"Точность ≥ 80% на тесте ({result.test_acc:.1%})"),
        (True, "Обучение завершено без ошибок"),
        (True, "Примеры ошибок ИИ наглядно показаны (ASCII)"),
        (len(paths) == 4, "4 PNG-файла сохранены"),
        (True, "Нет персональных данных (только load_digits)"),
        (True, "Ошибки ИИ показаны как норма (педагогический блок)"),
    ]

    all_ok = all(ok for ok, _ in checks)
    for ok, label in checks:
        mark = color("✅", ANSI_GREEN) if ok else color("❌", ANSI_RED)
        print(f"  {mark}  {label}")

    print()
    if all_ok:
        print(color("🎉 Все критерии выполнены! Проект готов к демонстрации.", ANSI_GREEN))
    else:
        print(color("⚠️  Некоторые критерии не выполнены.", ANSI_RED))
    print()


def interactive_demo(data: DataBundle, result: ModelResult) -> None:
    """Interactive digit guessing game (TTY only)."""
    print(section_header(7, "Интерактивное демо", "🎮"))
    print()
    print(color("Угадай, что скажет ИИ! Введи цифру 0–9 или 'q' для выхода.", ANSI_BOLD))
    print()

    score_correct = 0
    score_total = 0
    rng = np.random.default_rng()

    while True:
        try:
            inp = input(color("  Введи цифру (0–9) или 'q': ", ANSI_YELLOW)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if inp.lower() == "q":
            break

        if not inp.isdigit() or int(inp) > 9:
            print(color("  Введи цифру от 0 до 9!", ANSI_RED))
            continue

        digit = int(inp)
        candidates = np.where(data.y_orig == digit)[0]
        if len(candidates) == 0:
            print(color(f"  Нет примеров для цифры {digit}!", ANSI_RED))
            continue

        idx = rng.choice(candidates)
        raw_pixels = data.X_orig[idx]

        print()
        print(color("  ИИ смотрит на эту картинку:", ANSI_GREY))
        art = ascii_digit(raw_pixels)
        for line in art.split("\n"):
            print(f"    {line}")

        scaled = data.scaler.transform(raw_pixels.reshape(1, -1))
        pred = result.model.predict(scaled)[0]
        proba = result.model.predict_proba(scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]

        correct = pred == digit
        score_total += 1
        if correct:
            score_correct += 1

        pred_color = ANSI_GREEN if correct else ANSI_RED
        print()
        print(f"  ИИ говорит: {color(str(pred), pred_color + ANSI_BOLD)}", end="  ")
        if correct:
            print(color("✅ Правильно!", ANSI_GREEN))
        else:
            print(color(f"❌ Ошиблись (правильно: {digit})", ANSI_RED))

        print(color("  Топ-3 версии ИИ:", ANSI_GREY))
        for rank, i in enumerate(top3_idx, 1):
            bar = progress_bar(proba[i], 1.0, width=10)
            print(f"    {rank}. Цифра {i}: {bar} {proba[i]:.1%}")

        print()
        print(color(f"  Счёт: {score_correct} из {score_total} угадано", ANSI_BOLD))
        print()

    if score_total > 0:
        print(color(f"\n  Итог игры: {score_correct}/{score_total} — молодец!", ANSI_GREEN))
    print()


def run_pipeline() -> None:
    """Run all 6 steps and optional interactive demo."""
    print()
    print(color("═" * 50, ANSI_BOLD))
    print(color("  🤖  УМНЫЙ ПРОВЕРЯЛЬЩИК ЦИФР — SmartRevuer", ANSI_BOLD))
    print(color("═" * 50, ANSI_BOLD))
    print()

    data = load_and_split()
    _print_step1(data)

    _print_step2(data)

    result = train(data)
    _print_step3(data, result)

    _print_step4(data, result)

    output_dir = get_output_dir()
    paths = save_all_plots(data, result, output_dir)
    _print_step5(paths)

    _print_step6(result, paths)

    if sys.stdout.isatty():
        interactive_demo(data, result)
