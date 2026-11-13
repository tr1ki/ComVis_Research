# CNN vs ViT: устойчивость к искажениям на CIFAR-10-C

Сравниваем, как аугментации при дообучении влияют на устойчивость CNN и Vision Transformers. В экспериментах участвуют ResNet-50, ConvNeXt-T, DeiT-S и Swin-T с весами ImageNet-1k. Обучение проходит на CIFAR-10, оценка — на исходных изображениях и CIFAR-10-C.

В репозитории сохранены 72 результата: 24 комбинации модели и режима обучения, по три seed на каждую. Датасеты и обученные веса в репозиторий не входят.

## Команда

| Участник | Роль |
|---|---|
| **Balakarev Sergey** | Project Management & Core Architecture |
| **Sultan Abayev** | Backend Development & Data Processing |
| **Erasyl Umirbay** | Frontend, Testing & Analytics |

### Выполненные задачи

**Balakarev Sergey** — планирование этапов исследования, общий протокол сравнения моделей и архитектура кода обучения и оценки.

**Sultan Abayev** — загрузка и подготовка данных, реализация аугментаций, обработка батчей, сохранение метрик в JSON и очередь запусков на GPU.

**Erasyl Umirbay** — оформление ноутбуков и результатов, проверка метрик и таблиц, статистический анализ и графики. Frontend здесь относится к представлению экспериментов в ноутбуках и отчётах.

Подробное распределение задач — в [docs/TEAM.md](docs/TEAM.md).

## Структура проекта

```text
.
├── README.md
├── docs/
│   └── TEAM.md                     # роли и выполненные задачи
├── notebooks/
│   ├── round1_single_run.ipynb     # ResNet-50 и DeiT-S: baseline, noise+blur, AugMix
│   ├── run_A_new_models.ipynb      # ConvNeXt-T и Swin-T
│   ├── run_B_transfer.ipynb        # перенос между семействами искажений
│   ├── run_C_probability.ipynb     # вероятность применения аугментаций
│   └── run_D_intensity.ipynb       # интенсивность аугментаций
├── scripts/
│   ├── cv_robustness.py            # первый этап экспериментов
│   ├── cv_robustness_v2.py         # четыре модели и дополнительные режимы
│   ├── analyze.py                 # основные таблицы и графики
│   └── analyze_v2.py              # сравнение четырёх моделей и расширенный анализ
├── results/
│   └── *.json                     # 72 файла: модель, режим и seed
└── analysis/
    ├── results_summary.md         # основные результаты
    ├── results_summary_v2.md      # расширенный анализ
    ├── table_main.csv
    └── fig_*.png                  # семь графиков
```

## Эксперименты

Используются модели `resnet50.a1_in1k`, `convnext_tiny.fb_in1k`, `deit_small_patch16_224.fb_in1k` и `swin_tiny_patch4_window7_224.ms_in1k` из `timm`.

Общий режим дообучения: AdamW, learning rate `1e-4`, weight decay `0.05`, batch size `128`, 6 эпох, warm-up на первых 5% шагов и cosine schedule. Label smoothing — `0.1`, AMP включается на CUDA. Изображения увеличиваются с 32×32 до 224×224 билинейной интерполяцией. Seed: `0`, `1`, `2`.

Для ResNet-50 и DeiT-S проверены baseline (crop + flip), noise+blur (`corrupt_aug`), AugMix, отдельные семейства noise, blur и digital, а также варианты вероятности и интенсивности noise+blur. Для ConvNeXt-T и Swin-T сохранены baseline и `corrupt_aug`. AugMix используется с ограниченным набором операций и без JSD loss.

Каждый запуск оценивается на 15 искажениях CIFAR-10-C при пяти уровнях силы. Основная метрика — относительное падение точности: `(clean accuracy − corrupted accuracy) / clean accuracy`. Группы seen и unseen обозначают семейства noise+blur и weather+digital относительно режима `corrupt_aug`; тестовые изображения при обучении не используются.

## Как запустить

Ноутбуки рассчитаны на Kaggle с включённым интернетом. Для `run_A`–`run_D` использовались две NVIDIA Tesla T4: по одному заданию на GPU.

Подключите датасеты [CIFAR-10 Python](https://www.kaggle.com/datasets/pankrzysiu/cifar10-python) и [CIFAR10-C](https://www.kaggle.com/datasets/faigarvn/cifar10-c). Если они не подключены, код скачает данные самостоятельно. Для короткой проверки выставьте `QUICK_TEST = True`, затем для полного эксперимента верните `False` и выполните Save & Run All. Результаты сохраняются в `results/*.json`; ноутбуки `run_A`–`run_D` также собирают ZIP-архив.

Чтобы пересобрать отчёты по готовым результатам локально:

```bash
python -m pip install numpy scipy matplotlib
python -X utf8 scripts/analyze.py results analysis
python -X utf8 scripts/analyze_v2.py results analysis
```

Для обучения нужны PyTorch, torchvision и timm. Перед локальным запуском v2 задайте `CV_DATA_DIR` для хранения датасетов; модель и режим выбираются через `CV_MODEL` и `CV_CONDITION`. Версии библиотек пока не закреплены. Повторный запуск с тем же именем модели, режимом и seed перезаписывает соответствующий JSON.

## Данные и источники

CIFAR-10 — Krizhevsky (2009), CIFAR-10-C — Hendrycks и Dietterich (ICLR 2019). Датасеты не распространяются вместе с кодом. CIFAR-10-C загружается с [Zenodo](https://zenodo.org/records/2535967), предобученные веса — через `timm` из Hugging Face Hub.
