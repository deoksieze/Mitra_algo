# SymmetryDetection: текущая кодовая база

Обновлено 9 октября 2026 года после этапов 1–8 и удаления старой C++/MATLAB-ветки.
Рабочий проект — самостоятельный C++17 детектор с CPU-reference и optional
CUDA-backend. Команды сборки и запуска: [RUNNING_RU.md](RUNNING_RU.md).
Результаты проверок, времени и памяти: [CUDA_STAGE8_RU.md](CUDA_STAGE8_RU.md).

## Что реализовано

Треугольная поверхность → локальные признаки → похожие пары точек → голоса
за преобразования → mean shift → геометрическая проверка → два участка поверхности.
Это общая схема алгоритма Митры для частичных и приближённых симметрий.
Текущая реализация поддерживает поворот со сдвигом и чистое отражение в плоскости;
масштаб всегда равен 1. Полное воспроизведение статьи или старой MATLAB-версии
не подтверждено. MATLAB, Eigen, libigl и NanoGUI для текущей сборки не нужны.

Кластер голосов — гипотеза. Подтверждение задают участки, coverage и RMS после
проверки преобразования по поверхности. CLI работает без окна; результат
можно посмотреть в автономном HTML-viewer.

## Карта файлов

| Каталог / файл | Назначение |
|---|---|
| `CMakeLists.txt` | Явные CPU/CUDA targets, CLI, CTest и benchmarks. CUDA по умолчанию выключена. |
| `include/symmetry/` | Публичные типы, параметры, Backend, алгоритм и ввод/вывод. |
| `src/core/` | Общая математика, подготовка mesh/topology, sampling и управление pipeline. |
| `src/cpu/` | CPU-reference каждого вычислительного этапа и surface BVH. |
| `src/cuda/` | CUDA kernels, device buffers, GPU геометрия и optional profiling. |
| `src/io/` | Чтение OFF/OBJ, запись JSON/CSV/OFF. |
| `src/app/main.cpp` | Разбор CLI, выбор backend, вывод прогресса и экспорт. |
| `tests/` | Геометрия, данные/контракты, CPU/GPU сравнения, pipeline и viewer smoke. |
| `examples/` | reflection, rigid, partial, single: синтетические OFF модели. |
| `tools/` | Генератор demo, renderer/viewer, загрузчик Stanford Bunny. |
| `benchmarks/` | Изолированные pairing/clustering, pipeline runner, CUDA profile. |
| `benchmarks/results/` | Сохранённые измерения, параметры окружения и логи этапа 8. |
| `CPU_STAGE*.md`, `CUDA_STAGE*.md` | Отчёты этапов: формулы, контракты, проверки и ограничения на момент этапа. |
| `RUNNING_RU.md` | Актуальные команды для пользователя. |
| `LICENSE` | Лицензия проекта, MPL-2.0. |
| `build/`, `build-cuda/` | Локальные сборки текущего проекта; игнорируются Git. |
| `outputs/` | Локальные результаты и viewers текущего проекта; игнорируются Git. |

## Поток вычислений

`src/app/main.cpp` читает mesh и создаёт `make_cpu_backend()` либо
`make_cuda_backend()`. `src/core/pipeline.cpp::detect` вызывает этапы по порядку:

1. **prepare** — проверка параметров/геометрии, обработка faces, optional
   subdivision, box size/center, нормали, area weights, рёбра и CSR смежность.
   Это общий host этап в `src/core/prepare.cpp`.
2. **curvature** — tensor по внутренним рёбрам: подписанный угол между нормалями
   граней × длина ребра / средняя длина; gather на вершинах, smoothing и
   аналитическая 2×2 eigen-задача в касательной плоскости. Выход: normal,
   kmin/kmax, dmin/dmax, valid. CPU: `src/cpu/curvature.cpp`, CUDA: `src/cuda/curvature.cu`.
3. **sample_prune** — взвешенная случайная выборка остаётся host; точки с
   неустойчивыми/почти равными кривизнами отбрасываются. CPU: `selection.cpp`,
   CUDA pruning: `backend.cu`. Seed задаёт повторяемый порядок RNG.
4. **pairing** — сравнение подписей кривизны и пространственное отсечение.
   CPU перебирает выбранные точки; GPU сортирует kmin, ищет консервативные
   диапазоны, проверяет кандидатов пакетами и делает count → scan → fill.
   Сортировка принятых target ordinals восстанавливает CPU порядок.
5. **transformation** — допустимые локальные frames пары дают rigid голоса,
   а плоскость через середину пары — mirror кандидата. Сохраняются полные R/t
   и нормализованный descriptor: 12 компонент для rigid, 4 для reflection.
   CPU: `transformations.cpp`, CUDA: `backend.cu`.
6. **mean_shift** — веса Epanechnikov внутри bandwidth, движение центра до
   сходимости/лимита, support counts, merge modes и назначение голосов.
   Seeds/RNG/merge/winner ties общие на host. GPU выполняет до восьми
   последовательных итераций траектории одним cooperative block; ordered sums
   сохраняют порядок арифметики CPU. Реализация: `src/cpu/clustering.cpp` и
   `src/cuda/clustering.cu`.
7. **verification** — полное T и обратное преобразование, ближайшая точка
   поверхности через BVH, distance/normal фильтры, mirror side и исключение
   неподвижных точек. Связные компоненты смежности формируют участки.
   GPU использует итеративные CSR labels; BVH строится host. CPU:
   `verification.cpp`/`surface.h`, CUDA: `verification.cu`/`geometry.cuh`.

После этапов CLI экспортирует промежуточные данные и результат; viewer читает
только JSON и не запускает вычисления. `--no-patching` позволяет отключить
геометрическую проверку; детали параметров — в CLI `--help`.

## Данные и граница CPU/GPU

`include/symmetry/storage.h` задаёт владельцев и views. Координаты, features и
votes используют SoA; смежность и переменные списки — CSR offsets + indices.
Scalar = double, Index = int32, Offset = uint64. Dense S² candidate matrix и
C×P membership matrix не выделяются. На плотных signatures число проверок
pairing всё равно может быть квадратичным; max_pairs/max_votes ограничивают
результат, а sparse support имеет отдельный budget.

Все аргументы Backend — borrowed **host** views. CUDA методы сами выделяют
RAII buffers, делают transfers и возвращают завершённые host-owned outputs.
Между этапами device residency и overlap ещё нет. CPU-only сборка не зависит
от CUDA toolkit; CUDA backend не означает полностью device-resident приложение.

CPU/GPU проверяются при одинаковых seed/options. IDs и membership сравниваются
точно, числа — с допуском; principal directions учитывают знаковую неоднозначность.
CUDA FMA отключено ради численного контракта. Ordered sums ограничивают
параллелизм: GPU mean shift на небольших demo медленнее CPU. Изолированный
pairing на 10000 выбранных точках ускорился примерно в 8 раз, но ускорения всего
CLI не получено. Подробные таблицы и пределы этих выводов — в отчёте этапа 8.

## С чего начать чтение

1. `RUNNING_RU.md`: собрать и посмотреть reflection/partial viewer.
2. `src/app/main.cpp` и `src/core/pipeline.cpp`: управление и порядок этапов.
3. `include/symmetry/pipeline.h`, `options.h`, `storage.h`: данные и контракты.
4. `src/cpu/curvature.cpp`, `pairing.cpp`, `transformations.cpp`: происхождение голосов.
5. `src/cpu/clustering.cpp`, `verification.cpp`, `surface.h`: modes и подтверждённые участки.
6. Соответствующие CUDA модули и `src/cuda/README.md`: где распараллеливание,
   где общий host контроль и почему нужно сохранять порядок сумм.
7. `tests/compare_cuda.py`, direct CUDA tests и `benchmarks/`: критерии корректности
   и измерения. `tests/compare_reference.py` сравнивает два CLI builds.

## Что удалено при очистке

Удалены 79 tracked файлов: старые монолитные C++/GUI исходники и headers в
корне, MATLAB скрипты (включая `graph/`), `.mat` эксперименты, архив
`SymmetryDetection0412.rar`, legacy CMake/libigl scripts и Windows CMakeFiles.
Текущие исходники, тесты, примеры, инструменты, лицензия и отчёты сохранены.
После удаления выполнены чистые Release сборки CPU и CUDA: прошли все 3 CPU
и 12 CPU/CUDA групп CTest, а также viewer smoke. Старых файлов для сборки
и запуска не требуется.

Старая реализация доступна в Git на commit `60268d1` и более ранних commits:
например, `git show 60268d1:graph/compute_curvature_OLD.m`.

Отчёты этапов 1–7 фиксируют промежуточные реализации; актуальное состояние
определяют этот обзор, RUNNING_RU.md и CUDA_STAGE8_RU.md. Проверка реальных
сканов пока не выполнена из-за DNS; полноценные Compute Sanitizer/Nsight GPU
проверки ограничены WDDM/правами на counters. Поиск масштаба и ускорение всего
pipeline остаются отдельными задачами.
