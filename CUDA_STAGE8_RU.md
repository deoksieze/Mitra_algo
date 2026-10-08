# Этап 8: итоговые проверки и измерения CPU/CUDA

Измерено 8 октября 2026 года на RTX 4070 Laptop GPU (8188 MiB), WSL/WDDM,
драйвер 610.91, CUDA 13.3.73, GCC 13.3, CMake 3.28.3, architecture 89.
Release, double, CUDA FMA отключено. Исходные результаты и параметры сохранены
в [benchmarks/results](benchmarks/results). Команды запуска — [RUNNING_RU.md](RUNNING_RU.md).

Все вычислительные методы Backend имеют CUDA-реализацию; host сохраняет
подготовку mesh/topology/BVH, subdivision, sampling/RNG, seeds/merge и сборку
результатов. CPU остаётся backend по умолчанию. Реализованный режим — rigid
поворот/сдвиг и чистое отражение в плоскости, масштаб 1. Это исправленный
reference по общей схеме исходного проекта; совпадение с историческим MATLAB
кодом и полная реализация всех режимов статьи не утверждаются.

**Ускорение всего CLI на имеющихся небольших моделях не получено.**
Ускорение подтверждено отдельно для индексированного pairing. Mean shift
остаётся основным узким местом; наличие GPU kernels само по себе не делает
детектор быстрее. Проверка реальных сканов и полноценные NVIDIA GPU
sanitizer/profiler проверки остаются открытыми из-за ограничений окружения.

## Проверки корректности

Прошли все **12 групп CTest** в CUDA сборке: geometry, storage_backend,
pipeline, cuda_backend, cuda_pairing, cuda_clustering, cuda_verification,
cuda_curvature, cuda_profile, extended_cuda, compare_cuda, pipeline_cuda.
После изменения драйвера траекторий CPU экспорт сравнен с сохранённым
reference на восьми сценариях: совпадение без timings точное.
Отдельная CPU Debug сборка с AddressSanitizer + UndefinedBehaviorSanitizer
прошла все три CPU группы после чистой пересборки.

Сквозное сравнение CPU/GPU: reflection, rigid (поворот + сдвиг), partial и single,
каждый с sample-ratio=1 и .5. Mesh и pairs сравниваются побайтно; flags,
sample/pruned, IDs, memberships и patches — точно. Features и направления
сравниваются с учётом знаковой неоднозначности, неиспользуемые invalid directions
пропускаются. Допуск сквозного численного сравнения — 2e-10*(1+abs(reference)),
локальных геометрических проверок — 1e-10*(1+abs(reference)). Это допуски
численного сравнения, отдельно от геометрического verify-threshold в долях
boxSize. Surface-query тесты проверяют расстояния непосредственно.

Добавленный extended_cuda проверяет регулярный октаэдр (сфероподобный случай
без устойчивых principal directions, все точки отсеяны), level-1 зеркальные
копии, небольшой шум, масштабированную копию и один асимметричный компонент.
Допуск extended сравнения — 1e-9*(1+abs(reference)). Масштабированная копия
проверяет согласованное поведение backends в режиме scale=1; поиск масштаба
этим тестом не подтверждается. Плоская открытая сетка проверяется в curvature.
Пустой и вырожденный OFF отвергаются до создания output. Другие тесты покрывают
пустые пары/голоса, произвольную плоскость отражения, forward/inverse rigid,
расстояния до вырожденных треугольников, повторяемость, лимиты и CSR компоненты.
При CUDA_VISIBLE_DEVICES='' CLI возвращает понятную ошибку отсутствия устройства.

cuda_profile проверяет точные размеры копий на маленьком fixture, число
custom launches, освобождение отслеживаемых buffers, сохранение результата/RNG
и отсутствие записей при выключенном профилировании.

Готовый итоговый пример: `outputs/cuda-final/viewer.html`. У первого зеркального
кластера reflection.off coverage=1, RMS≈3.37e-6 в координатах модели. Это
подтверждённый синтетический ответ; остальные кластеры включают дополнительные
гипотезы, которые нужно оценивать по участкам/остатку.

## Время всего CLI

Для каждого backend/model: один разогрев и три измеренных запуска, медиана.
Запуски последовательные, без параллельных тестов. Каждый запуск — новый процесс,
поэтому wall time включает CUDA context startup, загрузку, экспорт и завершение.
Времена этапов измеряет steady_clock; инструментированное профилирование
выключено. Параметры одинаковы: sample-ratio=1, pair-threshold=.08,
verify-threshold=.01, seed=42, остальные defaults (включая bandwidth=.08).

| Модель | N/F | S | пары/голоса/кластеры | CPU CLI, ms | CUDA CLI, ms | CUDA/CPU |
|---|---:|---:|---:|---:|---:|---:|
| reflection | 132/256 | 132 | 1898/4452/20 | 323.8 | 13956.0 | 43.1× |
| rigid | 132/256 | 132 | 1898/4328/20 | 223.2 | 6678.1 | 29.9× |
| partial | 198/384 | 198 | 2378/5630/20 | 360.8 | 8564.4 | 23.7× |
| single | 66/128 | 66 | 458/1122/20 | 22.3 | 930.1 | 41.7× |

N/F — vertices/faces, S — pruned points. Наборы результатов одинаковы по
контракту сравнения выше. Raw runs и медианы каждого этапа содержатся в
`rtx4070_wsl_cli.json`. Медианы этапов не обязаны суммироваться в медиану CLI.
На reflection медиана mean shift: CPU 272.93 ms, CUDA 13114.95 ms; verification:
3.89 ms и 200.83 ms. Между запусками CUDA времена заметно варьируются; например,
отдельный последующий viewer-run mean shift занял 5429.85 ms. Эти числа
характеризуют данное WSL окружение и малые задачи, а не устойчивый предел GPU.

Отдельный pairing benchmark: S=10000, 100 групп признаков, P=495000;
точные пары/порядок проверены, один разогрев и медиана трёх повторов:
CPU **93.54 ms**, CUDA **11.47 ms**, примерно **8.15×**. Здесь CUDA context
уже создан; время включает вызов backend и передачи. Это структурированный
синтетический input, его результат нельзя переносить на весь CLI или сканы.

## Телеметрия и память

`make_cuda_backend(true)` включает ProfileStage; `take_cuda_metrics()` возвращает
метрики текущего host thread. Обычный backend остаётся без профилирования.
`benchmarks/profile.cpp` запускает инструментированный pipeline отдельно.

- H2D/D2H bytes — явные cudaMemcpy внутри методов, включая частичные scalar/state
  downloads. Host-memory traffic не учитывается.
- Peak buffer bytes — максимальная сумма одновременно живых явных device Buffer
  данного этапа, включая переданный CUB scratch. CUDA context, events, драйверные
  caches и host allocations не учитываются; это не полное потребление VRAM.
- Custom launches — только наши kernels; внутренние CUB kernels не считаются.
- GPU interval — CUDA events от начала до конца метода, включая ожидания host,
  allocation, CUB, копии и промежутки подачи работы.
- Custom kernel ms — сумма event intervals вокруг наших launches. Они могут
  включать промежутки подачи работы; это не чистое hardware execution time.
  Copy ms имеет такую же event семантику. Профилирование добавляет синхронизацию,
  поэтому его времена не подменяют обычные benchmarks.

Reflection N=132, F=256, S=132, пары=1898, P=4452, C=20, limited_iterations=0:

| Этап | H2D bytes | D2H bytes | peak bytes | custom launches | GPU interval ms | custom intervals ms |
|---|---:|---:|---:|---:|---:|---:|
| curvature | 27864 | 11748 | 98660 | 10 | 10.35 | 0.82 |
| sample_prune | 2772 | 536 | 4463 | 1 | 4.24 | 0.62 |
| pairing | 5808 | 15192 | 50463 | 5 | 2.34 | 0.10 |
| transformation | 27856 | 1019516 | 1079027 | 2 | 5.59 | 0.09 |
| mean_shift | 849268 | 533356 | 575451 | 7683 | 8125.08 | 3713.86 |
| verification | 23880 | 48672 | 21112 | 348 | 118.09 | 16.23 |

Все отслеживаемые buffers освобождены к концу методов. Максимальный stage peak
≈1.03 MiB (transformation), а не сумма пиков всех этапов. Методы синхронны и
возвращают host-owned outputs; большинство массивов затем снова загружается
следующим методом. Device residency/overlap между этапами пока не реализованы.

## Изменение mean shift по результатам профиля

Исходный evaluator этапа 5 запускал distance + ordered sums + compaction на
каждой итерации и возвращал neighborhood на host. Для reflection это давало
13348 custom launches. На этапе 8 введён общий CPU/GPU trajectory evaluator:
GPU хранит центр и support counts на device, делает до восьми последовательных
итераций одним cooperative block, возвращает state после batch и компактный
support только после траектории. Seeds, RNG, visited, merge и winner ties
остались host и сохранили порядок. CPU формулы и экспорт не изменились.

Теперь для той же задачи 7683 custom launches (примерно на 42% меньше).
Сокращение числа запусков подтверждено, ускорение этого изменения отдельно
контролируемым A/B benchmark не заявляется. Многие траектории короткие,
state transfers остаются, ordered sums ограничивают параллелизм одним block.
Это компромисс ради воспроизводимости: parallel tree reduction меняла последние
биты и выбор ближайшей грани в равных геометрических случаях (см. этап 5).

Рост P на отдельном синтетическом случае с четырьмя смешанными modes:
один разогрев, медиана трёх обычных вызовов; профиль отдельно. Memberships
и CSR offsets проверяются на каждом повторе. В этих случаях восемь custom
launches соответствуют четырём коротким траекториям и сборке support.

| P | CPU ms | CUDA ms | peak bytes | H2D/D2H bytes | custom launches |
|---|---:|---:|---:|---:|---:|
| 1024 | 0.121 | 5.459 | 133239 | 99776/12768 | 8 |
| 4096 | 0.381 | 10.383 | 529527 | 397760/49632 | 8 |
| 16384 | 2.033 | 32.670 | 2114679 | 1589696/197088 | 8 |

GPU workspace растёт линейно с P, без плотной C×P матрицы. Sparse host support
сохраняет общий лимит 5 млн entries; расходы host map в peak не входят.
Этот лёгкий benchmark не моделирует тысячи seeds из полных demo моделей.

## Реальные данные и NVIDIA инструменты: что осталось непроверенным

Добавлен `tools/fetch_bunny.py`: загрузка coarse ASCII PLY `bun_zipper_res4.ply`
из [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/),
конвертация в OFF и запись URL/SHA256. Попытка загрузки в этом окружении
завершилась `Temporary failure in name resolution`. **Реальный скан не запускался**;
результаты всех таблиц относятся к синтетике. После восстановления DNS:

```bash
python3 tools/fetch_bunny.py outputs/stanford/bunny.off
./build-cuda/symmetry_cpu_cli outputs/stanford/bunny.off --backend cpu \
  --output outputs/bunny-cpu --sample-ratio .1
./build-cuda/symmetry_cpu_cli outputs/stanford/bunny.off --backend cuda \
  --output outputs/bunny-cuda --sample-ratio .1
```

Параметры для качества на скане ещё не подобраны; отсутствие гипотез при
defaults не означает отсутствие симметрии. Source model не добавлялся в Git.

Compute Sanitizer memcheck на итоговом clustering test сообщил две ошибки
инициализации: WDDM debugger interface недоступен / Device not supported.
Программа прошла свой функциональный тест, но **проверка GPU памяти не выполнена**;
даже exit code 0 инструмента в этом случае не означает успешный memcheck.
Nsight Compute завершился ERR_NVGPUCTRPERM (нет прав на GPU counters).
Nsight Systems создал `.nsys-rep`/SQLite, однако в них доступны только CUDA API
records, без GPU kernel/memory records. API CSV сохранён; raw reports находятся
в `/tmp/symmetry-stage8-pairing.*` и могут исчезнуть. API trace показывает расходы
cudaFree/cudaMemcpy/cudaMalloc и launches; по нему нельзя делать вывод об occupancy
или пропускной способности kernels. Логи ограничений сохранены вместе с результатами.

Нужен повтор memcheck/initcheck, Nsight GPU trace и проверка реальных сканов в
окружении с доступными драйверными интерфейсами/сетью. Для ускорения следующими
кандидатами являются device residency, сокращение коротких trajectory transfers,
согласованный численный контракт для parallel reductions и измеренная BVH/CSR
оптимизация. Float/mixed precision, GPU subdivision и независимые seeds не
включались: они требуют отдельной проверки качества и семантики.

## Воспроизвести измерения

```bash
python3 benchmarks/run_suite.py --binary build-cuda/symmetry_cpu_cli \
  --repeat 3 --output outputs/benchmarks/cli.json
./build-cuda/symmetry_pairing_benchmark
./build-cuda/symmetry_clustering_benchmark
./build-cuda/symmetry_cuda_profile examples/reflection.off
```

Не запускать benchmarks одновременно друг с другом или CTest. JSON/profile
сохраняют конкретный input/параметры; hardware metadata — `environment.json`.
