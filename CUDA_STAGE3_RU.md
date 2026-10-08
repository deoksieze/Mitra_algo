# Этап 3: первый работающий CUDA-backend

Реализован гибридный backend: GPU выполняет фильтрацию выбранных вершин и построение голосов жёстких преобразований/отражений. Его можно запустить на этом компьютере. CPU-reference остаётся доступным через тот же CLI, формат результатов и viewer сохранены.

Этапы 1–2 зафиксированы исходным коммитом `ba1f7e4`; реализация этапа 3 фиксируется отдельным следующим коммитом. Изменения следующих этапов также следует коммитить отдельно.

## Какие части работают на GPU

| Стадия | Где выполняется |
|---|---|
| Чтение, проверка сетки, subdivision, topology, нормали | CPU |
| Тензоры кривизны, smoothing, главные направления и kmin/kmax | CPU |
| Случайная выборка с весами площадей | CPU, общий sampling для обоих backend |
| Проверка valid, ненулевой кривизны и анизотропии; уплотнение retained IDs | **GPU** |
| Pairing по кривизнам и расстояниям | CPU |
| Зеркальная плоскость, проверка совместимости frames, два rigid-голоса, descriptors | **GPU** |
| Mean shift, геометрическая проверка и patches | CPU |
| Экспорт и viewer | CPU |

Первоначальную формулировку «признаки и преобразования» на этом проходе конкретизируем как обработку уже вычисленных локальных признаков: нормали и кривизны поступают из CPU-reference. Отдельного GPU-расчёта нормалей/тензоров или signature-индекса пока нет. Это позволяет сравнивать фильтрацию и преобразования на одинаковых frames, без смешения ошибок собственных векторов и CUDA. Pairing переносится на этапе 4.

## Сборка и запуск

Проверенное окружение: NVIDIA GeForce RTX 4070 Laptop GPU, CUDA Toolkit 13.3.73, GCC 13.3, CMake 3.28.3. Сборка проекта требует CMake >= 3.18. Архитектура 89 в командах соответствует этому GPU; на другой карте нужно указать её архитектуру. Код CUB проверен с установленным toolkit 13.3; совместимость со всеми более старыми toolkit не заявляется.

```bash
cd /home/ruslan/projects/SymmetryDetection
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSYMMETRY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure

./build-cuda/symmetry_cpu_cli examples/reflection.off \
  --backend cuda --output outputs/my-cuda-run --sample-ratio 1 \
  --pair-threshold 0.08 --verify-threshold 0.01
python3 tools/render_result.py outputs/my-cuda-run
```

Каталог результатов должен быть новым или пустым. Для сравнения заменить `--backend cuda` на `--backend cpu` и указать другой каталог. По умолчанию backend — CPU. Имя бинарника сохранено для совместимости с этапами 1–2.

Уже создан результат [outputs/cuda-stage3/viewer.html](outputs/cuda-stage3/viewer.html): 132 вершины, 1898 пар, 4452 голоса; ведущий кластер — зеркальная симметрия с coverage=1 и RMS примерно 3.37e-6. HTML можно открыть в браузере; вычисления при этом уже завершены. Это локальный сгенерированный файл, каталог outputs исключён из Git.

CPU без CUDA-зависимостей:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYMMETRY_ENABLE_CUDA=OFF
cmake --build build -j 2
ctest --test-dir build --output-on-failure
```

В CPU-сборке запрос CUDA сообщает, как пересобрать проект. В CUDA-сборке отсутствие доступного устройства или ошибка runtime приводят к явной ошибке, автоматической подмены GPU-этапов CPU-реализацией нет.

## Устройство реализации

- `include/symmetry/cuda.h`: фабрика `make_cuda_backend()`, подключение через `Symmetry::cuda`.
- `src/cuda/backend.cu`: kernels, собственные device POD-векторы/матрицы, RAII для device memory, реализация Backend с делегированием четырёх методов CPU.
- `include/symmetry/sampling.h`, `src/core/sampling.cpp`: общая host-выборка с прежним порядком вызовов RNG.
- `src/app/main.cpp`: `--backend cpu|cuda`.
- `tests/cuda_backend.cpp`: прямое сравнение контрактов GPU/CPU и геометрических инвариантов.
- `tests/compare_cuda.py`: сравнение экспортированных промежуточных данных и итогов.

Все Backend views по-прежнему указывают на host memory. CUDA выделяет отдельные buffers, копирует входные компоненты, запускает kernels, возвращает завершённые host-owned результаты. RAII освобождает память и при исключениях. Между методами нет скрытого сохранённого device state: следующий этап пока снова работает с host-массивами.

Фильтрация запускает поток на sampled vertex. Маска повторяет CPU-условие:

```text
valid && max(abs(kmin), abs(kmax)) > 1e-10
      && abs(kmin-kmax)/max(abs(kmin),abs(kmax)) > 1-prune
```

`cub::DeviceSelect::Flagged` уплотняет ID с сохранением исходного порядка. Пустая выборка возвращается без запуска kernel; нулевой и полный выход compaction поддерживаются.

Преобразования используют три прохода:

1. Поток на пару проверяет mirror-совместимость и записывает число голосов: 2 или 3.
2. `cub::DeviceScan::ExclusiveSum` считает 64-битные смещения. По последнему offset проверяется бюджет **до** выделения массивов голосов; результат не обрезается.
3. Поток на пару заполняет свой диапазон: необязательное отражение, rigid со знаком +1, rigid со знаком -1. Порядок совпадает с CPU независимо от порядка исполнения потоков.

Device-вход — SoA координат, нормалей, двух главных направлений и индексов пар. Выход — SoA 28 скалярных компонент на голос (R:9, t:3, n:3, d:1, descriptor:12), отдельно pair IDs и явный байтовый TransformKind. Host views и STL-контейнеры в kernels не разыменовываются.

Double precision и порядок сумм сохранены; CUDA компилируется с `--fmad=false`, без fast math. Знак зеркальной плоскости выбирается по доминирующей компоненте нормали, как на CPU. Нулевое направление обрабатывается тем же порогом 1e-14 и даёт нулевой вектор. Нулевые направления frame не «исправляются» новым алгоритмом: прямой вызов воспроизводит CPU-reference; в рабочем конвейере невалидные features отбрасываются до pairing. Противоположные нормали не требуют acos или clamp: используются матрицы и dot products. Масштаб остаётся фиксированным 1, scale-симметрии не добавлялись. Непустой CUDA-вызов преобразований с нулевым/нечисловым box size, нечисловым center или неверными pair indices отвергается явно.

Примитивы CUB используются через API temp storage установленного toolkit: [DeviceScan](https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html), [DeviceSelect](https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_select.cuh).

## Проверки и точность

На реальном GPU прошли все **6 групп CTest**: geometry, storage_backend, pipeline (CPU), cuda_backend, compare_cuda, pipeline_cuda. Отдельная CPU-сборка проходит прежние 3 группы.

Прямые CUDA-тесты проверяют:

- одинаковые sample/pruned ID и состояние RNG;
- строгую границу pruning, знаки кривизны, пустую и полную compaction;
- 1028 пар, несколько блоков, смешанные количества 2/3 и устойчивый порядок голосов;
- знак произвольной зеркальной плоскости, противоположные/параллельные нормали, неоднозначность знака главных направлений;
- совпадение матриц, переводов и descriptors с допуском 2e-12*(1+abs(reference));
- ортогональность R, det=+1/-1, отображение p в q, норму плоскости и нулевой хвост descriptor отражения;
- воспроизводимость повторного GPU-вызова;
- пустые mesh/pairs, нулевые направления, неверные индексы/context;
- бюджет ниже требуемого и бюджет ровно по размеру выхода.

Сквозные проверки:

```bash
python3 tests/compare_cuda.py build-cuda/symmetry_cpu_cli
python3 tests/compare_reference.py \
  /tmp/symmetry-stage3-baseline/symmetry_cpu_cli build/symmetry_cpu_cli
```

Для reflection, rigid, partial и single при sample-ratio 1 и 0.5 — 8 сравнений. Features, pairs и mesh совпали побайтно. Типы/порядок голосов, descriptors и JSON без timings совпали; максимальная нормированная численная разница CPU/GPU в этих fixtures равна 0, допустимый порог теста — 2e-10. На другом compiler/GPU побайтное совпадение floating point не гарантируется. CPU-режим отдельно побайтно совпал с сохранённым бинарником этапа 2 по всем экспортам без timings.

**Ограничение проверки памяти:** Compute Sanitizer memcheck и initcheck были запущены, но оба завершились ошибкой инициализации WDDM debugger interface: `Failed to initialize WDDM debugger interface` и `Device not supported`. Сам тест CUDA прошёл, однако инструмент не смог выполнить проверку GPU-памяти. Поэтому sanitizer-проверка остаётся невыполненной до настройки host debugger interface или запуска в поддерживаемом окружении. Это не следует считать успешным memcheck.

## Время работы и следующий этап

Медианы трёх повторов после одного прогрева для examples/reflection.off, sample-ratio=1, прочие параметры по умолчанию; времена host wall clock включают allocation/H2D/D2H внутри метода и исключают чтение/экспорт и создание backend. Каждый повтор benchmark запускает отдельный процесс: прогрев не сохраняет CUDA context между процессами.

| Стадия | CPU, ms | CUDA backend, ms |
|---|---:|---:|
| sample_prune | 0.012 | 2.327 |
| transformation | 2.226 | 4.905 |
| mean_shift (CPU в обоих случаях) | 963.470 | 975.547 |

Это маленькая модель: 132 вершины, 4470 пар и 10356 голосов при параметрах benchmark. GPU здесь не ускоряет конвейер: выделения памяти и передачи дороже коротких kernels. Эти времена не являются чистым kernel benchmark и не доказывают ускорения на больших моделях. CUDA events, удержание промежуточных массивов на device и профилирование остаются дальнейшей работой.

```bash
python3 benchmarks/run_cpu.py examples/reflection.off \
  --binary build-cuda/symmetry_cpu_cli --backend cuda --repeat 3 --sample-ratio 1
```

Следующий этап 4 — pairing на GPU: индекс сигнатур/поиск диапазонов кандидатов, count → scan → fill, ограничение памяти и сравнение порядка/содержимого пар с CPU-reference. Затем этап 5 переносит mean shift, который сейчас занимает основную долю времени. К этим этапам изменения в этом коммите не относятся.
