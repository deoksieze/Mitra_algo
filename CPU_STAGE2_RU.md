# Этап 2: общее ядро, SoA/CSR и контракт backend

Этап 2 разделяет CPU-reference первого этапа на вычислительные модули, ввод/вывод и приложение. CUDA-backend ещё не реализован, но для него подготовлены типы и тот же интерфейс этапов, который сейчас использует CPU. Математика, параметры CLI, порядок RNG и формат результатов первого этапа сохранены.

## Что теперь где находится

```text
include/symmetry/
  math.h          Vec, Mat и математические функции
  options.h       параметры и их проверка
  storage.h       Scalar/Index/Offset, SoA, CSR, borrowed views
  pipeline.h      Backend, Result, подготовка и запуск конвейера
  io.h            чтение сетки и экспорт
  symmetry.h      общий umbrella-header
src/core/
  math.cpp        численная геометрия
  prepare.cpp     проверка/дробление сетки, topology, нормали, веса
  pipeline.cpp    последовательность этапов, RNG, времена, progress callback
src/cpu/
  backend.cpp     реализация Backend через CPU-этапы
  internal.h      приватные объявления реализации
  curvature.cpp
  selection.cpp
  pairing.cpp
  transformations.cpp
  clustering.cpp
  verification.cpp
src/io/
  mesh.cpp        OFF/OBJ
  export.cpp      JSON, CSV, OFF
src/app/
  main.cpp        CLI
src/cuda/
  README.md       требования к будущему backend, kernels пока отсутствуют
benchmarks/
  run_cpu.py      медианы времени CPU-этапов
```

Старые монолитные файлы этапа 1 `src/cpu/symmetry.cpp`, `symmetry.h` и `main.cpp` заменены этими модулями. До изменения они вместе с бинарником сохранены в `/tmp/symmetry-stage2-baseline` для проверки совместимости; это временная локальная копия, не часть Git. Исторические C++/MATLAB-файлы в корне удалены при очистке после этапа 8; доступны в истории Git.

CMake создаёт отдельные библиотеки:

- `Symmetry::core`: общая математика, подготовка и управление конвейером;
- `Symmetry::cpu`: CPU-реализации этапов;
- `Symmetry::io`: чтение/экспорт;
- `symmetry_cpu_cli`: приложение, связывающее CPU и I/O.

Core и CPU не включают файловый ввод/вывод, MATLAB, viewer или печать в консоль. CLI подписывается на progress callback и показывает времена. Само ядро по умолчанию работает без сообщений. Viewer читает только сохранённый JSON; расчёт и визуализация не связаны через глобальное состояние.

## Как устроены данные

`Scalar = double`, `Index = int32_t`, `Offset = uint64_t`. Индексы объектов неотрицательные; -1 используется только для отсутствующего индекса. Смещения и размеры представлений 64-битные. Нельзя создавать mesh/pair/vote IDs выше INT32_MAX: подготовка сетки и проверка бюджетов отвергают такой диапазон. Смена precision на float пока не выполнялась.

**SoA** — каждый компонент хранится в отдельном непрерывном массиве:

- координаты: `Points.x`, `.y`, `.z`;
- треугольники: три массива индексов `Faces.component[0..2]`;
- признаки: отдельные x/y/z нормалей и главных направлений, kmin/kmax, байтовая valid-маска;
- пары: отдельные массивы a/b;
- голоса: pair IDs, явный `TransformKind`, девять компонентов R, координаты t/plane normal, d и двенадцать компонентов descriptor.

У descriptor фиксированная ёмкость 12, без отдельного heap allocation на каждый голос. У отражения используются первые 4 компонента, хвост нулевой; у жёсткого преобразования все 12. Порядок valid-компонентов и арифметика mean shift сохранены. Матрица R представляется по строкам: component 0 — R00, 1 — R01, ..., 8 — R22.

**CSR** — общий массив индексов и массив offsets; строка i занимает `[offsets[i], offsets[i+1])`. Пустые строки допустимы и имеют одинаковые соседние offsets. В CSR хранятся:

- vertex → neighboring vertices;
- vertex → incident faces (новый явный массив смежности);
- cluster → vote IDs;
- cluster → проверенные вершины стороны A/B;
- patch → вершины, с отдельными group offsets для cluster → диапазон patches.

Внутри host-подготовки и отдельных CPU-алгоритмов временно используются map/обычные списки для сортировки, обхода и накопления. Они не входят в контракт будущих kernels. Каноническое хранилище на границах этапов — SoA/CSR. Центры кластеров — небольшой массив `ClusterRecord`; таблица рёбер — отсортированный массив `Edge` с endpoints и инцидентными гранями. Эти небольшие записи пока AoS, а массовые координаты/признаки/голоса — SoA.

## Владение памятью и views

Владеющие структуры (`Mesh`, `Features`, `Pairs`, `Votes`, `Clusters`, `Topology`) содержат обычные STL-векторы. Методы `as_view()` возвращают read-only представления без копирования массивов. Представления содержат указатели и размеры, проверены как trivially copyable. `PointsView`, `FeaturesView`, `VotesView`, `CsrView` позволяют явно передавать backend только необходимые данные.

Правила:

1. Все передаваемые через нынешний Backend views указывают на **host memory**.
2. View не владеет данными и не продлевает их жизнь. После resize, move или уничтожения владельца нужно получить новый view. В частности, изменение одного элемента без realloc видно через существующий view.
3. При построении views проверяются размеры компонентов SoA; у CSR проверяются нулевой первый offset, монотонность и длина indices. Выход verification проверяется по числу кластеров и patch groups.
4. `operator[]` представлений не проверяет каждую границу: вызывающий код должен передавать валидные индексы. `as_view()` проверяет структуру, а не все математические инварианты.
5. Backend должен завершить работу с borrowed views до возврата. Будущий CUDA-backend сам владеет device buffers, явно выполняет transfers и возвращает завершённые **host-owned** результаты. Указатель host view нельзя считать device pointer.

В конвейере владельцы не resize-ятся после получения views для следующего этапа. Topology живёт до завершения расчёта, Result владеет сохранёнными промежуточными данными. Владельцы и Result допускают move; сырые views автоматически за move не следуют.

## Один интерфейс этапов

`Backend` в `include/symmetry/pipeline.h` задаёт:

| Метод | Входы | Host-owned выход |
|---|---|---|
| `curvature` | MeshView, TopologyView, options | Features (SoA) |
| `select` | MeshView, FeaturesView, area weights, options, RNG | Selection: sample/pruned IDs |
| `pairing` | MeshView, FeaturesView, pruned IDs, model context, options | Pairs (SoA) |
| `transformations` | MeshView, FeaturesView, PairsView, context, options | Votes (SoA) |
| `clustering` | VotesView, context, options, RNG | Clusters, CSR membership и диагностика итераций |
| `verification` | MeshView, TopologyView, ClustersView, context, options | Verification: CSR стороны/patches и метрики |

ModelContext содержит boxSize и среднее координат подготовленной сетки. RNG остаётся на host и передаётся по ссылке: сначала sampling, потом mean shift. Порядок случайных вызовов не изменился.

Общая подготовка сетки пока на CPU, затем `detect` вызывает выбранный backend. Добавление частичного CUDA-backend на следующем этапе возможно через делегирование ещё не перенесённых методов CPU. Для этого не понадобится переписывать CLI или экспорт.

Пример использования вычислений из другого C++-приложения:

```cpp
#include <symmetry/io.h>

auto backend = symmetry::make_cpu_backend();
symmetry::Options options;
auto result = symmetry::detect(
    symmetry::read_mesh("examples/reflection.off"), options, *backend);
symmetry::export_result(result, "outputs/new-run");
```

Подключить `Symmetry::cpu` и `Symmetry::io` через CMake. Для вычислений с уже созданной Mesh достаточно `pipeline.h` и `Symmetry::cpu`. Методы чтения и экспорта не входят в Backend. Старый overload `detect(mesh, options)` также сохранён как выбор CPU по умолчанию.

## Запуск и измерения

Команды этапа 1 остаются действительными:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure
./build/symmetry_cpu_cli examples/reflection.off \
  --output outputs/stage2-run --sample-ratio 1 \
  --pair-threshold 0.08 --verify-threshold 0.01
python3 tools/render_result.py outputs/stage2-run
```

Каталог результата должен быть новым или пустым. Существующие HTML-страницы первого этапа продолжают работать, schema_version и поля JSON/CSV не менялись.

Измерение медиан (один warmup + указанное число повторов):

```bash
python3 benchmarks/run_cpu.py examples/reflection.off --repeat 3
```

Скрипт запускает CLI во временных каталогах, читает времена этапов и удаляет свои временные результаты. Это вспомогательное CPU-измерение; времена не включают парсинг/экспорт и не заменяют будущие GPU events или большой benchmark.

## Проверки совместимости

Чистая Release-сборка проходит три группы CTest. Полный набор также прошёл в отдельной Debug-сборке с AddressSanitizer/UBSan: ошибок памяти и undefined behavior на проверенных сценариях не обнаружено. После добавления ограничения диапазонов повторно проверены геометрия и storage/backend в sanitizer-сборке.

Группы CTest:

- геометрия;
- storage/backend contract;
- сквозные геометрические сценарии первого этапа.

Контрактная проверка отдельно проверяет aliasing без копирования, move владельцев, рассогласование SoA-компонентов, пустые CSR-строки и вершину 0, 64-битные offsets, корректность incident faces, CSR patches и порядок вызовов backend. Delegating backend показывает, что общий pipeline реально использует методы интерфейса. Некорректные Features отвергаются до следующего этапа.

Для сравнения с сохранённым бинарником этапа 1:

```bash
python3 tests/compare_reference.py \
  /tmp/symmetry-stage2-baseline/symmetry_cpu_cli build/symmetry_cpu_cli
```

Проверены reflection, rigid, partial и single при sample-ratio 1 и 0.5 — восемь сценариев. Результаты JSON совпадают полностью без timings_ms; features/pairs/votes/descriptors CSV и mesh.off совпадают побайтно. Это проверка при том же окружении и компиляторе; между разными архитектурами допустимы численные различия, описанные в этапе 1.

## Переход к этапу 3

Этап 3 уже реализован отдельно; текущее состояние и команды запуска CUDA см. в [CUDA_STAGE3_RU.md](CUDA_STAGE3_RU.md). Ниже сохранено описание состояния на момент завершения этапа 2.

## Что оставалось для этапа 3

Появится первый CUDA-backend с kernels для фильтрации/признаков и вычисления преобразований. На данном этапе нет device allocations, CUDA language в CMake или GPU-вычислений. Pairing, mean shift и проверка по-прежнему CPU; скорость на больших моделях и математические ограничения этапа 1 сохраняются.
