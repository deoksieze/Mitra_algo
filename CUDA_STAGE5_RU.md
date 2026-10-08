# Этап 5: GPU mean shift с прежним порядком кластеризации

**Обновление этапа 8:** GPU теперь удерживает состояние траектории и счётчики support на device, выполняя до восьми последовательных итераций за запуск. Host получает центр по завершении batch и компактный support по завершении траектории. Seeds/merge и ordered sums сохранены. Ниже описана исходная реализация этапа 5; итоговые измерения — [CUDA_STAGE8_RU.md](CUDA_STAGE8_RU.md).

На GPU перенесены расстояния до descriptors, Epanechnikov weights, взвешенные суммы и стабильное уплотнение membership. Голоса загружаются на device один раз на метод. Seeds/RNG, visited, sparse support counters, merge и окончательная обработка центров остаются общим детерминированным host-драйвером.

## Реализация

`src/cuda/clustering.cu` вызывает `cpu::clustering_with` с GPU evaluator вместо CPU neighborhood loop. CPU-backend использует тот же driver с CPU evaluator; базовые условия остановки и порядок случайных вызовов не изменены. `src/cuda/internal.cuh` содержит общие RAII buffers и device-векторы/матрицы, выделенные из backend.cu.

Один GPU-поток на голос вычисляет squared distance в 4D (reflection) или 12D (rigid). Типы не смешиваются. Строгое условие distance < bandwidth², вес 1-distance/bandwidth² и поддержка seed сохраняются. DeviceSelect формирует membership IDs в прежнем порядке.

Суммы компонентов вычисляются на GPU независимо по компонентам, но **в исходном порядке голосов внутри каждой компоненты**. Проверенная параллельная tree reduction давала центры с различием порядка 1e-16, которое меняло выбор ближайшей грани в геометрически равных случаях и отдельные вершины patches. Поэтому итоговая реализация использует ordered accumulation. Это сознательный приоритет воспроизводимости, а не оптимизированный быстрый параллельный reduce. Пакетный запуск независимых seeds, меняющий visited/merge, не добавлялся.

Host получает только суммы и компактный список принятых ID. Данные не хранятся в плотной матрице C×P: GPU buffers O(P), sparse support host с прежним лимитом 5 000 000 entries. Ограничения max_iterations, max_clusters и 20 000 modes сохранены. Пустой набор голосов не запускает kernels и не расходует RNG.

## Проверки

Добавлен `cuda_clustering`: 0, 1, 64 и 2048 голосов, смешанные rigid/reflection modes, шум центров, max_iterations=1/100, одинаковые memberships, центры, состояние RNG, limited_iterations и повторяемость. Сравнение чисел использует 1e-10*(1+abs(reference)); IDs и CSR offsets сравниваются точно.

Сквозные проверки из предыдущих этапов сохранены. Исправлен comparator JSON: численное 1 и 1.0000000000000002 должны сравниваться с допуском, даже если json.loads представил первое как int. IDs и прочие целые значения по-прежнему сравниваются точно. CPU-режим сравнен с сохранённым бинарником этапа 4 на восьми сценариях: совпадение экспортов без timings побайтное.

```bash
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
python3 tests/compare_reference.py \
  /tmp/symmetry-stage5-baseline/symmetry_cpu_cli build-cuda/symmetry_cpu_cli
```

## Производительность и границы

Это рабочая CUDA-реализация, но на маленьких моделях GPU mean shift медленнее CPU. Последовательные траектории требуют многих launches, synchronizations и D2H transfers; ordered accumulation ограничивает параллелизм. Предварительное измерение reflection.off при sample-ratio=1 показало около 11.9 секунд для GPU варианта с tree sums против примерно 1 секунды CPU; оно относится к промежуточному варианту, а не является итоговым benchmark ordered sums. Итоговые замеры выполнены отдельно на этапе 8.

CPU-backend остаётся доступен через --backend cpu. Ускорение mean shift этим этапом не заявляется. Оптимизация численного контракта/редукций и удержание всей trajectory state на device требуют отдельной работы после профилирования. Compute Sanitizer ограничен WDDM debugger interface, как описано в этапах 3–4.

Далее этап 6 переносит surface queries и проверку участков на GPU; этап 7 — кривизны; этап 8 — итоговое качество, времена, память и профилирование.
