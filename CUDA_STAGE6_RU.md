# Этап 6: GPU surface verification и участки

Полная проверка T(p) и T^-1(p), расстояния до всей поверхности, фильтр нормалей и сторон зеркальной плоскости теперь выполняются на GPU. Сохранена семантика CPU-reference: global nearest-surface BVH, не локальная окрестность фиксированной target vertex.

## Реализация

- `src/cpu/surface.h`: общий детерминированный host BVH builder, выделенный без изменения CPU search. Балансировка median split, листья до восьми граней.
- `src/cuda/geometry.cuh`: device point→triangle distance с теми же region tests и fallback на три отрезка при вырожденном треугольнике.
- `src/cuda/verification.cu`: копирование дерева на GPU, итеративный stack traversal, полные forward/inverse проверки вершин и CSR-connected components.

Один поток проверяет вершину в одном из двух направлений. Обратное преобразование — transpose(R)*(p-t). Stationary displacement, ограничения стороны плоскости, порог distance² и dot(mapped_normal, face_normal)>=0.5 повторяют CPU. Traversal использует те же bounds, strict comparisons и порядок обхода детей; это сохраняет выбор ближайшей грани при равных расстояниях. Stack 64 entries достаточен для сбалансированного дерева с 32-битными face IDs; рекурсии на GPU нет.

Для patches используется итеративное распространение минимального allowed vertex label по CSR. Два buffers задают синхронные раунды; atomic changed flag сообщает о сходимости. Строки CSR обходятся независимо по вершинам. В отличие от исходного предложения compact frontier, здесь полный masked sweep; он проще для проверки, но имеет стоимость O(N*diameter). Число раундов ограничено N, зависимость от глубины рекурсии отсутствует. Изолированные вершины, вершина 0 и несколько компонент поддерживаются.

Host собирает CSR sides/patches в прежнем порядке, суммирует площади и RMS в CPU-порядке. Это детерминированная сборка результата, а не повтор вычисления расстояний на CPU. BVH construction и topology/remeshing остаются host. GPU memory линейна по N, F и CSR edges; кластеры обрабатываются последовательно с повторным использованием buffers, без dense C×N×F.

## Проверки

`cuda_verification` сравнивает known mirror, rigid motion и partial geometry с CPU, включая оба направления, membership/offsets/groups, coverage/RMS и повторяемость. Identity transformation даёт пустые стороны; vertex 0 проверяется явно. Отдельный kernel test сравнивает 1024 point-triangle queries с CPU, включая triangle→segment→point degeneracy и точку на вершине. Пустой query batch поддерживается.

```bash
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
```

Сквозное сравнение CPU/GPU на восьми сценариях сохраняется: exported pairs/votes, cluster memberships и patches совпадают, численный допуск 2e-10. Итого на этом этапе 9 групп тестов. CPU-only путь и формат экспорта сохранены.

GPU kernels доступны в обычном --backend cuda; no-patching по-прежнему пропускает verification. Подробные времена и профилирование — в этапе 8. Маленькие задачи могут быть медленнее из-за launches и синхронизации label rounds; ускорение этой стадии на произвольной сетке не заявляется. Ограничение Compute Sanitizer/WDDM сохраняется.
