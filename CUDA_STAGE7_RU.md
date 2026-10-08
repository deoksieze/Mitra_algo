# Этап 7: локальная геометрия на GPU, без MATLAB

CUDA-backend теперь самостоятельно вычисляет нормали граней/вершин, edge-тензоры кривизны, усреднение на вершинах, smoothing и tangent eigenframes. Все вычислительные методы Backend имеют GPU-реализацию. Общая подготовка/проверка mesh, построение CSR/BVH, subdivision, sampling/RNG и детерминированная сборка кластеров/результатов остаются host.

MATLAB не нужен для сборки или запуска ни CPU, ни CUDA пути. Исторические MATLAB/libigl исходники удалены при очистке после этапа 8 и доступны в истории Git. Независимость CPU-reference от них была обеспечена ещё на этапе 1.

## Как устроен src/cuda/curvature.cu

1. Поток на грань: unit cross product даёт face normal.
2. Поток на вершину: ordered CSR-gather incident face normals, затем unit. Порядок совпадает с host topology builder.
3. Поток на ребро: длина. Средняя длина всех рёбер суммируется на device последовательно для воспроизводимости.
4. Поток на внутреннее ребро: подписанный угол atan2(dot(cross(n1,n2),direction), clamp(dot(n1,n2))), затем outer product direction×direction с весом beta*length/mean_length. Boundary edges не вносят tensor.
5. Host строит дополнительный incident-edge CSR в порядке исходных edge IDs. Поток на вершину суммирует edge tensors в этом порядке и делит на число внутренних инцидентных рёбер, без atomic floating-point additions.
6. Smoothing использует два 9-компонентных SoA buffers: собственный tensor + ordered neighbors, деление на degree+1. Число iterations задано options.smoothing, 0 также поддерживается.
7. Поток на вершину ограничивает tensor tangent plane, решает ту же аналитическую 2×2 eigen-задачу, что CPU-reference, и сохраняет normal/dmin/dmax/kmin/kmax/valid.

Исходный обзор предлагал symmetric eigenproblem 3×3. Работающий CPU-reference использует явную tangent-plane формулу, поэтому перенесена именно она: замена solver одновременно с CUDA изменила бы математику. Нормаль остаётся отдельно связанной с tangent frame, kmin<=kmax. Валидность требует конечных кривизн и eigenvalue gap>1e-10; слабые нормали и degree<3 дают нулевой Feature, как на CPU.

GPU storage O(N+F+E+CSR), без квадратичных таблиц. Ремешинг/subdivision оставлен CPU: его перенос не оправдан измерениями на имеющихся моделях. GPU тригонометрия и hypot могут отличаться от host libm в последних разрядах, поэтому features теперь сравниваются численно, principal directions — с учётом знаков. Неиспользуемые направления invalid features не участвуют в сравнении.

## Проверки и запуск

```bash
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
./build-cuda/symmetry_cpu_cli examples/reflection.off \
  --backend cuda --output outputs/my-stage7-run --sample-ratio 1 \
  --pair-threshold 0.08 --verify-threshold 0.01
```

Добавлен cuda_curvature: reflection/rigid/partial/single, smoothing 0/3/7, validity, kmin/kmax, normal, направления с точностью 1e-10 и знаковой неоднозначностью; ортогональность frames, единичные tangent directions, порядок eigenvalues, repeatability и плоская открытая сетка без устойчивых направлений.

Сквозное сравнение восьми CPU/GPU сценариев прошло: pairs и mesh совпадают побайтно, flags/sample/pruned совпадают точно, descriptors/votes/centers/patches совпадают в допуске 2e-10. Максимальная нормированная численная разница в проверенном окружении — около 1.02e-15. Набор расширен до 10 групп CTest.

Скорость и память всего CUDA пути измеряются на этапе 8. Сам факт переноса всех методов на GPU не означает ускорения: serial trajectory control/ordered sums/синхронизация на WSL могут доминировать. CPU backend сохранён без изменений формул.
