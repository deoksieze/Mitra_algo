# Запуск текущего CPU/CUDA детектора

Рабочий код находится в `src/` и `include/symmetry/`. MATLAB/libigl не нужны.
Режим: поворот + сдвиг и отражение относительно плоскости, масштаб равен 1.
Это исправленный reference по общей схеме Митры, а не точная копия старой MATLAB-ветки.

## Сборка

```bash
cd /home/ruslan/projects/SymmetryDetection
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
ctest --test-dir build --output-on-failure

# RTX 4070: architecture 89; для другой карты указать её архитектуру
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSYMMETRY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc
cmake --build build-cuda -j 2
ctest --test-dir build-cuda --output-on-failure
```

CUDA сборке нужны toolkit и CUB/CCCL headers, запуску — доступная NVIDIA GPU.
Имя `symmetry_cpu_cli` сохранено для обоих backends; по умолчанию выбран CPU.

## Попробовать модель

```bash
./build-cuda/symmetry_cpu_cli examples/reflection.off \
  --backend cuda --output outputs/my-reflection \
  --sample-ratio 1 --pair-threshold .08 --verify-threshold .01
python3 tools/render_result.py outputs/my-reflection
```

Открыть `outputs/my-reflection/viewer.html` в браузере. Для CPU заменить backend на
`cpu`; можно использовать и бинарник из `build/`. Также доступны `rigid.off`,
`partial.off` и `single.off` в `examples/`; CLI читает OFF/OBJ.
Выбирать новый каталог output для каждого опыта. Полный список параметров:
`./build-cuda/symmetry_cpu_cli --help`.

`result.json` содержит гипотезы, участки, coverage/RMS и времена этапов;
`features.csv`, `pairs.csv`, `votes.csv`, `descriptors.csv` — промежуточные данные.
Кластер без подтверждённых участков ещё не означает найденную симметрию.
На небольших моделях удобнее CPU: итоговые измерения и ограничения приведены
в [CUDA_STAGE8_RU.md](CUDA_STAGE8_RU.md).

Для удалённого VS Code можно обслуживать viewer локальным сервером:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory outputs
```

Перенаправить порт 8765 через Ports и открыть `/my-reflection/viewer.html`.
