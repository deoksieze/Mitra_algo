# SymmetryDetection

Самостоятельный C++17 проект поиска частичных симметрий треугольных поверхностей:
CPU-reference и optional CUDA-backend, CLI, JSON/CSV экспорт и HTML-viewer.
Поддерживает rigid поворот/сдвиг и отражение в плоскости, масштаб 1.
MATLAB и libigl не нужны.

- [Сборка и запуск](RUNNING_RU.md)
- [Устройство текущего кода](CODEBASE_OVERVIEW_RU.md)
- [Проверки, измерения и ограничения](CUDA_STAGE8_RU.md)

CPU — backend по умолчанию. На маленьких demo весь CUDA pipeline медленнее CPU;
изолированный GPU pairing даёт ускорение. Реальные сканы и полноценные NVIDIA
sanitizer/profiler проверки пока не подтверждены в текущем окружении.

Лицензия: [MPL-2.0](LICENSE). Старые C++/MATLAB/GUI файлы удалены из рабочего
дерева после этапа 8 и доступны в истории Git.
