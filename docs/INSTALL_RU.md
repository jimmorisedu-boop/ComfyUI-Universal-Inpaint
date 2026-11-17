# Universal Inpaint 0.1.1 — переносимая установка

Репозиторий является custom-node пакетом для ComfyUI. Безопасная стартовая
точка — профиль `editor` (CPU, редактор маски). Профили генерации подключаются
отдельно и не заменяют Python, PyTorch, CUDA, torchvision, torchaudio или
xformers, уже принадлежащие установленному ComfyUI.

## Выбор Python и проверка

Выполните из каталога ComfyUI:

```powershell
cd custom_nodes
git clone https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint.git
cd ComfyUI-Universal-Inpaint
```



Укажите каталог с `main.py` (или пакетом `comfy`). Скрипт найдёт рядом
`python_embeded/python.exe` у Windows portable, `.venv` или примет путь через
`--python`. Сканирования дисков и путей конкретного компьютера нет.

```powershell
python scripts/setup.py check --comfy-root "<ComfyUI-root>" --json
python scripts/setup.py plan --comfy-root "<ComfyUI-root>" --profile editor --json
```

`plan` ничего не меняет и используется по умолчанию. Несовпадающий Git remote
или revision, существующий workflow, модель или addon не удаляются и не
перезаписываются — они отмечаются как конфликт.

Старый каталог `custom_nodes/ths_universal_inpaint/` считается дубликатом:
план остановится и потребует ручной миграции с резервной копией. Скрипт его не
удаляет и не запускает одновременно две копии одинаковых node ID.

## Профили и установка

```powershell
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile editor --apply
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile sam3 --apply
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile autoprompt --apply
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile h3 --apply
```

Пакет копируется в `ComfyUI/custom_nodes/ComfyUI-Universal-Inpaint/`.
Перед установкой выполняется `pip --dry-run --report`; если resolver предлагает
изменить torch/CUDA/vision-пакет, операция останавливается. Внешние custom nodes клонируются только на закреплённые ревизии из
`config/dependencies.json`; optional `comfy-env`/pixi-установщик SAM3 не
запускается. H3 использует нативный `comfy_extras.nodes_minimax_h3.py`
совместимого ComfyUI.

## Модели и закрытая лицензия SAM3

Загрузка моделей — отдельная явная операция:

```powershell
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile h3
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile h3 --apply
```

Первая команда только строит план свободного места. Загрузка использует
возобновляемый `.part`-кэш, проверяет размер/SHA-256 и не перезаписывает
существующие файлы.

SAM3 на Hugging Face gated. Сначала самостоятельно примите условия на URL из
сообщения скрипта, затем выполните `hf auth login` или задайте `HF_TOKEN` и
явно разрешите загрузку:

```powershell
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile sam3 --allow-gated --apply
```

Лицензия автоматически не принимается, токены в репозитории не сохраняются.
При отсутствии доступа используйте редактор/Optical Flow или вручную положите
совместимый safetensors checkpoint в путь из плана.

## Ограничения

`check` подтверждает только обнаружение интерпретатора и нативного H3-модуля;
это не визуальная и не GPU-приёмка. Для публичного релиза 0.1.1 чистый
end-to-end H3-рендер не заявляется.

Для Windows portable без команды `python` замените её на
`& "../../../python_embeded/python.exe"` (из каталога репозитория).
Профиль `editor` ставится первым; остальные добавляются по необходимости.
После проверки pip устанавливается весь зафиксированный список зависимостей
с SHA-256, включая транзитивные зависимости. Если для SAM3 нужен другой Triton,
установите совместимую с вашим ComfyUI версию отдельно: скрипт GPU-стек не меняет.
