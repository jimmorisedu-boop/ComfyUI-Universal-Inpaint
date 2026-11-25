# Установка

Репозиторий содержит custom-node пакет для ComfyUI с нодой интерактивного выделения и трекинга видеомасок (`THSUniversalMaskEditor`). Базовый профиль `editor` работает на CPU (`Static`, `Keyframes`, `Optical Flow`); профиль `sam3` подключается отдельно и не затрагивает установленные в ComfyUI версии Python, PyTorch, CUDA, `torchvision` или `torchaudio`.

## 1. Проверка окружения

Склонируйте репозиторий в `ComfyUI/custom_nodes/`:

```powershell
cd custom_nodes
git clone https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint.git
cd ComfyUI-Universal-Inpaint
```

Проверьте окружение до внесения изменений (`check` и `plan` работают только на чтение):

```powershell
python scripts/setup.py check --comfy-root "<ComfyUI-root>" --json
python scripts/setup.py plan --comfy-root "<ComfyUI-root>" --profile editor --json
```

## 2. Установка базового редактора (`editor`)

```powershell
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile editor --apply
```

Перед установкой выполняется `pip --dry-run --report`: если резолвер предлагает изменить `torch`/CUDA-стек, установка останавливается. После установки перезапустите ComfyUI и откройте `workflows/Universal_Video_Mask_Editor.json`.

## 3. Подключение SAM3 (`sam3`)

```powershell
python scripts/setup.py install --comfy-root "<ComfyUI-root>" --profile sam3 --apply
```

Модель `facebook/sam3` требует явного принятия лицензии на Hugging Face. Примите условия на странице модели, авторизуйтесь (`hf auth login` или `HF_TOKEN`) и выполните:

```powershell
python scripts/setup.py download --comfy-root "<ComfyUI-root>" --profile sam3 --allow-gated --apply
```

Для портативного Windows ComfyUI замените `python` на путь к `python_embeded\python.exe`.
