# ComfyUI Universal Inpaint

[English](README.md) · [Установка](docs/INSTALL_RU.md) · [Все кнопки и настройки](docs/USER_GUIDE_RU.md)

Universal Inpaint — переносимый custom-node репозиторий для ComfyUI. Он
помогает восстановить выбранную пользователем область короткого видео и
сохраняет явный путь исходных пикселей вне маски. Это альфа-версия рабочего
графа, а не обещание поддержки любого видео или окружения.

Репозиторий клонируется напрямую в `ComfyUI/custom_nodes/`. Точкой входа
является корневой `__init__.py`, а браузерная часть обслуживается из
`WEB_DIRECTORY = "./web"`.

## Графы

| Файл | Назначение | Нужен H3? |
| --- | --- | --- |
| `V70_Universal_Repair_Workbench.json` | Основной workbench: источник, маска, промпт, H3, сравнение и экспорт | Да для генерации |
| `V69_Universal_Masked_Repair_MASK_ONLY.json` | Расчёт, просмотр и экспорт маски | Нет |
| `V69_Universal_Masked_Repair.json` | Ручная маска → H3 repair | Да для генерации |
| `V69_Universal_AutoPrompt_Repair.json` | Необязательная локальная подпись контекста | Да для генерации; caption-модель необязательна |
| `V69_Universal_Reference_Repair.json` | Reference/corrected-frame режимы с одним переключателем | Да для генерации |

В редакторе маски режимы трекинга не смешиваются. **SAM3** запускается двумя
явными шагами: сначала расчёт маски на одном кадре, затем трекинг диапазона.
**Optical Flow** — отдельный CPU/legacy путь; ошибка SAM3 не подменяется им
автоматически. Для машины без H3 используйте mask-only граф.

## Установка

Выполните из каталога ComfyUI:

```powershell
cd custom_nodes
git clone https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint.git
cd ComfyUI-Universal-Inpaint
```



```text
ComfyUI/custom_nodes/ComfyUI-Universal-Inpaint/
```

Скопируйте или клонируйте репозиторий в этот каталог. До перезапуска установите
профиль редактора в тот же Python, которым запускается ComfyUI:

```powershell
python scripts/setup.py install --profile editor --comfy-root "<COMFYUI_ROOT>" --apply
```

Затем перезапустите ComfyUI и откройте
`workflows/V69_Universal_Masked_Repair_MASK_ONLY.json`. Выберите видео и
загрузите кадры. Для CPU выберите **«Область · Optical Flow»**; режим SAM3
требует отдельной установки. Для генерации через H3 сначала установите профиль `h3` и его модели.
Подробные инструкции и список моделей:
[docs/INSTALL.md](docs/INSTALL.md), [docs/INSTALL_RU.md](docs/INSTALL_RU.md),
[docs/MODELS.md](docs/MODELS.md), `scripts/setup.py`.

## Возможности и зависимости

- CPU: редактор, ручные правки, Optical Flow при наличии OpenCV, предпросмотр,
  точное сохранение пикселей вне маски и mask-only граф.
- Необязательный GPU: полная генерация требует совместимого ComfyUI, upstream
  H3/Video-нод и выбранных в графе моделей. Веса не входят в репозиторий и не
  скачиваются автоматически.
- AutoPrompt использует только уже установленный Florence-2/PromptGen и путь
  из `folder_paths` или `THS_UNIVERSAL_INPAINT_LLM_PATH`.
- Native SAM3 требует совместимого `comfyui-sam3`, tokenizer и локального
  checkpoint. Лицензии этих моделей принадлежат upstream-проектам; здесь они
  не распространяются и не переоформляются.

## Ограничения

Проверка текущего редактора охватила один портретный клип: 119 кадров,
1080×1920, 30 FPS, native SAM3 и Optical Flow. Это ограниченная интеграционная
проверка, не универсальная гарантия качества. Для публичной сборки новый H3
рендер не выполнялся; результат H3 зависит от локальных моделей, VRAM и
версий ComfyUI.

В workflow-файлах оставлены явные placeholders
`REPLACE_WITH_INPUT_VIDEO.mp4`, `REPLACE_WITH_REPAIR_PROMPT` и т.п. Не
добавляйте личные видео, маски, prompts, модели, пути хоста или acceptance
материалы в git.

## Проверки

```powershell
python -m pytest -q
node --test tests/*.test.mjs
```

См. [docs/USER_GUIDE_RU.md](docs/USER_GUIDE_RU.md) и
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Продукт: **Universal Inpaint**, владелец: **jimmorisedu-boop**, версия **0.1.0**.
Репозиторий:
[jimmorisedu-boop/ComfyUI-Universal-Inpaint](https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint).
