# ComfyUI Universal Inpaint

[English](README_EN.md) · [Справочник нод](docs/NODES_RU.md) · [Установка](docs/INSTALL_RU.md) · [Пошаговое руководство V70 со скриншотами](docs/USER_GUIDE_RU.md)

Universal Inpaint изменяет выбранный участок видео по текстовой задаче, референсу или исправленному кадру. Выберите объект через SAM3 или нарисуйте область для Optical Flow. Редактор покажет рамку H3 и позволит рассчитать короткую пробу перед экспортом полного ролика.

[Открыть воркфлоу V70](workflows/V70_Universal_Repair_Workbench.json) · [Пройти пример от загрузки до экспорта](docs/USER_GUIDE_RU.md)

[По тексту](docs/USER_GUIDE_RU.md#по-тексту--without_reference) · [По фотографии](docs/USER_GUIDE_RU.md#по-фотографии--with_reference) · [По исправленному кадру](docs/USER_GUIDE_RU.md#по-исправленному-кадру--corrected_frame) · [Серая заливка и окружение](docs/USER_GUIDE_RU.md#заливка-перед-h3)

![Редактор V70 с маской, рамкой H3 и кнопкой пробного запуска](docs/images/v70/flow-crop.png)

## Какие ноды появятся после установки

В библиотеке ComfyUI откройте **Universal Inpaint**. Пакет добавляет 13 нод. В готовой схеме V70 для первой правки понадобятся следующие.

| Где | Что сделать |
| --- | --- |
| **03 · Редактор маски** | Выделите предмет по клику в SAM3 или нарисуйте область для Optical Flow. Проверьте, что выделение держится на нужных кадрах. |
| **03a · Режим H3** | Выберите замену по тексту, фотографию-референс или исправленный кадр. |
| **03b · Автопромпт** | Для замены по тексту напишите задачу по-русски, например «Удали знак с мяча». |
| **03c · Исправленный кадр** | Если загрузили отредактированный кадр, укажите его номер в исходном видео. Первый кадр имеет номер 1. |
| **Сравнение** | Нажмите «Проба на фрагменте» в редакторе маски. После расчёта передвиньте разделитель по изображению в ноде сравнения. Если правка подходит, нажмите «Запустить H3». |

Зачем нужны остальные ноды, что подать на каждый вход и куда передать результат, описано в [справочнике всех нод](docs/NODES_RU.md).

## Графы

| Файл | Назначение | Нужен H3? |
| --- | --- | --- |
| `V70_Universal_Repair_Workbench.json` | Основной граф с автопромптом, референсом и исправленным кадром. Общие маска, сравнение и экспорт | Да для генерации |
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

В графе H3 поле **«Что изменить?»** принимает задачу обычным текстом и
формирует промпт при запуске. Для русского текста установите локальный
переводчик из профиля `instruction` по [инструкции](docs/INSTALL_RU.md).

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

Локально проверены SAM3 и Optical Flow на портретном клипе из 119 кадров. На видео 544×576 проверены пробы H3 во всех трёх режимах и полный экспорт 56 кадров при 24 FPS в текстовом режиме. Время и качество зависят от модели, семплера и оборудования. [Подробности проверок](docs/LIVE_VALIDATION.md)

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

Продукт: **Universal Inpaint**, владелец: **jimmorisedu-boop**, версия **0.1.2**.
Репозиторий:
[jimmorisedu-boop/ComfyUI-Universal-Inpaint](https://github.com/jimmorisedu-boop/ComfyUI-Universal-Inpaint).
