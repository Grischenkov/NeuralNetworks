# Лабораторная работа №6. Трансформеры
## Задание
*Цель:* написать трансформер для распознавания строк, написанных Петром I (AI Jorney 2020)

В качестве метрик использовать CER (Character Error Rate), WER (Word Error Rate) и String Accuracy (https://sites.google.com/site/textdigitisation/qualitymeasures/computingerrorrates)

## Результаты

Используемая модель трансформера: https://huggingface.co/docs/transformers/model_doc/trocr

Метрики:

| CER    | WER    | string_acc |
|--------|--------|------------|
| 0.8461 | 1.8270 | 0.0100     |