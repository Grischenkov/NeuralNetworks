# Лабораторная работа №4. Рекуррентные нейронные сети(NumPy)
## Задание
*Цель:* написать RNN, GRU и LSTM для прогнозирования Usage_kWh.
*Датасет:* http://archive.ics.uci.edu/ml/datasets/Steel+Industry+Energy+Consumption+Dataset

Сравнить качество работы моделей по MSE, RMSE и R^2 и сделать выводы.

## Результаты
Метрики:

|      | MSE        | RMSE      | R^2      |
|------|------------|-----------|----------|
|RNN   | 541.282010 | 23.265468 | 0.433514 |
|GRU   | 371.262692 | 19.268178 | 0.611450 |
|LSTM  | 483.203748 | 21.981896 | 0.494296 |

#### RNN
История обучения:

![История обучения RNN](images/rnn/history.png?raw=true "history")

График целевой переменной:

![График целевой переменной RNN](images/rnn/chart.png?raw=true "chart")

#### GRU
История обучения:

![История обучения GRU](images/gru/history.png?raw=true "history")

График целевой переменной:

![График целевой переменной GRU](images/gru/chart.png?raw=true "chart")

#### LSTM
История обучения:

![История обучения LSTM](images/lstm/history.png?raw=true "history")

График целевой переменной:

![График целевой переменной LSTM](images/lstm/chart.png?raw=true "chart")