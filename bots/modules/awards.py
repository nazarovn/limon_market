import numpy as np


def get_profit(data, weight_delta):
    """
    Вычисляет прибыль для продавци и покупателя на каждой итерации
    
    Parameters
    ----------
    data : {np.array} массив с 3 столбцами:
        - тип авто (0 - лимон, 1 - персик),
        - цена продавца
        - решение покупателя (0 - не купил, 1 - купил)
    weight_delta : {float} вес абсолютной разницы в прибилы в выигрыше продавца 
        При отрицательных weight_delta получаем штраф за обман.
    
    Return
    ------
    {np.array} массив с 2 столбцами: выигрыш продавца и покупателя
    """
    res = np.zeros([len(data), 2])
    res[:, 0] = data[:, 2] * (data[:, 0] * (data[:, 1] - 10) + (1 - data[:, 0]) * (data[:, 1] - 1))
    res[:, 1] = data[:, 2] * (data[:, 0] * (12 - data[:, 1]) + (1 - data[:, 0]) * (2 - data[:, 1]))
    
    res[:, 0] = res[:, 0] + weight_delta * np.abs(res[:, 0] - res[:, 1])
    return res


def get_target_q_learning(data, lambda_Q=0.5, weight_delta=0):
    """
    Вычисляет таргеты для обучения моделей продавца и покупателя на каждой итерации.
    Таргет - приведённый ожидаемый выигрыш (с дисконтированием)
    
    Parameters
    ----------
    data : {np.array} массив с 3 столбцами:
        - тип авто (0 - лимон, 1 - персик),
        - цена продавца
        - решение покупателя (0 - не купил, 1 - купил)
    lambda_Q : {float} коэффициент дисконтирования
    weight_delta : {float} вес абсолютной разницы в прибилы в выигрыше продавца 
        При отрицательных weight_delta получаем штраф за обман.
    
    Return
    ------
    {np.array} массив с 2 столбцами: выигрыш продавца и покупателя
    """
    n = len(data)
    profit = get_profit(data, weight_delta)
    
    weights = np.ones([n, 1]) * np.arange(1, n + 1) - np.arange(1, n + 1).reshape(-1, 1)
    weights = (lambda_Q ** weights) * (weights >= 0)
    res = weights.dot(profit)
    return res

