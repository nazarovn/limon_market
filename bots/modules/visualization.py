import numpy as np
import pandas as pd

from .awards import get_profit


def get_count_trade(data):
    """Считает количество сделок за игру
    
    Parameters
    ----------
    data : {np.array} Информация о прошлых итерациях:
        - тип авто (0 - лимон, 1 - персик),
        - цена продавца
        - решение покупателя (0 - не купил, 1 - купил)
    """
    return (data[:, 2] == 1).sum()


def get_part_honest(data):
    """Считает долю честных предложений продавца"""
    data_ = data[data[:, 0] == 0]
    return (data_[:, 1] < 5).mean()


def get_part_trust(data):
    """Считает долю доверчивых ходов покупателя"""
    data_ = data[data[:, 1] > 5]
    return (data_[:, 2] == 1).mean()


def get_award(data, player_type):
    """Считает выигрыш игрока"""
    profit = get_profit(data, 0)
    if player_type == 'seller':
        return profit[:, 0].sum()
    elif player_type == 'customer':
        return profit[:, 1].sum()
    raise ValueError


def calculate_statistic(list_data, n_pairs):
    """Вычисляет статистики по данным симитированных игр.
    
    Parameters
    ----------
    list_data : {List[np.array]} список массивов с 6 столбцами:
        0 - номер итерации
        1 - номер продавца
        2 - номер покупателя
        3 - тип автомобиля (= -1)
        4 - ход продавца (= -1)
        5 - ход покупателя (= -1)
    n_pairs : {int} количество пар покупатель-продавец

    Return
    ------
    {Dict[pd.DataFrame]} словарь {игрок: статистики по играм}
    """
    statistics = {'sellers': {i: [] for i in range(n_pairs)}, 'customers': {i: [] for i in range(n_pairs)}}
    for data in list_data:
        for idx in range(n_pairs):
            data_ = data[data[:, 1] == idx, 3:]
            statistics['sellers'][idx].append(
                [get_count_trade(data_),
                 get_part_honest(data_),
                 get_part_trust(data_),
                 get_award(data_, 'seller')]
            )
            data_ = data[data[:, 2] == idx, 3:]
            statistics['customers'][idx].append(
                [get_count_trade(data_),
                 get_part_honest(data_),
                 get_part_trust(data_),
                 get_award(data_, 'customer')]
            )

    for player_type in ['sellers', 'customers']:
        for idx in range(n_pairs):
            statistics[player_type][idx] = pd.DataFrame(
                statistics[player_type][idx],
                columns=['count_trade', 'part_honest', 'part_trust', 'award']
            )
    return statistics
        
        
    
    
    
    
    
    
    