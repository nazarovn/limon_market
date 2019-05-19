import numpy as np
from sklearn.model_selection import train_test_split


class Player:
    def __init__(self, strategies, model, target_func, alpha=1):
        self.model = model
        self.strategies = np.array(strategies)
        self.target_func = target_func
        self.alpha = alpha

        self.hist_X_y = {'X': [], 'y': []}
        self.current_X_y = {'X': [], 'y': []}

    def _get_X_y(self, n_games):
        """Берёт признаки по последним играм для обучения"""
        X = np.vstack(self.hist_X_y['X'][-n_games:])
        y = np.hstack(self.hist_X_y['y'][-n_games:])
        return (X, y)

    def _get_last_features(self, data):
        """Вычисляет признаки для последний итерации текущей игры"""
        raise NotImplementedError

    def end_game(self, data):
        """Обновляет историю данных для обучения.
        
        Parameters
        ----------
        data : {np.array} массив с 3 столбцами:
            - тип авто (0 - лимон, 1 - персик),
            - цена продавца
            - решение покупателя (0 - не купил, 1 - купил)
        """
        self.hist_X_y['X'].append(np.array(self.current_X_y['X']))
#         print('data', data)
        self.hist_X_y['y'].append(self.target_func(data))
#         print('target_func', self.target_func(data))
        self.current_X_y = {'X': [], 'y': []}

    def fit(self, n_games, batch_size=100, epochs=1):
        """
        Обучение модели предсказания ожидаемого выигрыша.
        
        Parameters
        ----------
        data : {np.array} массив с 3 столбцами:
            - тип авто (0 - лимон, 1 - персик),
            - цена продавца
            - решение покупателя (0 - не купил, 1 - купил)
        n_games : {int} количество последних игр для обучения
        batch_size - параметр обучения нейроной сети на keras
        epochs - параметр обучения нейроной сети на keras
        """
        X, y = self._get_X_y(n_games)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#         print(X.shape, y.shape, y)
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0., verbose=0)

    
        predict_train = self.model.predict([X_train]).T[0]
        score_train = ((predict_train - y_train) ** 2).mean()
        predict_test = self.model.predict([X_test]).T[0]
        score_test = ((predict_test - y_test) ** 2).mean()
#         print(f'Model {self.name.upper()} scores:\t train = {round(score_train, 1)}\t test = {round(score_test, 1)}')

    def action(self, data):
        """
        Generate Player's action.

        Parameters
        ----------
        data : {np.array} Информация о прошлых итерациях:
            - тип авто (0 - лимон, 1 - персик),
            - цена продавца
            - решение покупателя (0 - не купил, 1 - купил)
        # strategies: {np.array} множество стратегий игрока
        alpha: {float, >0} множитель выигрыша в soft-max

        Return
        ------
        action - элемент из множества strategies
        """
        features = self._get_last_features(data)

        X = []
        if (self.name == 'seller') and (data[-1, 0] == 1):
            strategies = self.strategies[self.strategies > 5]
        else:
            strategies = self.strategies


        for strat in strategies:
            features_with_strat = features.copy()
            features_with_strat[0] = strat
            X.append(features_with_strat)

        profit = self.model.predict([X]).T[0]
        profit -= np.min(profit)
#         print('\nprofit', profit)
        exp_profit = np.exp(profit * self.alpha)
        prob_strategy = exp_profit / np.sum(exp_profit)
#         print('data', data[-1])
#         print(self.name, np.round(prob_strategy,4))

        action = np.random.choice(strategies, p=prob_strategy)
        features[0] = action
        self.current_X_y['X'].append(features)
        return action


class Seller(Player):
    name = 'seller'

    def _get_last_features(self, data):
        """
        Вычисляет признаки для последний итерации текущей игры.
        
        Parameters
        ----------
        data : {np.array} Информация о прошлых итерациях:
            - тип авто (0 - лимон, 1 - персик),
            - цена продавца
            - решение покупателя (0 - не купил, 1 - купил)

        Return
        ------
        features : {np.array}
            0 - стратегия продавца
            1 - тип автомобиля (!)
            -- статистика прошлых итераций (сколько раз произошло) --
            2 - лимон - дорого - купил
            3 - лимон - дорого - не купил
            4 - лимон - дешево - купил
            5 - лимон - дешево - не купил
            6 - персик - дорого - купил
            7 - персик - дорого - не купил
            8 - персик - дешево - купил
            9 - персик - дешево - не купил
        """
        res = np.zeros(10)
        res[1] = data[-1, 0]
        if len(data) > 1:
            res[2] = ((data[:, 0] == 0) & (data[:, 1] > 5) & (data[:, 2] == 1)).sum()
            res[3] = ((data[:, 0] == 0) & (data[:, 1] > 5) & (data[:, 2] == 0)).sum()
            res[4] = ((data[:, 0] == 0) & (data[:, 1] < 5) & (data[:, 2] == 1)).sum()
            res[5] = ((data[:, 0] == 0) & (data[:, 1] < 5) & (data[:, 2] == 0)).sum()
            res[6] = ((data[:, 0] == 1) & (data[:, 1] > 5) & (data[:, 2] == 1)).sum()
            res[7] = ((data[:, 0] == 1) & (data[:, 1] > 5) & (data[:, 2] == 0)).sum()
            res[8] = ((data[:, 0] == 1) & (data[:, 1] < 5) & (data[:, 2] == 1)).sum()
            res[9] = ((data[:, 0] == 1) & (data[:, 1] < 5) & (data[:, 2] == 0)).sum()
        return res


class Customer(Player):
    name = 'customer'
    
    def _get_last_features(self, data):
        """
        Вычисляет признаки для последний итерации текущей игры.
        
        Parameters
        ----------
        data : {np.array} Информация о прошлых итерациях:
            - тип авто (0 - лимон, 1 - персик),
            - цена продавца
            - решение покупателя (0 - не купил, 1 - купил)

        Return
        ------
        features : {np.array}
            0 - стратегия покупателя
            1 - цена продавца (!)
            2 - номер итерации
            3 - количество лимонов за игру
            4 - количество отказов от дорогих предложений
            5 - количество отказов от дешёвых предложений
            6 - количество принятых дешёвых предложений
            7 - количество принятых дорогих обманов
            8 - количество принятых дорогих честных преложений
        """
        res = np.zeros(10)
        res[1] = data[-1, 1]
        if len(data) > 1:
            res[2] = ((data[:, 0] == 0) & (data[:, 1] > 5) & (data[:, 2] == 1)).sum()
            res[3] = ((data[:, 0] == 0) & (data[:, 1] > 5) & (data[:, 2] == 0)).sum()
            res[4] = ((data[:, 0] == 0) & (data[:, 1] < 5) & (data[:, 2] == 1)).sum()
            res[5] = ((data[:, 0] == 0) & (data[:, 1] < 5) & (data[:, 2] == 0)).sum()
            res[6] = ((data[:, 0] == 1) & (data[:, 1] > 5) & (data[:, 2] == 1)).sum()
            res[7] = ((data[:, 0] == 1) & (data[:, 1] > 5) & (data[:, 2] == 0)).sum()
            res[8] = ((data[:, 0] == 1) & (data[:, 1] < 5) & (data[:, 2] == 1)).sum()
            res[9] = ((data[:, 0] == 1) & (data[:, 1] < 5) & (data[:, 2] == 0)).sum()
        return res
