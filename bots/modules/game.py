import numpy as np


class Game:
    """
    Imitation game.

    Parameters
    ----------
    seller: Seller object
    customer: Customer object
    count_game: int
        The number of simulated games.

    Inside parameters
    -----------------
    self.data: list of tuple
        Element of list if one simulated game.
        (actions' players, benefits' players)
    """

    def __init__(self, sellers, customers):
        self.sellers = sellers
        self.customers = customers
        assert len(sellers) == len(customers)
        
        self.count_iter = 20
        self.car_types = [0, 1]
        self.car_prob = [0.25, 0.75]
        self.data_hist = []
        self.agg_hist = []

    def _run_one_game(self, data):
        """Имитирует розыгрыш одной игры
        
        Parameters
        ----------
        data : {np.array} с 6 столбцами:
            0 - номер итерации
            1 - номер продавца
            2 - номер покупателя
            3 - тип автомобиля (= -1)
            4 - ход продавца (= -1)
            5 - ход покупателя (= -1)
        """
        n_pairs = len(self.sellers)
        for idx_iter in range(self.count_iter):
            for idx_seller in range(n_pairs):
                idx_customer = int(data[(data[:, 0] == idx_iter) & (data[:, 1] == idx_seller)][0, 2])

                # seller
                seller_data = data[(data[:, 0] <= idx_iter) & (data[:, 1] == idx_seller)][:, 3:]
#                 print('seller_data', seller_data)
#                 print('seller', seller_data)
                seller_action = self.sellers[idx_seller].action(seller_data)
#                 print('seller action', seller_action)
#                 print('seller data before action', data)
                data[(data[:, 0] == idx_iter) & (data[:, 1] == idx_seller), 4] = seller_action
#                 print('seller data after action', data)
                # customer
                customer_data = data[(data[:, 0] <= idx_iter) & (data[:, 2] == idx_customer)][:, 3:]
#                 print('customer', customer_data)
                customer_action = self.customers[idx_customer].action(customer_data)
                data[(data[:, 0] == idx_iter) & (data[:, 1] == idx_customer), 5] = customer_action

        # говорим игрокам, что игра закончилась
        for idx in range(n_pairs):
            seller_data = data[data[:, 1] == idx][:, 3:]
            self.sellers[idx].end_game(seller_data)
            customer_data = data[data[:, 2] == idx][:, 3:]
            self.customers[idx].end_game(customer_data)

        self.data_hist.append(data)

#         target = get_target_q_learning(game_data)
#         one_game = {'game': game_data, 'seller_features': X_seller,
#                     'customer_features': X_customer, 'target': target}
#         return one_game

#     def _update_agg_hist(self, count_game):
#         agg_hist = {'car_type_0': {'customer_0': Counter(), 'customer_1': Counter()},
#                     'car_type_1': {'customer_0': Counter(), 'customer_1': Counter()}}
#         for game in self.data[-count_game:]:
#             for car_type, seller, customer in game['game']:
#                 agg_hist['car_type_{}'.format(int(car_type))]['customer_{}'.format(int(customer))][seller] += 1
#         self.agg_hist.append(agg_hist)


    def _generate_pairs_players(self):
        """Генерирует случайные пары продавцов-покупателей и типы автомобилей для одной игры,
        формирует массив для записи результатот имитации игры.
        
        Return
        ------
        {np.array} с 6 столбцами:
            0 - номер итерации
            1 - номер продавца
            2 - номер покупателя
            3 - тип автомобиля (= -1)
            4 - ход продавца (= -1)
            5 - ход покупателя (= -1)        
        """
        n_pairs = len(self.sellers)
        count_iter = self.count_iter
        data = np.zeros([n_pairs * count_iter, 6]) - 1
        # iteration number
        data[:, 0] = (np.arange(count_iter) * np.ones((n_pairs, 1))).T.reshape(-1)
        # sellers
        data[:, 1] = (np.arange(n_pairs) * np.ones((count_iter, 1))).reshape(-1)
        # customers
        data[:, 2] = (np.random.choice(np.arange(n_pairs), n_pairs, replace=False) * np.ones((count_iter, 1))).reshape(-1)
        # cars
        data[:, 3] = np.random.choice(self.car_types, count_iter * n_pairs, p=self.car_prob)

        return data

    def _play(self, play_games):
        """Имитируем розыгрыш игр.
        
        Parameters
        ----------
        play_games : {int} количесвто игр.
        """
        for idx_game in range(play_games):
            data = self._generate_pairs_players()
            one_game = self._run_one_game(data)

        # self._update_agg_hist(count_game)
            

    def _fit_players(self, fit_games):
        for seller in self.sellers:
            seller.fit(n_games=fit_games)
        for customer in self.customers:
            customer.fit(n_games=fit_games)

    def run(self, fit_games=10, play_games=10):
        """
        Запускаем имитацию розыгрышей игры, и после этого производит дообучение моделей.
        
        Parameters
        ----------
        fit_games : {int} количество последних игр, для обучения
        count_games : {int} количество имитируемых игр
        """
        self._play(play_games=play_games)
        self._fit_players(fit_games=fit_games)
