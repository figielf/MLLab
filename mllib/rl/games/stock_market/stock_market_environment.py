import itertools

import numpy as np


class stock_market:
    def __init__(self, initial_cash, stock_price_history):
        self._initial_portfolio_cash = initial_cash
        self.stock_price_history = stock_price_history
        self.n_steps, self.n_stocks = stock_price_history.shape

        self.state_dim = 2 * self.n_stocks + 1  # (stocks_in_portfolio, current_stock_price, cash_in_hand)

        # equity price history
        self.current_stock_price = None

        # portfolio
        self.cash_in_hand = None
        self.stocks_in_portfolio = None

        # helpers
        self.t = None
        self.action_space = np.arange(self.n_stocks ** 3, dtype=int)

        # create list of all possible combinations of buy, hold, sell for each equity, 0 - sell, 1 - hold, 2 - buy
        single_stock_action_space = ['S', 'H', 'B']
        self.action_space_map = [list(i) for i in itertools.product(single_stock_action_space, repeat=self.n_stocks)]

        assert len(self.action_space) == len(self.action_space_map)

        # set all running env properties
        self.reset()

    def reset(self):  # returns state
        self.t = 0

        self.current_stock_price = np.array(self.stock_price_history[self.t])

        self.cash_in_hand = self._initial_portfolio_cash
        self.stocks_in_portfolio = np.zeros(self.n_stocks, dtype=int)
        return self.current_state()

    def step(self, action_id):
        assert action_id in self.action_space

        old_value = self.calc_current_portfolio_value()

        self.t += 1
        self.current_stock_price = self.stock_price_history[self.t]

        portfolio_action = self.action_space_map[action_id]

        self.rebalance_portoflio(portfolio_action)

        new_value = self.calc_current_portfolio_value()
        reward = new_value - old_value

        info = {'current_portfolio_value': new_value}
        done = self.t == (self.n_steps - 1)
        return self.current_state(), reward, done, info

    def calc_current_portfolio_value(self):
        stocks_value = self.stocks_in_portfolio.dot(self.current_stock_price)
        return self.cash_in_hand + stocks_value

    def current_state(self):
        state = np.empty(self.state_dim)
        state[:self.n_stocks] = self.stocks_in_portfolio
        state[self.n_stocks: 2 * self.n_stocks] = self.current_stock_price
        state[-1] = self.cash_in_hand
        return state

    def rebalance_portoflio(self, action):
        assert len(action) == self.n_stocks

        sell_idxs = []
        buy_idxs = []
        for idx, a in enumerate(action):
            if a == 'S':
                sell_idxs.append(idx)
            elif a == 'B':
                buy_idxs.append(idx)

        self._sell_stocks(sell_idxs)  # sell all stocks of these indexes
        self._buy_stocks(buy_idxs)  # buy stocks of these indexes for all available cash with round robin buy strategy

    def _sell_stocks(self, stock_idxs):
        for idx in stock_idxs:
            self.cash_in_hand += self.stocks_in_portfolio[idx] * self.current_stock_price[idx]
            self.stocks_in_portfolio[idx] = 0

    def _sell_stocks(self, stock_idxs):
        for idx in stock_idxs:
            self.cash_in_hand += self.stocks_in_portfolio[idx] * self.current_stock_price[idx]
            self.stocks_in_portfolio[idx] = 0

    def _buy_stocks(self, buy_idxs):
        if buy_idxs:
            can_buy = True
            while can_buy:
                for idx in buy_idxs:
                    if self.cash_in_hand > self.current_stock_price[idx]:
                        self.cash_in_hand -= self.current_stock_price[idx]
                        self.stocks_in_portfolio[idx] += 1  # buy one equity
                    else:
                        can_buy = False
