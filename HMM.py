from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import copy
import itertools


class HMMStrategy:
    def __init__(self, code, df, test_size=0.33, n_hidden_states=4, n_latency_days=10, n_steps_frac_change=50,
                 n_steps_frac_high=10, n_steps_frac_low=10):

        self.data4cal = copy.deepcopy(df)
        self.hmm = GaussianHMM(n_components=n_hidden_states)
        self.n_latency_days = n_latency_days

        self._split_train_test_data(test_size)
        self._compute_all_possible_outcomes(
            n_steps_frac_change, n_steps_frac_high, n_steps_frac_low)

        self.start_date = ""
        self.end_date = ""

    def _split_train_test_data(self, test_size):
        self._train_data_df, self._test_data_df = train_test_split(self.data4cal, test_size=test_size, shuffle=False)
        # 因为是时间序列，关闭shuffle，以避免随机选点

    # 为什么要静态？
    @staticmethod
    def _extract_features(df):
        # creat multi-observation matrix
        open_price = np.array(df["open"])
        high_price = np.array(df["high"])
        low_price = np.array(df["low"])
        close_price = np.array(df["close"])

        # A股的涨跌幅限制是 (price - pre_close) / pre_close
        # TODO后续将open_price,替换为pre_close
        frac_change = (close_price - open_price) / open_price
        frac_high = (high_price - open_price) / open_price
        frac_low = (open_price - low_price) / open_price  # 正常应该是abs((low_price - open_price) / open_price)

        return np.column_stack([frac_change, frac_high, frac_low])

    def fit(self):
        # 从 train_data 中分离出 frac_change, frac_high, frac_low
        feature_vectors = HMMStrategy._extract_features(self._train_data_df)
        self.hmm.fit(feature_vectors)

    def _compute_all_possible_outcomes(self, n_steps_frac_change, n_steps_frac_high, n_steps_frac_low):
        # 自己设置 frac_change, frac_high, frac_low 的取值范围， 在[-10%, 10%]的范围内创建所有可能的取值。
        frac_change_range = np.linspace(-0.1, 0.1, n_steps_frac_change)  # 涨跌幅在 -10% - 10%
        frac_high_range = np.linspace(0, 0.1, n_steps_frac_high)  # 涨幅 10%
        frac_low_range = np.linspace(0, 0.1, n_steps_frac_low)  # 跌幅 10%

        # 将上述结果排列组合, 如果不用itertools, 我就得三重for循环
        # 这里必须有list一步转换，才能转换为 np.array
        self._possible_outcomes = np.array(list(itertools.product(
            frac_change_range, frac_high_range, frac_low_range
        )))

    # 使用[day_index - self.n_latency_days, day_index - 1]的数据
    # 预测day_index的 (frac_change, frac_high, frac_low)
    def _get_most_probable_outcome(self, day_index):
        previous_data_start_index = max(0, day_index - self.n_latency_days)  # max可以控制不出界啊，牛逼！！
        previous_data_end_index = max(0, day_index - 1)  # TODO需要 -1 吗? [start:end_index]就不包括end_index了啊
        # previous_data_end_index = max(0, day_index) #TODO这里就不-1了，因为[start:end_index]就不包括end_index了啊
        # 而 self._test_data_df.iloc[day_index]索引的是 day_index

        # 最开始划分出来的训练集和测试集中的测试集
        previous_data = self._test_data_df.iloc[previous_data_start_index: previous_data_end_index]
        # 从测试集中抽离出[frac_change, frac_high, frac_low])
        previous_data_features = HMMStrategy._extract_features(previous_data)

        outcome_score = []
        # 遍历所有的结果集合

        """
        将possible_outcome 挨个接在 previous_data_featurs下面，卧槽，这不就是我一直在寻找的append方法吗？？
            |   |   |
            |   |   |
            |   |   |
            o   o   o  这里面o:代表possible_outcome

        """
        for possible_outcome in self._possible_outcomes:
            total_data = np.row_stack(
                (previous_data_features, possible_outcome))
            # score之前需要先fit，不然报错
            # NotFittedError: This GaussianHMM instance is not fitted yet.
            # Call 'fit' with appropriate arguments before using this method.
            outcome_score.append(self.hmm.score(total_data))  # 第一类问题：给定λ = (Π, A, B) 计算观测序列出现概率
        most_probable_outcome = self._possible_outcomes[np.argmax(
            outcome_score)]

        # 返回的是 (frac_change, frac_high, frac_low)
        return most_probable_outcome

    def predict_close_price(self, day_index):
        # TODO，用pre_close预测，代替open预测
        open_price = self._test_data_df.iloc[day_index]["open"]
        predict_frac_change, _, _ = self._get_most_probable_outcome(day_index)
        return open_price * (1 + predict_frac_change)

    def predict_close_price_for_days(self, days):
        predicted_close_prices = []
        for day_index in tqdm(range(days)):
            predicted_close_prices.append(self.predict_close_price(day_index))
        return predicted_close_prices

    def predict_close_price_by_date(self, date):
        day_index = self.data4cal.index.get_loc(date)
        return self.predict_close_price(day_index)

    def predict_close_price_for_date_range(self, start_date, end_date):
        predicted_close_prices = []
        start_index = self.data4cal.index.get_loc(start_date)
        end_index = self.data4cal.index.get_loc(end_date)
        for day_index in tqdm(range(start_index, end_index, 1)):
            predicted_close_prices.append(self.predict_close_price(day_index))
        # 最好是能够按照start_date, end_date 中间的交易日作为index，返回Series
        return predicted_close_prices

    def getOrderList(self, start_date, end_date):
        for predicted_close_price in predicted_close_prices:
            if predicted_close_price > open_price:
                "buy"
            elif predicted_close_price < hold_price:
                "sell"

        pass
