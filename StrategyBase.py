import pandas as pd
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import Line


class Toward:
    buy = "buy"
    sell = "sell"

class StrategyBase:
    """
    输入df必有字段
                close
    trade_date
    """


    def __init__(self, code, data: pd.DataFrame, ensure_profit = True):
        self.code = code
        self.data4cal = copy.deepcopy(data)  # 避免污染原始数据
        self.init_funds = 100000
        self.current_funds = self.init_funds
        self.current_hands = 0
        self.last_buy_date = None
        self.events_list = []


        self.logger = True
        self.ensure_profit = ensure_profit
        self.holding = False  # 持仓控制
        # self.final_stock_price = 0 删掉了，通过property实现
        # self.final_value = 0
        # self.inited = False #初始化控制，保证只初始化一次


    def commission_cal(self, price, volume, toward):
        # 过户费
        transfer_fee = 0
        # 印花税
        commission_tax = 0
        # 佣金
        commission_broker = 0
        if toward == Toward.buy:
            transfer_fee = volume * 100 * 0.0006
            transfer_fee = transfer_fee if transfer_fee > 1 else 1
            commission_broker = price * volume * 100 * 0.0003
            commission_broker = commission_broker if commission_broker > 5 else 5
        elif toward == Toward.sell:
            transfer_fee = volume * 100 * 0.0006
            transfer_fee = transfer_fee if transfer_fee > 1 else 1
            commission_broker = price * volume * 100 * 0.0003
            commission_broker = commission_broker if commission_broker > 5 else 5


            commission_tax = price * volume * 100 * 0.0001
        else:
            raise AttributeError
        return transfer_fee + commission_tax + commission_broker


    def potential_profit(self, data: pd.DataFrame, bought_date, potential_selling_day, current_position):
        bought_price = data.loc[bought_date, "close"]
        potential_selling_price = data.loc[potential_selling_day, "close"]
        commission_buy = self.commission_cal(bought_price, current_position, Toward.buy)
        commission_sell = self.commission_cal(potential_selling_price, current_position, Toward.sell)
        profit = (potential_selling_price - bought_price) * current_position * 100 - commission_buy - commission_sell
        return profit


    def hands_can_bought(self, money, price):
        assert (money >= 0 and price >= 0)
        potential_hands = math.floor(money / price / 100)
        left_money = money - potential_hands * 100 * price - self.commission_cal(price, potential_hands, Toward.buy)
        if left_money >= 0:
            return potential_hands
        elif potential_hands >= 1:
            return potential_hands - 1
        elif potential_hands == 0:
            return 0


    def final(self, df: pd.DataFrame):
        for date, data in df.iterrows():
            if (not self.holding) and data.buy:
                num_hands_to_buy = self.hands_can_bought(self.current_funds, data.close)
                self.current_hands += num_hands_to_buy
                cost = num_hands_to_buy * 100 * data.close + \
                       self.commission_cal(data.close, num_hands_to_buy, Toward.buy)
                
                self.events_list.append(("b", date, self.current_funds, # 记录买仓之前的历史资金，求单次收益
                                         self.commission_cal(data.close, num_hands_to_buy, Toward.buy), data.close))
                self.current_funds -= cost
                self.last_buy_date = date


                self.holding = True
                if self.logger:
                    print("Bought {} hands of {} on {} at {} price, cost {}".format(num_hands_to_buy, self.code, date,
                                                                              data.close, cost))
            elif self.holding and data.sell:
                if self.ensure_profit:


                    potential_profit = self.potential_profit(df, self.last_buy_date, date, self.current_hands)
                    if potential_profit > 0:
                        # 不能直接加 potential_profit, 否则买入手续费会计算两次
                        self.current_funds += self.current_hands * 100 * data.close - \
                                              self.commission_cal(data.close, self.current_hands, Toward.sell)
                        if self.logger:
                            print("Sold {} hands of {} on {} at {} price, profit {}, current funds {}".format(
                                self.current_hands,self.code, date, data.close, potential_profit,self.current_funds))
                        self.current_hands = 0  # 一次卖空
                        self.holding = False
                        self.events_list.append(("s", date, self.current_funds, # 记录平仓之后的资金，求单次收益
                                                 self.commission_cal(data.close, self.current_hands, Toward.sell), data.close))
                else:
                    self.current_funds += self.current_hands * 100 * data.close - \
                                          self.commission_cal(data.close, self.current_hands, Toward.sell)
                    potential_profit = self.potential_profit(df, self.last_buy_date, date, self.current_hands)
                    if self.logger:
                        print("Sold {} hands of {} on {} at {} price, profit {}, current funds {}".format(
                            self.current_hands, self.code, date, data.close, potential_profit, self.current_funds))
                    self.current_hands = 0  # 一次卖空
                    self.holding = False
                    self.events_list.append(("s", date, self.current_funds,
                                            self.commission_cal(data.close, self.current_hands, Toward.sell), data.close))
    @property
    def final_stock_price(self):
        return self.data4cal.close.iloc[-1]
    @property
    def final_value(self):
        #最终可能会低于bought的钱数，因为我们又赔了
        return self.current_funds + self.current_hands * 100 * self.final_stock_price
    @property
    def final_return(self):
        return self.final_value/self.init_funds - 1
    @property
    def return_list(self):
        result = []
        for i in range(0, len(self.events_list)//2*2, 2): #//2*2 保证b-s成对出现
            res = self.events_list[i+1][2] / self.events_list[i][2] - 1
            result.append(res)
        return result
    @property
    def ave_return(self):
        return np.mean(self.return_list)
    @property
    def ave_return_dayly(self):
        return np.mean(self.return_dayly_list)
    @property
    def ave_return_yearly(self):
        return np.mean(self.return_yearly_list)
    
    @property
    def commission_total(self):
        result = 0
        for data_tuple in self.events_list:
            result += data_tuple[3]
        return result
    @property
    def return_yearly_list(self):
        result = []
        for i in range(0, len(self.events_list)//2*2, 2): #//2*2 保证b-s成对出现
            pre_return = self.events_list[i+1][2] / self.events_list[i][2]
            interval = self.events_list[i+1][1] - self.events_list[i][1]
            result.append(self.return_days2year(pre_return, interval.days)-1)
        return result
    @property
    def return_dayly_list(self):
        result = []
        for i in range(0, len(self.events_list)//2*2, 2): #//2*2 保证b-s成对出现
            pre_return = self.events_list[i+1][2] / self.events_list[i][2]
            interval = self.events_list[i+1][1] - self.events_list[i][1]
            result.append(self.return_days2day(pre_return, interval.days)-1)
        return result
    @property
    def prob_win(self):
        pos = 0
        neg = 0
        for r in self.return_list:
            if r > 0:
                pos += 1
            else: #将nan也记作失败
                neg += 1
            # elif r < 0:
            #     neg +=1
        try:
            prob = pos/(pos+neg)
        except:
            prob = 0
        return prob, pos, neg
    
    # 用于实时跟踪当前权益变化
    def value_list(self):
        pass
    
    def _init_process(self):
        raise NotImplementedError


    #尽量，不要传入参数，以统一格式
    def buy_condition(self):
        raise NotImplementedError


    def sell_condition(self):
        raise NotImplementedError
    
    def get_events(self, begin_date, end_date): #datetime 类型
        result = []
        for data in self.events_list:
            if data[1] >= begin_date and data[1] <= end_date:
                result.append((data[0], data[1]))
        return result

    def draw(self):
        l = (
            Line()
            .add_xaxis(xaxis_data=list(self.data4cal.index))
            .add_yaxis(series_name=self.code,
                      y_axis=list(self.data4cal.close))
        )
        for i in range(0, len(self.events_list)//2*2, 2):
            x = [self.events_list[i][1], self.events_list[i+1][1]]
            y = [self.events_list[i][4], self.events_list[i+1][4]]
            tmp_line = (
                Line()
                .add_xaxis(xaxis_data = x)
                .add_yaxis(
                    series_name="",
                    y_axis = y,
                    symbol="triangle",
                    symbol_size=20,
                    linestyle_opts=opts.LineStyleOpts(color="green", width=4, type_="dashed"),
                    itemstyle_opts=opts.ItemStyleOpts(
                        border_width=3,border_color="blue",color="blue"
                    )
                )
            )
            l.overlap(tmp_line)
        
        return l
    
    
    def run(self):
        self._init_process() #后续可以增加 从值中提取，不放在构造函数内部
        self.final(self.data4cal)
    
    def read(self,df):
        self.data4cal = df
        
    def return_days2year(self,return_days, days):
        return return_days**(365/days) #两种理解，1.365天里有多少个间隔days的间隔，然后取幂运算 2.先开days次方，求日收益，然后取365次方求年
    def return_days2day(self,return_days, days):
        return return_days**(1/days)