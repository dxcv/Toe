import pandas as pd
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import Line
from Slippage import SlippagePct

"""
统一变量：
value: 股票价值 = hands * price * 100
funds: 现金价值
equity:权益价值 = value + funds

hands: 持仓手数

"""

# 若模拟盘则应把df拆成一行行，一点一点的喂进去
# 或者，如果我们能pandas解决 部分成交/限价单问题。则更好。
# 对于限价单，我们仍然标记好 买卖信号， 在trade函数中统一处理此逻辑，因为trade也是遍历
# 但在这里面，我们既然是面向bid/ask 来操作的，就暂时不考虑限价单等问题。

class Toward:
    buy = "buy"
    sell = "sell"

# 此策略不考虑 资金分配问题，钱有多少就soha多少。
# 资金分配通过Strategy外层的 allocator去考虑。统一调度。
class StrategyBase4RT:
    """
    输入df必有字段
                买1     卖1     买1量   卖1量
                bid1    ask1    bsize1  asize1 
    date_time

    不考虑优化内存，就都用一个表就完事了。
    最终形成字段
                bid1    ask1    bsize1  asize1  [策略字段:DIF, DEA] buy_signal  sell_signal   handstraded  funds                hands
    date_time   1       2       3       4                  5    6   True        False           7             0                   0 这里就不用nan了
    date_time   1       2       3       4                  5    6   True        False           nan           8                   9
                                                                                        nan可以被drop掉，最终剩下交易的日子, 废弃掉，假如到 events_list里面
                                                                                        pd.DataFrame, df.columns = []

    
    扩充event_list字段， 加入价/量

    """



    def __init__(
        self, code, 
        data: pd.DataFrame, 
        init_funds = 100000, 
        ensure_profit = True, 
        logger = True
        ):
        self.code = code
        self.data4cal = copy.deepcopy(data)  # 避免污染原始数据
        self.init_funds = init_funds

        self.last_buy_date = None
        self.events_list = []
        self.slippage = SlippagePct()

        self.logger = True
        self.ensure_profit = ensure_profit

        # 通过变量名去操作 data4cal， 避免写死
        self.bid = "bid1"
        self.ask = "ask1"

    # 一个决定一次交易 买/卖 多少手股票的函数
    #@property，有参数无法设置成property
    def hands2trade(self, price:float, size:int, toward:str): #size, 买为asize1， 卖为bsize1
        """
        buy: price-ask1, size-asize1
        sell: price-bid1, size-bsize1
        size  不要写死成self.data4cal["asize"/"bsize"]， 还是留一个形参
        """
        assert(toward in [Toward.buy, Toward.sell])
        if toward == Toward.buy:
            hands_can_bought = self._handsCanBought(self.current_funds, price)
            if hands_can_bought > size:
                return size
            else:
                return hands_can_bought
        elif toward == Toward.sell:
            if self.current_hands > size:
                return size
            else:
                return self.current_hands
        else:
            raise AttributeError

    # 用于实时跟踪当前权益变化
    # 能后续直接推出来的，计算的时候就不放在df中
    @property
    def equity_series(self):
        return self.data4cal["funds"] + self.data4cal["hands"] * 100 * self.data4cal["bid1"] # 用买1来衡量吧，这样对我们来讲是更恶劣的条件

    def commissionCal(self, price, volume, toward):
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


    def potentialProfit(self, bought_date, potential_selling_day, hands2trade):
        bought_price = self.data4cal.loc[bought_date, "ask1"] # 卖1
        potential_selling_price = self.data4cal.loc[potential_selling_day, "bid1"]
        commission_buy = self.commissionCal(bought_price, hands2trade, Toward.buy)
        commission_sell = self.commissionCal(potential_selling_price, hands2trade, Toward.sell)
        profit = (potential_selling_price - bought_price) * hands2trade * 100 - commission_buy - commission_sell
        return profit


    def _handsCanBought(self, money, price):
        assert (money >= 0 and price >= 0)
        potential_hands = math.floor(money / price / 100)
        left_money = money - potential_hands * 100 * price - self.commissionCal(price, potential_hands, Toward.buy)
        if left_money >= 0:
            return potential_hands
        elif potential_hands >= 1:
            return potential_hands - 1
        elif potential_hands == 0:
            return 0

    @property
    def isholding(self):
        """
        有持仓，遇到sell_signal，就卖出
        """
        if self.current_hands > 0:
            return True
        else:
            return False
    @property
    def isenough(self, price):
        """
        有足够的钱，遇到buy_signal，就买入
        """
        if self._handsCanBought(self.current_funds, price) > 0:
            return True
        else:
            return False

    # 加入滑点, 
    def trade(self):
        for date, data in self.data4cal.iterrows():
            if (not self.isholding) and data["buy_signal"]:
                pass
            elif (self.isholding) and data["sell_signal"]:
                pass




    # self.current_hands 修改为 pd的一个column，这样方便计算 实时权益
    def final(self, df: pd.DataFrame):
        for date, data in df.iterrows():
            if (not self.holding) and data.buy:
                num_hands_to_buy = self._handsCanBought(self.current_funds, data.close)
                self.current_hands += num_hands_to_buy
                cost = num_hands_to_buy * 100 * data.close + \
                       self.commissionCal(data.close, num_hands_to_buy, Toward.buy)
                
                self.events_list.append(("b", date, self.current_funds, # 记录买仓之前的历史资金，求单次收益
                                         self.commissionCal(data.close, num_hands_to_buy, Toward.buy), data.close))
                self.current_funds -= cost
                self.last_buy_date = date


                self.holding = True
                if self.logger:
                    print("Bought {} hands of {} on {} at {} price, cost {}".format(num_hands_to_buy, self.code, date,
                                                                              data.close, cost))
            elif self.holding and data.sell:
                if self.ensure_profit:


                    potential_profit = self.potentialProfit(df, self.last_buy_date, date, self.current_hands)
                    if potential_profit > 0:
                        # 不能直接加 potential_profit, 否则买入手续费会计算两次
                        self.current_funds += self.current_hands * 100 * data.close - \
                                              self.commissionCal(data.close, self.current_hands, Toward.sell)
                        if self.logger:
                            print("Sold {} hands of {} on {} at {} price, profit {}, current funds {}".format(
                                self.current_hands,self.code, date, data.close, potential_profit,self.current_funds))
                        self.current_hands = 0  # 一次卖空
                        self.holding = False
                        self.events_list.append(("s", date, self.current_funds, # 记录平仓之后的资金，求单次收益
                                                 self.commissionCal(data.close, self.current_hands, Toward.sell), data.close))
                else:
                    self.current_funds += self.current_hands * 100 * data.close - \
                                          self.commissionCal(data.close, self.current_hands, Toward.sell)
                    potential_profit = self.potentialProfit(df, self.last_buy_date, date, self.current_hands)
                    if self.logger:
                        print("Sold {} hands of {} on {} at {} price, profit {}, current funds {}".format(
                            self.current_hands, self.code, date, data.close, potential_profit, self.current_funds))
                    self.current_hands = 0  # 一次卖空
                    self.holding = False
                    self.events_list.append(("s", date, self.current_funds,
                                            self.commissionCal(data.close, self.current_hands, Toward.sell), data.close))
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
    
    
    def _init_process(self):
        raise NotImplementedError


    #尽量，不要传入参数，以统一格式
    def buyCondition(self):
        raise NotImplementedError


    def sellCondition(self):
        raise NotImplementedError
    
    def getEvents(self, begin_date, end_date): #datetime 类型
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

"""
1.

"""