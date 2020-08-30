import pandas as pd
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import pyecharts.options as opts
from pyecharts.charts import Line
from Slippage import SlippagePct
from Log import Logger


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

    data4cal["history_funds"]   delta_funds         data4cal["history_hands"]
    剩余资金                     买+/卖- (这样记录符号，方便求平均持仓成本)
    1100(初始资金)                  0                        
    1000                         +100                       1            持仓成本：100/手
    1200                         -200                       0            持仓成本: NaN -> 0，平掉之后要不要归0？
    900                          +300                       3            持仓成本: (100 - 200 + 300)/3 = 66.6/手  还是直接 100/手
    1300                         -400                       1            持仓成本：(100 - 200 + 300 - 400)/1 = -200/手
    这么列出来就清楚了 sum(delta_funds) + 1300 = 初始资金1100
    1000 = 1100 - 100
    1200 = 1000 - (-200)
     990 = 1200 - 300
    
    假设初始 200元买2手，那么1手的持仓成本就是100元，150元卖一手，那么剩余1手的持仓成本就是50元
    假设初始 200元买2手，那么1手的持仓成本就是100元，10元又买一首，那么剩余3手的持仓成本就是210/3=70元
    负的持仓成本说明历史上一定平过仓，还赚到了钱。

    """

    def __init__(
        self, code, 
        data: pd.DataFrame, 
        init_funds = 100000, 
        # ensure_profit = True, 
        logger = True
        ):
        self.code = code
        self.data4cal = copy.deepcopy(data)  # 避免污染原始数据
        self.init_funds = init_funds
        self.current_funds = init_funds
        self.current_hands = 0

        self.last_buy_date = None
        self.trades_list = []
        self.slippage = SlippagePct()

        self.logger = Logger(self.code)

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
    # 不能加property
    def isFundsEnough(self, price):
        """
        有足够的钱，遇到buy_signal，就买入
        """
        if self._handsCanBought(self.current_funds, price) > 0:
            return True
        else:
            return False

    """
    current_funds：################
    current_hands：##########
    trades_list: bsbbssbbbb  当没有asize/bsize时，trades_list 可以保证为 bsbsbsbsbsb， 因为曾经我们的buy和sell 统一用一个开关 isholing控制，
                现在我们对于buy 和 sell 不统一控制，钱够+buy_signal就买入，有持仓+sell_signal 就卖，这样就不会bsbsbs了
                所以我们计算每笔交易收益的时候，如bbs，则需要平均bb的购买价格，且不能再使用手里的资金计算收益率
    
    data4cal["history_funds"]: append(current_funds),例：[(time1, current_funds1), (time2, current_funds2),...,(timen, current_fundsn)]
    data4cal["history_hands"]: append(current_hands),例：[(time1, current_hands1), (time2, current_hands2),...,(timen, current_handsn)]
    不在trade时更改，有了trades_list之后，def history_cal(self) 来获得

    """
    class P:
        ask1 = "ask1"
        bid1 = "bid1"
        close = "close"
    class Size:
        asize1 = "asize1"
        bsize1 = "bsize1"
    def trade(self, p1:P = P.ask1, p2:P = P.bid1, s1:Size = Size.asize1, s2:Size = Size.bsize1):
        """
        ************************主要作用：创建trades_list, 同时创建data4cal["history_funds]/data4cal["history_hands]
        :param p1:
        :param p2:
        :param s1:
        :param s2:
        :return:
        """
        history_funds_list = []
        history_hands_list = []
        delta_funds_list = [] #与原始资金无关的一个list，用于计算平均持仓成本

        for date, data in self.data4cal.iterrows():

            # 用delta_hands 代替 hands_to_trade, 用delta_funds 代替cost/earn，以统一买卖代码格式
            if self.isFundsEnough(data[p1]) and data["buy_signal"]:
                # 购买的持仓
                delta_hands = self.hands2trade(data[p1],data[s1],Toward.buy)
                # 消耗的资金
                delta_funds = delta_hands * 100 * data[p1] + self.commissionCal(data[p1],delta_hands,Toward.buy)
                # 记录在交易列表
                self.trades_list.append((date, "b"))
                # 更新持仓
                self.current_hands += delta_hands
                # 更新资金
                self.current_funds -= delta_funds
                # 记录上一次买仓时间
                self.last_buy_date = date
                # log
                if self.logger:
                    self.logger.info("Bought {} hands of {} on {} at {} price, cost {}, left {} CNY, hold {} hands".format(
                                   delta_hands, self.code, date, data[p1], delta_funds, self.current_funds, self.current_hands))
                # 记录持仓成本(开仓记录为正)
                delta_funds_list.append(delta_funds)
            # 避免出现同时买/卖的情况
            elif (self.isholding) and data["sell_signal"]:

                # 卖出的手数
                delta_hands = self.hands2trade(data[p2], data[s2], Toward.sell)
                # 赚取的资金
                delta_funds = delta_hands * 100 * data[p2] - self.commissionCal(data[p2], delta_hands, Toward.sell)
                # 记录在交易列表
                self.trades_list.append((date, "s"))
                # 更新持仓
                self.current_hands -= delta_hands
                # 更新资金
                self.current_funds += delta_funds
                if self.logger:
                    self.logger.info("Sold {} hands of {} on {} at {} price, profit {}, left {} CNY, hold {} hands".format(
                                        delta_hands, self.code, date, data[p2], delta_funds, self.current_funds, self.current_hands))
                # 记录持仓成本 (卖仓，记录为-值)
                delta_funds_list.append(-delta_funds)
            else:
                delta_funds_list.append(0)
            
            # 所以要不停的更新 self.current_funds 和 self.current_hands
            history_funds_list.append(self.current_funds)
            history_hands_list.append(self.current_hands)
        # 历史剩余资金
        self.data4cal["history_funds"] = history_funds_list
        # 历史剩余手数
        self.data4cal["history_hands"] = history_hands_list
        # 历史交易成本
        self.data4cal["delta_funds"] = delta_funds_list
    
    def sellSignal(self):
        raise NotImplementedError
    def buySignal(self):
        raise NotImplementedError
    def _init(self):
        self.data4cal["buy_signal"] = self.buySignal()
        self.data4cal["sell_signal"] = self.sellSignal()

if __name__ == "__main__":
    print("xx")
