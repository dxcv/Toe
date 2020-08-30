import pandas as pd 
import copy
from trade_date import *

idx = pd.IndexSlice

class Toward:
    buy = "buy"
    sell = "sell"

class Account:
    def __init__(self):
        self.init_funds = 100000000

        self.current_position = {} # {"000001.SZ": 2手, "000002.SZ": 10手}
        self.current_funds = self.init_funds
        # 不要这个 self.data4cal了， 在Trader中获取price 

        """
        if (code, trade_date) in self.history_position.index:
            self.history_position.loc
                                   hands
        trade_date      code
        "2018-05-18"  "000001.SZ"   1
        "2018-05-18"  "000002.SZ"   3

        考虑应用双指针 遍历
        df["value"] = row["price"][self.history_position.loc["2018-05-18", "code"]] * self.history_position.loc["2018-05-18", "volumne"]
        """

        # self.history_position 和 self.history_funds 的 keys是一致的
        self.history_position = {} #pd.DataFrame(columns=["code", "trade_date", "hands"]).set_index(["code", "trade_date"]) #code要在前
        self.history_funds = {} # {"2018-05-18":100000, }
    
    # 在buy外面判断 资金是否充裕
    def buy(self, code, price, volume, date):
        commission = self.commission_cal(price, volume, Toward.buy)
        # 消耗的资金
        delta_funds = volume * 100 * price + commission

        if self.current_funds < delta_funds:
            raise ValueError

        if code in self.current_position:
            self.current_position[code] += volume
        else:
            self.current_position[code] = volume
        # multiindex的第一个index， 可以通过in的方式判断
        # if code in self.history_position.index:
        #     self.history_position.loc[(code, date), "hands"] = volume + self.history_position.loc[code, "hands"].iloc[-1] #volume + 上一个交易日的code持仓
        # else:
        #     self.history_position.loc[(code, date), "hands"] = volume
        self.current_funds -= delta_funds
        # !!!!!!!!!!!!!!!!必须得是 copy.deepcopy，不然因为都是引用，最终每一天都是一样了
        self.history_position[date] = copy.deepcopy(self.current_position)
        self.history_funds[date] = self.current_funds

    def sell(self, code, price, volume, date):
        # 持仓判断
        if code in self.current_position:
            # 改为在内部判断持仓充裕与否
            # 1. 持仓 < 交易量的时候 报错
            if self.current_position[code] < volume:
                raise ValueError
            # 2. 持仓 = 交易量的时候 删除
            elif self.current_position[code] == volume:
                del self.current_position[code]
            # 3. 剩下就正常交易
            else:
                self.current_position[code] -= volume
        else:
            raise ValueError

        commission = self.commission_cal(price, volume, Toward.sell)
        # 赚取的资金
        delta_funds = volume * 100 * price - commission
        self.current_funds += delta_funds
        self.history_position[date] = copy.deepcopy(self.current_position)
        self.history_funds[date] = self.current_funds

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
            raise ValueError
        return transfer_fee + commission_tax + commission_broker


    
class Trader:
    """
    trade_list 实际是一个 time_list, 具体code_list中每一只股票买多少支的逻辑在Trader里实现
    trade_list = [
        {
            Toward: "buy",
            datetime: "2018-05-18 00:00:00",
            code_set: {"000001.SZ", "000002.SZ"}
        },
    ]
    value: 股票价值
    equity: 股票 + 钱
    funds: 钱
    """


    def __init__(self, df:pd.DataFrame):
        self.data4cal = df
        self.order_list = []
        self.start_date = ""#self.trade_list[0]["datetime"]  #"2018-05-18"
        self.end_date = ""#self.trade_list[-1]["datetime"]   #"2019-04-12"

        self.account = Account()


    def init(self, order_list):
        self.order_list = order_list
        self.data4cal = self.data4cal.reset_index().set_index(["code", "trade_date"]).sort_index()
        self.start_date = self.order_list[0]["datetime"]
        self.end_date = self.order_list[-1]["datetime"]

    def readTradeList(self, trade_list):
        self.order_list = trade_list

    def trade(self):
        # 需要用trade这个步骤来获得 history_position, 不能用 self.trade_list的，因为这里面买点/卖点都有，无法保证持仓，且持仓数量未知
        for order in self.order_list:
            if order["Toward"] == Toward.buy:
                for code in order["code_set"]:
                    self.account.buy(code, self.data4cal.loc[(code, order["datetime"]), "S_DQ_CLOSE"][0], 1, order["datetime"]) # 暂时买一手
            elif order["Toward"] == Toward.sell:
                for code in order["code_set"]:
                    self.account.sell(code, self.data4cal.loc[(code, order["datetime"]), "S_DQ_CLOSE"][0], 1, order["datetime"]) # 暂时卖一手
    
    @property
    def acc_value_Series(self):

        current_position_Series = None
        history_position_dict_dict = self.account.history_position

        result = {}
        for date in trade_date_range(self.start_date, self.end_date):
            # 这里的实现我应该记一下，这样就没用到双指针了
            if date in history_position_dict_dict.keys():
                current_position_Series = pd.Series(history_position_dict_dict[date])

            # 不能直接两个Series相乘，因为一个的index是code，另一个是[code, trade_date]
            # position使用current_position_Series.index来索引就可以保证code是对应的了
            position_value = (self.data4cal.loc[idx[current_position_Series.index, date], "S_DQ_CLOSE"].values * current_position_Series.values).sum()
            result[date] = position_value
        return pd.Series(result)

    @property
    def acc_funds_Series(self):

        history_funds_dict = self.account.history_funds
        result = {}
        current_funds = self.account.init_funds
        for date in trade_date_range(self.start_date, self.end_date):
            if date in history_funds_dict.keys():
                current_funds = history_funds_dict[date]
            result[date] = current_funds
        return pd.Series(result)

    @property
    def acc_equity_Series(self):
        return self.acc_funds_Series + self.acc_value_Series

def modify4alpha101(df):
    df.rename(columns={"代码":"code","简称":"short_name","日期":"trade_date","前收盘价(元)":"pre_close",
                            "开盘价(元)":"S_DQ_OPEN","最高价(元)":"S_DQ_HIGH","最低价(元)":"S_DQ_LOW",
                            "收盘价(元)":"S_DQ_CLOSE","成交量(股)":"S_DQ_VOLUME","成交金额(元)":"S_DQ_AMOUNT"
                            ,"涨跌(元)":"change","涨跌幅(%)":"S_DQ_PCTCHANGE","均价(元)":"S_DQ_AVEPRICE"}, inplace=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.set_index("trade_date", inplace=True)
    df["S_DQ_CLOSE"].fillna(value=df["S_DQ_LOW"], inplace=True)
    df["S_DQ_HIGH"].fillna(value=df[["S_DQ_OPEN", "S_DQ_CLOSE"]].max(axis=1), inplace=True)
    df["S_DQ_OPEN"].fillna(value=df["S_DQ_LOW"], inplace=True)
    # 临时这么处理, 处理过后所有的alpha 都非空了
    df["S_DQ_VOLUME"].fillna(method='ffill', inplace=True)
    df["S_DQ_AMOUNT"].fillna(value = df["S_DQ_VOLUME"] * df["S_DQ_CLOSE"], inplace=True)
    
    df["S_DQ_PCTCHANGE"].fillna(value = (df["S_DQ_CLOSE"]/df["S_DQ_CLOSE"].shift(1) - 1), inplace=True)
    # 对于全市场数据就不要返回 tuple
    return df.loc["2018-05-01":"2019-04-30", ["code", "S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE", "S_DQ_VOLUME", "S_DQ_AMOUNT", "S_DQ_PCTCHANGE"]]
 
