import pandas as pd 
from StrategyBase import StrategyBase, Toward
from utilities import MACD

def modifydf4macd(df:pd.DataFrame):
    df.rename(columns={"代码":"code","日期":"trade_date","收盘价(元)":"close"}, inplace=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # df["trade_date"] = pd.to_numeric(df["trade_date"])
    df.set_index("trade_date", inplace=True)
    df4cal = df.loc["2018-05-01": "2019-04-30"]
    return df4cal[["code","close"]]

class StrategyMACD(StrategyBase):
    def __init__(self, code, data: pd.DataFrame, ensure_profit = True):
        super().__init__(code, data, ensure_profit)


    def _init_process(self):
        # self.data4cal.reset_index(inplace=True,drop=True)
        # self.data4cal.set_index("trade_date", inplace=True)


        macd_t = MACD(self.data4cal["close"])
        self.data4cal["DIF"] = macd_t[0]
        self.data4cal["DEA"] = macd_t[1]
        self.data4cal["MACD"] = macd_t[2]
        
        self.data4cal["buy"] = self.buy_condition()
        self.data4cal["sell"] = self.sell_condition()


    def buy_condition(self):
        """
        金叉买入
        :return: 是否是金叉的bool list
        """
        result = [False]
        for i in range(1, len(self.data4cal)):
            if (self.data4cal.iloc[i - 1]["DIF"] < self.data4cal.iloc[i]["DIF"] and \
                    self.data4cal.iloc[i - 1]["DEA"] < self.data4cal.iloc[i]["DEA"] and \
                    self.data4cal.iloc[i - 1]["DIF"] < self.data4cal.iloc[i - 1]["DEA"] and \
                    self.data4cal.iloc[i]["DIF"] > self.data4cal.iloc[i]["DEA"]):
                result.append(True)
            else:
                result.append(False)
        return result


    def sell_condition(self):
        """
        死叉卖出
        :return: 是否是死叉的bool list
        """
        result = [False]
        for i in range(1, len(self.data4cal)):
            if (self.data4cal.iloc[i - 1]["DIF"] > self.data4cal.iloc[i]["DIF"] and \
                    self.data4cal.iloc[i - 1]["DEA"] > self.data4cal.iloc[i]["DEA"] and \
                    self.data4cal.iloc[i - 1]["DIF"] > self.data4cal.iloc[i - 1]["DEA"] and \
                    self.data4cal.iloc[i]["DIF"] < self.data4cal.iloc[i]["DEA"]):
                result.append(True)
            else:
                result.append(False)
        return result

if __name__ == "__main__":
    stock1 = pd.read_csv("E:/data/000001.SZ.CSV", encoding = "gb18030")
    df = modifydf4macd(stock1)
    stock1_cal = StrategyMACD("000001.SZ", df, False)
    stock1_cal.run()