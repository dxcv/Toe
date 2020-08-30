import os
import pandas as pd



class PreExecutor:

    def __init__(self, filename):
        self.filename = self._genfilename(filename)
        self.df = pd.read_csv(self.filename, encoding="gb18030", sep=",")

    def setDF(self, df:pd.DataFrame):
        self.df = df

    def _genfilename(self, filename):
        return "E:/data/" + filename + ".csv"
    
    def modify4macd(self):
        self.df.rename(columns={"日期":"trade_date", "收盘价(元)":"close"}, inplace=True)
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.df.set_index("trade_date", inplace=True)
        
        self.df = self.df[["close"]]
    
    def modify4macdRT_tmp(self):
        self.df.rename(columns={"日期":"trade_date", "收盘价(元)":"close"}, inplace=True)
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.df.set_index("trade_date", inplace=True)

        self.df[""]
        
        self.df = self.df[["close"]]
    def modify4alpha101(self):
        self.df.rename(columns={"代码":"code","简称":"short_name","日期":"date","前收盘价(元)":"pre_close","开盘价(元)":"S_DQ_OPEN","最高价(元)":"S_DQ_HIGH",
                       "最低价(元)":"S_DQ_LOW","收盘价(元)":"S_DQ_CLOSE","成交量(股)":"S_DQ_VOLUME","成交金额(元)":"S_DQ_AMOUNT"
                      ,"涨跌(元)":"change","涨跌幅(%)":"S_DQ_PCTCHANGE","均价(元)":"S_DQ_AVEPRICE"}, inplace=True)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index("date", inplace=True)
        return self.df.loc["2018-05-01":"2019-04-30", ["code", "S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE", "S_DQ_VOLUME", "S_DQ_AMOUNT", "S_DQ_PCTCHANGE"]]


if __name__ == "__main__":
    pre = PreExecutor("000001.SZ")
    pre.modify4macd()
    print(pre.df)
    
    
