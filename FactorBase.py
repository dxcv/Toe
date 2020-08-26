import pandas as pd 
import datetime
import numpy as np 


class FactorBase:
    def __init__(self, df:pd.Dataframe, start_date, end_date):
        self.init_data = df
        self.data4cal = pd.Dataframe()
        self.ports = []

        self.raw_factors = [] # corr 初步筛选有效
        self.refined_factors = [] # port1 excess/below hs300

        self.adjust_frequency = None
        self.start_date = start_date
        self.end_date = end_date
    
    def init(self):
        for code, data in self.init_data.groupby("code"):
            pd.concat([self.data4cal, self.getFactors(data)], axis=0)
        # self.
        # for code, data in self.init_data.groupby("code"):
            # alpha101_dict 只是为了concat的一个中间缓解
    def getFactors(self, data):
        raise NotImplementedError

    def codeList(self):
        pass

    def splitIntoPorts(self, split_num):
        pass

    def weightedReturnSameDay(self):
        pass

    def weightedReturnBetweenDays(self):
        # 是用未来的收益率来计算相关系数吗？
        # 然后假定因子仍有效，从下个月开始持有，直到下下个月？
        pass

    def factors_ports_corrCal(self):
        pass
    
    def probExcessHs300(self):
        pass
    
    def probBelowHs300(self):
        pass

    def rawFactors(self):
        pass

    def refineFactors(self):
        pass
    
    def getHolding(self):
        pass

    def changeHolding(self):
        pass
    

