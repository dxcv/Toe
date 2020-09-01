import pandas as pd 
import datetime
import numpy as np 
from Alpha101_code_1 import get_alpha
from trade_date import *
from scipy.stats import pearsonr
from tqdm import tqdm
import copy

class FactorBase:
    """
    一次计算 只能筛选一次factor，利用start_date 和 end_date来计算start_date~endate之间的持有收益与factor之间的相关性
    若需要多次筛选factors，则要重复创建不同时间窗口的FactorBase或其子类
    要注意提前保证 self.start_date 和 self.end_date 是交易日
    """

    def __init__(self, df:pd.DataFrame, start_date, mid_date, end_date):
        self.init_data = copy.deepcopy(df)
        self.data4cal = pd.DataFrame()
    
        self.hs300 = pd.DataFrame()

        self.port_stk_num = 2
        self.interval = 20

        self.ports = []

        self.original_factors = []
        self.raw_factors = [] # corr 初步筛选有效
        self.refined_factors = [] # port1 excess/below hs300
        self.final_factors = []
        
        self.factor_corr_dict = {}
        self.adjust_frequency = None
        self.start_date = start_date
        self.mid_date = mid_date

        self.end_date = end_date
        
        i = 0
        for code, data in tqdm(self.init_data.groupby("code"), desc="进度"):
            if i > 9:
                break
            self.data4cal = pd.concat([self.data4cal, self.getFactors(data)], axis=0)
            i += 1
        
        self.data4cal.reset_index(inplace=True)
        self.data4cal.set_index(["trade_date", "code"], inplace=True)
        self.data4cal["S_DQ_CLOSE"] = df.reset_index().set_index(["trade_date", "code"])["S_DQ_CLOSE"]

        self.data4cal_code_trade_date = self.data4cal.reset_index().set_index(["code", "trade_date"]).sort_index()
        self.data4cal_trade_date = self.data4cal.reset_index().set_index("trade_date").sort_index()


    def getFactors(self, data):
        return get_alpha(data)
        #raise NotImplementedError
    
    # 此处保留port_stk_num, 以保证函数意义
    def getPorts(self, date, port_stk_num, factor_col):
        # 套个list，避免ports.append(Series)
        code_list = list(self.data4cal_trade_date.loc[date].sort_values(by=factor_col, ascending=False)["code"])
        ports = []
        split_num = len(code_list)//port_stk_num
        i = 0
        for i in range(split_num):
            ports.append(code_list[port_stk_num * i : port_stk_num * (i + 1)])
        if split_num * port_stk_num != len(code_list):
            ports.append(code_list[port_stk_num*(i+1) :])
        return ports

    def portReturn(self, port, date, interval, factor_col):

        # 在date日，根据factor选出股票，持有interval天后，获得的factor加权收益
        # 不要这个了太影响计算效率
        #df = self.data4cal.reset_index().set_index(["code", "trade_date"])
        df = self.data4cal_code_trade_date
        return_list = []
        factor_list = []
        for code in port:
            # self.data4cal的close 是存在问题的，需要整改
            rtn = df.loc[(code, get_next_trade_date(date, interval)), "S_DQ_CLOSE"][0] / df.loc[(code, date), "S_DQ_CLOSE"][0] - 1
            return_list.append(rtn)
            factor_list.append(df.loc[(code, date), factor_col][0])
            
        return_array = np.array(return_list)
        factor_array = abs(np.array(factor_list)) # abs 不能直接套在list外面
        port_weighted_return = (return_array * factor_array / factor_array.sum()).sum()
        return port_weighted_return

    def portsReturnList(self, ports, date, interval, factor_col):
        #ports index same day return list
        # 求解 ports,在date时interval时间后的return_list

        ports_weighted_return_list = []
        for port in ports:
            ports_weighted_return_list.append(self.portReturn(port, date, interval, factor_col))
        return ports_weighted_return_list       

    def portsReturnSeriesYearly(self, port_stk_num, start_date, mid_date, interval, factor_col):
        # 计算start_date, end_date 中间有多少个交易日, 那就应该包括start_date和 end_date
        total_trade_days = get_intermediate_trading_days(start_date, mid_date)
        # 按照interval间隔划分，可将交易日划分为多少段(向下取整，~end_date中间的部分不计算)
        interval_num = total_trade_days // interval
        # 计算每个段的 ports_return_list
        date = start_date
        # [[port1, port2, port3...], [port1, port2, port3...], []]
        #   date1                      date2                   date3       
        ports_return_list = [] 

        """
        |  |  |  |  |  |  |  |
           ———————  ———————
        1次      2次    not3次(因为interval超过)
        """
        for i in range(interval_num): 

            ports = self.getPorts(date, port_stk_num, factor_col)
            
            # 不只要考虑date不超过 end_date, 还要考虑 date + interval不超过，因为我们要求的是interval后的收益率，所以
            # date + interval 也不允许超过
            ports_return_list.append(self.portsReturnList(ports, date, interval, factor_col))
            date = get_next_trade_date(date, interval)

        # 将所有的ports_return_list equal-weighted平均
        """
                port1 port2 port3 ...
        date1
        date2
        """
        df4mean = pd.DataFrame(ports_return_list)
        return df4mean.mean(axis=0)

    def factorsPortsCorrCal(self, port_stk_num, start_date, mid_date, factor_col):

        # df = df.reset_index().set_index("trade_date").sort_index()
        if self.data4cal[factor_col].isnull().all():
            return 0,0
        if self.data4cal[factor_col].dtype == "bool":
            return 0,0
        ports_wa_yearly_return_array = np.array(self.portsReturnSeriesYearly(
            port_stk_num, start_date, mid_date, self.interval, factor_col))
        
        # 获取逆序的index
        index = np.array([len(ports_wa_yearly_return_array) - i for i in range(len(ports_wa_yearly_return_array))])
        # 获取非nan的index
        index = index[~np.isnan(ports_wa_yearly_return_array)]
        
        # 获取非nan的值
        ports_wa_yearly_return_array = ports_wa_yearly_return_array[~np.isnan(ports_wa_yearly_return_array)]
        if len(ports_wa_yearly_return_array) < 2: # 若列表的长度达不到2，则无法求相关系数
            return 0,0
        corr = pearsonr(index, ports_wa_yearly_return_array)#[0]
        return corr

    def readHs300(self, filepath):
        self.hs300 = pd.read_excel(filepath)
        self.hs300["return_daily"] = self.hs300["close"] / self.hs300["close"].shift(1) - 1
        self.hs300 = self.hs300.reset_index().set_index("trade_date").loc[self.start_date:self.end_date]

    # 需要确保 start_date, end_date是交易日，没有做非交易日的处理
    def probExcessHs300(self, port, start_date, mid_date, factor_col):
        """
        start_date ~ end_date 之间的每一天(date)，若因子的当日的加权收益超过hs300，则记录当天为超过hs300，否则为没超过
        概率 = 超过的天数/总交易天数
        """
        end_date_modify = get_next_trade_date(mid_date, - self.interval)
        # hs300_df 索引的时候，要减少interval的天数
        hs300_df = self.hs300.loc[start_date: end_date_modify]
        port_return_list = []
        for date in trade_date_range(start_date, end_date_modify):
            
            # 如果全部遍历 start_date, end_date，那这个20会导致超范围
            port_return_list.append(self.portReturn(port, date, self.interval, factor_col))
#         if len(port_return_list) != len(hs300_df): 废弃，因为我们end_date_modify了
#             raise ValueError

        tmp_array = np.array(port_return_list) - np.array(hs300_df["return_daily"])
        gt_num = len(tmp_array[tmp_array > 0])
        return gt_num/len(port_return_list)
    
        
    # 保留了start_date, end_date作为函数形参，没有用self写死在函数内部，这样可读性会稍微好一些
    def probBelowHs300(self, port, start_date, mid_date, factor_col):
        end_date_modify = get_next_trade_date(mid_date, -self.interval)
        # hs300_df 索引的时候，要减少interval的天数
        hs300_df = self.hs300.loc[start_date: end_date_modify]
        port_return_list = []
        for date in trade_date_range(start_date, end_date_modify):
            
            # 如果全部遍历 start_date, end_date，那这个20会导致超范围
            port_return_list.append(self.portReturn(port, date, self.interval, factor_col))
#         if len(port_return_list) != len(hs300_df):
#             raise ValueError

        tmp_array = np.array(port_return_list) - np.array(hs300_df["return_daily"])
        lt_num = len(tmp_array[tmp_array < 0])
        return lt_num/len(port_return_list)


    def setFactors(self):
        #raise NotImplementedError
        self.original_factors =  ['alpha001', 'alpha002',
       'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha007', 'alpha008',
       'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014',
       'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020',
       'alpha021', 'alpha022', 'alpha023', 'alpha024', 'alpha025', 'alpha026',
       'alpha027', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032',
       'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038',
       'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044',
       'alpha045', 'alpha046', 'alpha047', 'alpha049', 'alpha050', 'alpha051',
       'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha057', 'alpha060',
       'alpha061', 'alpha062', 'alpha064', 'alpha065', 'alpha066', 'alpha068',
       'alpha071', 'alpha072', 'alpha073', 'alpha074', 'alpha075', 'alpha077',
       'alpha078', 'alpha081', 'alpha083', 'alpha084', 'alpha085', 'alpha086',
       'alpha088', 'alpha092', 'alpha094', 'alpha095', 'alpha096', 'alpha098',
       'alpha099', 'alpha101']

    # 通过next_date/date - 1的收益率来计算rawFactors
    # 后续将0.5这个阈值设置为 形参，取消掉port_stk_num, start_date, end_date, interval,df
    def rawFactors(self, threshold = 0.5):
        self.raw_factors = []
        for factor in tqdm(self.original_factors):
            corr = self.factorsPortsCorrCal(self.port_stk_num, self.start_date, self.mid_date, factor)
            self.factor_corr_dict[factor] = corr
            if abs(corr[0]) > threshold:
                self.raw_factors.append(factor)
        return self.raw_factors

    # 这里显然不应再输入port_stk_num, 后续处理掉
    # 还是 threshhold
    def refineFactors(self, threshhold = 0.5):
        self.refined_factors = []
        for factor in tqdm(self.raw_factors):
            ports = self.getPorts(self.start_date, self.port_stk_num, factor)

            if self.factor_corr_dict[factor][0] > 0:
                # start_date计算出后续需要持有的股票后，在持有期间计算大因子port跑赢hs300的概率(若正相关)
                if self.probExcessHs300(ports[0], self.start_date, self.mid_date, factor) > threshhold:
                    self.refined_factors.append(factor)
            else:
                # start_date 计算出后续需要持有的股票后，在持有期间计算大因子port跑输hs300的概率(若负相关)
                if self.probBelowHs300(ports[0], self.start_date, self.mid_date, factor) > threshhold:
                    self.refined_factors.append(factor)
            # 不考虑小因子的股票跑赢/跑输了，因为小因子可能会出现Nan效果不好

    def getMaxReturnFactor(self, factor_list):
        # 用第一个元素，不用0作为初始的max_return了
        max_factor = factor_list[0]
        if self.factor_corr_dict[max_factor][0] > 0:
            max_return = self.portsReturnSeriesYearly(self.port_stk_num, self.start_date, self.mid_date, self.interval, factor_list[0]).iloc[0]
        else:
            max_return = self.portsReturnSeriesYearly(self.port_stk_num, self.start_date, self.mid_date, self.interval, factor_list[0]).iloc[-1]
        
        # 不用 range(1, len(factor_list), 1)了, 不然调用不太方便
        for factor in factor_list:

            tmp_return_list = self.portsReturnSeriesYearly(self.port_stk_num, self.start_date, self.mid_date, self.interval, factor)
            # 不能用tmp_return, 得用port1的return， portsWaYearlyReturnList是 [port1_return, port2_return, ...]
            # 不声明tmp_return会怎样
            if self.factor_corr_dict[factor][0] > 0:
                tmp_return = tmp_return_list.iloc[0]
            else:
                tmp_return = tmp_return_list.iloc[-1]
            
            if tmp_return > max_return:
                max_return = tmp_return
                max_factor = factor
        return max_factor            
            

    def corrFilteredFactors(self, threshold, raw_factors):
        # factor_dict 中不一定包含所有的raw_factors的，可能有的factor就是和谁都不相关
        # 调取raw_factor对应的 df
        corrdf = self.data4cal.reset_index().set_index("trade_date").loc[self.start_date:self.mid_date, raw_factors].corr()
        corrdf_bool = (corrdf > threshold) & (corrdf < 1)
        factor_dict = {}
        for i, row in corrdf_bool.iterrows():
            # 如果这行有相关的col
            if row.any():
                # 将相关的col 存在一起
                col_list = []
                for col_idx in row.index:
                    if row[col_idx]:
                        col_list.append(col_idx)
                factor_dict[i] = col_list
            # 如果没有和他相关的就把他写作key存储
            else:
                factor_dict[i] = []
        # 如果不自己定义ks，那for i in factor_dict.keys()循环的时候del其中的元素，会影响到整个循环体
        ks = list(factor_dict.keys())
        for k in ks:
            # 如果k是factor_dict的一个key
            if k in factor_dict.keys():
                # 遍历v中的factor
                for factor in factor_dict[k]:
                    # 如果factor在factor_dict的keys中
                    if factor in factor_dict.keys():
                        # 删除 这个key-value
                        del factor_dict[factor]

        def turn2list_list(dct):
            result  = []
            buffer = []
            for k in dct.keys():
                buffer = [k]
                for v in dct[k]:
                    buffer.append(v)
                result.append(buffer)
            return result
        factor_list_list = turn2list_list(factor_dict)
        
        self.final_factors = []
        for factor_list in factor_list_list:
            self.final_factors.append(self.getMaxReturnFactor(factor_list))

        return self.final_factors


    # 筛选的start_end 和 选股的start_end 不一致
    def getAlpha4rank(self):
        if "factor4rank" not in self.data4cal.columns:
            # 首先应该归一化，然后再sum
            def z_score(x):
                if self.factor_corr_dict[x.name][0] > 0: # 正相关时直接归一化
                    return (x - x.mean())/x.std() #要有return才能用到apply里面，且只能是df.apply才可以
                else:
                    return -(x - x.mean())/x.std() # 负相关时取相反数
            self.data4cal_trade_date["factor4rank"] = self.data4cal_trade_date[self.final_factors].apply(z_score, axis=0).sum(axis=1)


    # 需要有date，以创建截面数据，否则无法排序获得ports
    def getHolding(self, date):
        
        ports = self.getPorts(date, self.port_stk_num, "factor4rank")
        return ports[0]
    
    
    def getOrderList(self, mid_date, end_date):
        count = 1
        code_set = set(self.getHolding(mid_date))
        trade_dict = {
            "Toward":"buy",
            "datetime":mid_date,
            "code_set":code_set
        }
        trade_list = [trade_dict]
        for date in trade_date_range(mid_date, end_date):
            if count > self.interval:
                count = 1
                code_set2 = set(self.getHolding(date))
                if (code_set2 == code_set):
                    continue
                # _________________________________________    
                #TODO，code_set2 和 code_set1 存在交集的处理
                #___________________________________________
                
                trade_dict = {
                    "Toward" : "sell",
                    "datetime":date,
                    "code_set":code_set
                }
                trade_list.append(trade_dict)
                code_set = code_set2
                trade_dict = {
                    "Toward" : "buy",
                    "datetime":date,
                    "code_set":code_set
                }
                trade_list.append(trade_dict)
            else:
                count += 1
        return trade_list

    def run(self):
        # hs300_code_list = pd.read_csv("E:/data_all/hs300_codelist.csv")
        # hs300_code_refine = []
        # all_code = self.data4cal.reset_index().set_index("code").index.unique()
        # for code in hs300_code_list["code"]:
        #     if code in all_code:
        #         hs300_code_refine.append(code)
        self.setFactors()
        self.rawFactors()

        self.readHs300("E:/data_all/hs300_index.xlsx")
        self.refineFactors()
        self.corrFilteredFactors(0.5, self.refined_factors)
        # 去除 final_factors 中的空字符串
        buffer = []
        for factor in self.final_factors:
            if factor:
                buffer.append(factor)
        self.final_factors = buffer
        print(self.final_factors)

        self.getAlpha4rank()
        return self.getOrderList(self.mid_date, self.end_date)



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
    return df.loc["2017-05-01":"2019-04-30", ["code", "S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE", "S_DQ_VOLUME", "S_DQ_AMOUNT", "S_DQ_PCTCHANGE"]]
 
