from datetime import date
from WindPy import w
import pandas as pd
import numpy as np
import os
import concurrent.futures
import time
import datetime

w.start()
w.isconnected()

def get_tick(code, start_time, end_time, dirpath):
    try:
        data_wind = w.wst(code, 
        "pre_close,open,high,low,last,ask,bid,volume,amt,vol_ratio,iopv,limit_up,limit_down,ask5,ask4,ask3,ask2,ask1,bid1,bid2,bid3,bid4,bid5, asize5,asize4,asize3,asize2,asize1,bsize1,bsize2,bsize3,bsize4,bsize5", 
        start_time, 
        end_time, 
        "")
        arr = np.array(data_wind.Data)
        df = pd.DataFrame(arr.T, index=data_wind.Times, columns=data_wind.Fields)
        df["code"] = data_wind.Codes[0]
        df.index.name = "datetime"
        filepath = dirpath + code  + ".csv"
        df.to_csv(filepath, encoding="utf_8_sig")
    except:
        print("{} is not insert".format(code))
    return code


today = str(datetime.date.today())
dir_name = "E:/data/tick/2020/" + today + "/"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
start_time = today + " 09:00:00"
end_time = today + " 15:01:00" 

code_df = pd.read_csv("E:data/code.csv", encoding="gb18030")
code_list = list(code_df["WindCodes"])


begin = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor() as executor:
    result = [executor.submit(get_tick, code, start_time, end_time, dir_name) for code in code_list[0:10]]

end = time.perf_counter()
print(end - begin)

w.stop()


