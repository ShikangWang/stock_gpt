import baostock as bs
import pandas as pd
from tqdm import tqdm

# 登录系统
lg = bs.login()

# 显示登录返回信息
print(f'login respond error_code: {lg.error_code}')
print(f'login respond error_msg: {lg.error_msg}')

start_date = '2008-09-01' # 开始日期
end_date = '2024-09-02'   # 结束日期

#获取所有股票代码
rs_start = bs.query_all_stock(day=start_date)
data_list_start = []
while (rs_start.error_code == '0') & rs_start.next():
    # 获取一条记录，将记录合并在一起
    data_list_start.append(rs_start.get_row_data())

rs_end = bs.query_all_stock(day=end_date)
data_list_end = []
while (rs_end.error_code == '0') & rs_end.next():
    # 获取一条记录，将记录合并在一起
    data_list_end.append(rs_end.get_row_data())

#获取交集
stock_list = list(set([item[0] for item in data_list_start]) & set([item[0] for item in data_list_end]))


for stock_code in tqdm(stock_list):
    # 获取股票的日线数据
    rs = bs.query_history_k_data_plus(stock_code,
                                    "date,code,open,high,low,close,preclose,volume,amount,pctChg",
                                    start_date=start_date, end_date=end_date,
                                    frequency="d", adjustflag="3")

    # 检查数据返回状态
    print(f'query history k data respond error_code: {rs.error_code}')
    print(f'query history k data respond error_msg: {rs.error_msg}')

    # 将数据存入DataFrame
    data_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一行记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)

    # save to csv
    df.to_csv(f'./datasets/original/{stock_code}.csv', index=False)

# 登出系统
bs.logout()