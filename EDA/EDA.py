#-----------------------------------------------------------------------------#
'''導入Module'''
import numpy as np
import pandas as pd
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
'''讀取CSV檔案並轉換成DataFrame資料型態進行資料處理'''
DataSet = pd.read_csv("AER_credit_card_data.csv")
df = pd.DataFrame(DataSet)

#-----------------------------------------------------------------------------#
'''將原DataSet欄位標題轉為中文'''
df.rename(columns={"card": "核卡狀況",#
                   "reports": "聯徵紀錄",#
                   "age": "年齡",
                   "income": "年收入",
                   "share": "收支比",
                   "expenditure": "每月信用卡支出平均",
                   "owner": "房地產",
                   "selfemp": "創業人士",
                   "dependents": "扶養人數",
                   "months": "居住時間",
                   "majorcards": "持有主卡",
                   "active": "活躍帳戶",
},inplace=True)
#-----------------------------------------------------------------------------#
'''進行資料處理工程'''
df['居住時間'] = df['居住時間'] // 12
df["年齡"] = df["年齡"].astype(int)
df["年收入"] = df["年收入"]*10
df.drop(df[df["年齡"]<15].index, inplace=True)
#-----------------------------------------------------------------------------#
'''One-Hot-Encoding'''
Card = pd.get_dummies(df["核卡狀況"], prefix="核卡狀況", dtype=int)
Owner = pd.get_dummies(df["房地產"], prefix="房地產", dtype=int)
Selfemp = pd.get_dummies(df["創業人士"], prefix="創業人士", dtype=int)
#-----------------------------------------------------------------------------#
'''Z-score正規化'''
reportZ = (df["聯徵紀錄"]-df["聯徵紀錄"].mean()) / df["聯徵紀錄"].std()
df["聯徵紀錄"] = reportZ
incomeZ = (df["年收入"]-df["年收入"].mean()) / df["年收入"].std()
df["年收入"] = incomeZ
shareZ = (df["收支比"]-df["收支比"].mean()) / df["收支比"].std()
df["收支比"] = shareZ
expendZ = (df["每月信用卡支出平均"]-df["每月信用卡支出平均"].mean()) / df["每月信用卡支出平均"].std()
df["每月信用卡支出平均"] = expendZ
#-----------------------------------------------------------------------------#
'''合併處理完的資料'''
df = pd.concat([df ,Card, Owner, Selfemp], axis=1)
df.drop(columns=["房地產", "創業人士"], inplace=True)
#-----------------------------------------------------------------------------#
'''將原DF資料輸出成New_DataSet.csv'''
TrainData = df
TrainData.to_csv('New_DataSet.csv', index=False)

















































