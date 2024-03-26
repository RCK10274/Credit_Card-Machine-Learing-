#-----------------------------------------------------------------------------#
'''導入Module'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#-----------------------------------------------------------------------------#
'''讀取CSV檔案'''
DataSet = pd.read_csv("AER_credit_card_data.csv")
df = pd.DataFrame(DataSet)
#-----------------------------------------------------------------------------#
'''資料處理'''
df.drop(df[df["age"] <15].index, inplace=True)
df["age"] = df["age"].astype(int)
df["income"] = df["income"] * 10000
df["months"] = df["months"] // 12
df['share'] = df['share']*100
#-----------------------------------------------------------------------------#
'''matplotlib全域中文字體設定'''
plt.rcParams['font.sans-serif'] = ['SimSun']
#-----------------------------------------------------------------------------#
'''核卡核准比率 - 圓餅圖'''
# sizes = df["card"].value_counts()
# labels = sizes.index.tolist()

# plt.pie(sizes, colors=['gold', 'lightcoral'], autopct='%1.1f%%')
# plt.title("信用卡核准比例")
# plt.legend(loc="upper left", labels=["核准通過", "核准不通過"])
# plt.savefig("信用卡核准比例-圓餅圖")
#-----------------------------------------------------------------------------#
'''年齡分布 - 箱型圖及長條圖'''
# age_median = df["age"].median()       #中位數:31歲
# age_mode = df["age"].mode().values[0] #眾數:28歲
# age_mean = df["age"].mean()           #平均數33.38歲
# age_Q1 = df['age'].quantile(0.25)     #Q1: 25歲
# age_Q2 = df['age'].quantile(0.5)      #Q2: 31歲
# age_Q3 = df['age'].quantile(0.75)     #Q3: 39歲

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(x=df["age"], color="skyblue", orient="h")
# plt.xlabel("年齡")
# plt.xticks([20, 40, 60, 80])

# plt.title("年齡-箱型圖")
# plt.grid(axis='x')

# plt.subplot(1, 2, 2)
# sns.histplot(df["age"][df["card"]=='yes'], bins=20, color="skyblue")
# plt.xlabel("年齡")
# plt.ylabel("人數")
# plt.xticks([20, 25, 30, 35, 40, 45, 50])
# plt.yticks([0, 50, 100, 150])
# plt.title("年齡直方圖")
# plt.grid(axis='y')
# plt.savefig("年齡分布 - 箱型圖及直方圖")
#-----------------------------------------------------------------------------#
'''年齡與核卡核准率 - 交叉分析長條圖'''
# df["年齡分組"] = pd.cut(df["age"], bins=[17, 29, 39, 49, 59, 69, 99], labels=["18~29歲", "30~39歲", "40~49歲","50~59歲", "60~69歲", "70歲以上"])
# No_counts = df[df['card'] == 'no']['年齡分組'].value_counts()
# Yes_counts = df[df['card'] == 'yes']['年齡分組'].value_counts()

# bar_1 = np.arange(len(No_counts))
# bar_2 = bar_1 + 0.35

# plt.figure(figsize=(10, 6))

# plt.bar(bar_1, No_counts, color='skyblue', width=0.35, label='No')
# plt.bar(bar_2, Yes_counts, color='lightcoral', width=0.35, label='Yes')
# plt.xticks(bar_1 + 0.35 / 2, No_counts.index)

# plt.xlabel('年齡')
# plt.ylabel('人數')
# plt.title('年齡&信用卡核准率 - 交叉分析圖')
# plt.grid(axis='y')
# plt.legend()
# plt.savefig("年齡&信用卡核准率 - 交叉分析圖")
#-----------------------------------------------------------------------------#
'''年齡與核卡結果關係 - 長條圖'''
# Age_total = df["年齡分組"].value_counts()
# Card_yes = df[df['card'] == 'yes']['年齡分組'].value_counts()

# yes_rates = (Card_yes / Age_total) * 100

# plt.figure(figsize=(10, 6))
# plt.bar(yes_rates.index, yes_rates.values, width=0.5, color='red')

# plt.xlabel('年齡分布')
# plt.ylabel('核准率 (%)')
# plt.title('各年齡區間&信用卡核准率-長條圖')
# plt.grid(axis='y')
# plt.savefig("各年齡區間&信用卡核准率-長條圖")
# plt.show()
#-----------------------------------------------------------------------------#
'''年收入 - 箱型圖及分布圖'''
# income_mean = df["income"].mean()       #33.6
# income_mode = df["income"].mode()[0]    #30
# income_median = df["income"].median()   #29
# income_Q1 = (df['income']/1000).quantile(0.25)     #Q1: 22.3
# income_Q2 = (df['income']/1000).quantile(0.5)      #Q2: 29.0
# income_Q3 = (df['income']/1000).quantile(0.75)     #Q3: 49.0

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(df["income"]/10000, color="skyblue", orient="h")
# plt.xlabel("年收入")
# plt.xticks([0, 5, 10])
# plt.title("年收入-箱型图")
# plt.grid(axis='y')

# plt.subplot(1, 2, 2)
# plt.hist((df["income"]/1000)[df["card"]=="yes"], bins=30, color="skyblue", width=4.5)
# plt.xlabel("年收入")
# plt.ylabel("人數")
# plt.xticks([20, 40, 60, 80])
# plt.title("年收入直方圖")
# plt.grid(axis='y')
# plt.savefig("年收入-箱型圖及直方圖")
#-----------------------------------------------------------------------------#
'''年收入與核卡結果關係 - 交叉分析長條圖'''
df["年收分組"] = pd.cut(df["income"]/1000, bins=[20, 39, 59, 79, np.inf], labels=["20~39K", "40~59K", "60K~79K","80K以上"])

# No_counts = df[df['card'] == 'no']['年收分組'].value_counts()
# Yes_counts = df[df['card'] == 'yes']['年收分組'].value_counts()

# bar_1 = np.arange(len(No_counts))
# bar_2 = bar_1 + 0.35

# plt.figure(figsize=(10, 6))
# plt.bar(bar_1, No_counts, color='skyblue', width=0.35, label='核准未通過')
# plt.bar(bar_2, Yes_counts, color='lightcoral', width=0.35, label='核准通過')
# plt.xticks(bar_1 + 0.35 / 2, No_counts.index)
# plt.grid(axis='y')
# plt.xlabel('年收入(單位:千)')
# plt.ylabel('人數')
# plt.title('年收入與核準結果關係-交叉分析長條圖')
# plt.legend()
# plt.savefig("年收入與核準結果關係-交叉分析長條圖")
#-----------------------------------------------------------------------------#
'''年收入與核卡結果關係 - 長條圖'''
# income_total = df["年收分組"].value_counts()
# Card_yes = df[df['card'] == 'yes']['年收分組'].value_counts()
# df['income'] = df['income']/1000

# yes_rates = (Card_yes / income_total) * 100

# plt.figure(figsize=(10, 6))
# plt.bar(yes_rates.index, yes_rates.values, width=0.5, color='red')

# plt.xlabel('年收入(單位:千)')
# plt.ylabel('核卡成功率 (%)')
# plt.title('年收入&信用卡核准率 - 長條圖')
# plt.yticks([20, 40, 60, 80])
# plt.grid(axis='y')
# plt.savefig("年收入&信用卡核准率 - 長條圖")
#-----------------------------------------------------------------------------#
'''租屋族與買房族'''
# Owner_counts = df['owner'].value_counts()

# plt.pie(Owner_counts, autopct='%1.f%%', colors=['lightcoral', 'yellow'], startangle=249)
# plt.title("租屋族與買房族人數比例 - 圓餅圖")
# plt.axis('equal')
# plt.legend(loc="upper right", labels=["租屋族", "購屋族"])
# plt.savefig("租屋族與買房族人數比例 - 圓餅圖")
#-----------------------------------------------------------------------------#
'''租屋族群&信用卡核准率-長條圖'''
# Owner_total = df["owner"].value_counts()
# Card_yes = df[df['card'] == 'yes']['owner'].value_counts()
# yes_rates = (Card_yes / Owner_total) * 100

# plt.bar(["租屋族", "買屋族"], yes_rates.values, width=0.5, color='red')
# plt.xlabel('租屋族群')
# plt.ylabel('核卡成功率 (%)')
# plt.title('租屋族群&信用卡核准率-長條圖')
# plt.yticks([20, 40, 60, 80])
# plt.tight_layout()
# plt.grid(axis='y')
# plt.savefig("租屋族群&信用卡核准率-長條圖")
#-----------------------------------------------------------------------------#
'''創業人士及雇員的申辦者比例 - 圓餅圖'''
# Selemp_conuts = df["selfemp"].value_counts()
# plt.pie(Selemp_conuts, autopct='%1.f%%', colors=['lightcoral', 'yellow'], startangle=115)
# plt.title("創業人士及雇員的申辦者比例 - 圓餅圖")
# plt.axis('equal')
# plt.legend(loc="upper right", labels=["雇員", "創業人士"])
# plt.savefig("創業人士及雇員的申辦者比例 - 圓餅圖")
#-----------------------------------------------------------------------------#
'''創業人士成功核卡率 - 長條圖'''
# Selfemp_total = df["selfemp"].value_counts()
# Selfemp_yes = df[df["card"]== "yes"]['selfemp'].value_counts()
# yes_rates = (Selfemp_yes / Selfemp_total) * 100

# plt.bar(["雇員", "創業人士"], yes_rates.values, width=0.5, color='red')
# plt.xlabel("職務身份")
# plt.ylabel("核卡成功率")
# plt.grid(axis='y')
# plt.savefig("職務身份與核卡成功率關係 - 長條圖")
#-----------------------------------------------------------------------------#
'''扶養人數&信用卡核准率 - 箱型圖及分布圖''' 
# Dependents_median = df["dependents"].median()       #中位數:1
# Dependents_mode = df["dependents"].mode().values[0] #眾數:0
# Dependents_mean = df["dependents"].mean()           #平均數:0.99
# Dependents_Q1 = df["dependents"].quantile(0.25)    #Q1: 0
# Dependents_Q2 = df["dependents"].quantile(0.5)     #Q2: 1
# Dependents_Q3 = df["dependents"].quantile(0.75)    #Q3: 2

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(x=df["dependents"], color="skyblue", orient="h")
# plt.xlabel("扶養人數")
# plt.title("扶養人數 - 箱型圖")

# plt.subplot(1, 2, 2)
# plt.hist(df["dependents"][df["card"]=='yes'], bins=15, color="skyblue", width=1.2)
# plt.xlabel("扶養人數")
# plt.ylabel("申辦者人數")
# plt.title("扶養人數&信用卡核准率-直方圖")
# plt.grid(axis='y')
# plt.tight_layout()
# plt.savefig("扶養人數&信用卡核准率 - 箱型圖及直方圖")
#-----------------------------------------------------------------------------#
'''扶養人數核卡率長條圖'''
# Dependents_total = df["dependents"].value_counts()
# Dependents_yes = df[df["card"]== "yes"]["dependents"].value_counts()
# yes_rates = (Dependents_yes / Dependents_total) * 100

# plt.bar(Dependents_total.index, yes_rates.values, width=0.5, color='red')
# plt.xlabel("扶養人數")
# plt.ylabel("核卡成功率")
# plt.grid(axis='y')
# plt.savefig("扶養人數核卡率 - 長條圖")
#-----------------------------------------------------------------------------#
'''居住時間(Year)統計圖'''
# Months_median = df["months"].median()       #中位數:2
# Months_mode = df["months"].mode().values[0] #眾數:1
# Months_mean = df["months"].mean()           #平均數:4.3
# Months_Q1 = df["months"].quantile(0.25)    #Q1: 1
# Months_Q2 = df["months"].quantile(0.5)     #Q2: 2
# Months_Q3 = df["months"].quantile(0.75)    #Q3: 6

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(x=df["months"], color="skyblue", orient="h")
# plt.xlabel("居住時間(單位:年)")
# plt.grid(axis='x')
# plt.title("居住時間 - 箱型圖",)

# plt.subplot(1, 2, 2)
# plt.hist(df["months"][df["card"]=='yes'], bins=15, color="lightcoral", width=3)
# plt.xlabel("居住時間(單位:年)")
# plt.ylabel("申辦人數")
# plt.title("居住時間 -直方圖")
# plt.grid(axis='y')
# plt.savefig("居住時間 - 箱型圖及直方圖")
#-----------------------------------------------------------------------------#
'''居住時間與核卡率關係 - 長條圖'''
# df['居住時間分組'] = pd.cut(df['months'], bins=[-np.inf,5,10,15,20,25,30,35,40,np.inf],
#                       labels=["0~5","5~10","10~15","15~20","20~25","25~30","30~35","35~40","40up"])
# Month_total = df['居住時間分組'].value_counts()
# Month_yes = df['居住時間分組'][df["card"]=='yes'].value_counts()
# yes_rates = df['居住時間分組'][df["card"]=='yes'].value_counts()/df['居住時間分組'].value_counts() *100

# plt.bar(yes_rates.index, yes_rates.values, width=0.5, color='lightcoral')
# plt.xlabel('居住時間（單位:年）')
# plt.ylabel('核卡率 (%)')
# plt.title('居住時間與核卡率關係')
# plt.grid(axis='y')
# plt.savefig("居住時間與核卡率關係-長條圖")
#-----------------------------------------------------------------------------#
'''是否擁有主卡 - 圓餅圖'''
# Majorcards_conuts = df["majorcards"].value_counts()
# plt.pie(Majorcards_conuts, autopct='%1.f%%', colors=['lightcoral', 'yellow'], startangle=156)
# plt.title("是否擁有主卡 - 圓餅圖")
# plt.axis('equal')
# plt.legend(loc="upper right", labels=["有主卡", "有主卡"])
# plt.savefig("是否擁有主卡 - 圓餅圖")
#-----------------------------------------------------------------------------#
'''是否擁有主卡與核卡率關係 - 長條圖'''
# Majorcards_total = df["majorcards"].value_counts()
# Majorcards_yes = df[df["card"]== "yes"]['majorcards'].value_counts()
# yes_rates = (Majorcards_yes / Majorcards_total) * 100  # 有主卡 79.5%  沒主卡68.5%

# plt.bar(["有主卡", "沒主卡"], yes_rates.values, width=0.5, color='red')
# plt.xlabel("持有主卡")
# plt.ylabel("信用卡核准率")
# plt.grid(axis='y')
# plt.savefig("主卡數&信用卡核准率 - 長條圖")
#-----------------------------------------------------------------------------#
''' 活躍帳戶數量 - 箱型圖及分布圖''' # 以上的圖片都以存檔
# Active_mean = df["active"].mean()       #6.9
# Active_mode = df["active"].mode()[0]    #0
# Active_median = df["active"].median()   #6
# Active_Q1 = (df['active']/1000).quantile(0.25)     #Q1: 0.002
# Active_Q2 = (df['active']/1000).quantile(0.5)      #Q2: 0.006
# Active_Q3 = (df['active']/1000).quantile(0.75)     #Q3: 0.011

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(df["active"], color="skyblue", orient="h")
# plt.xlabel("活躍帳戶數量")
# plt.xticks([0, 10, 20, 30, 40])
# plt.title("活躍帳戶數量-箱型圖及分布圖")
# plt.grid(axis='x')

# plt.subplot(1, 2, 2)
# plt.hist(df["active"][df["card"]=='yes'], bins=20, color="skyblue", width=3.5)
# plt.xlabel("活躍帳戶數量")
# plt.ylabel("人數")
# plt.xticks([0, 10, 20, 30, 40, 50])
# plt.yticks([0, 50, 100, 150, 200])
# plt.title("活躍帳戶數量 - 分布圖")
# plt.grid(axis='y')
# plt.savefig("活躍帳戶數量-箱型圖及分布圖")
#-----------------------------------------------------------------------------#
'''活躍帳戶數量核准率 - 長條圖'''
# df["活躍分組"] = pd.cut(df["active"], bins=[-np.inf, 3, 6, 9, 12, 15, 18, 21, 25, 46], 
#                     labels=["0~3","3~6","6~9","9~12","12~15","15~18","18~21","21~25","25~26"])

# Active_total = df["活躍分組"].value_counts()
# Active_yes = df[df["card"]== "yes"]["活躍分組"].value_counts()
# yes_rates = (Active_yes / Active_total) * 100

# plt.bar(yes_rates.index, yes_rates.values, color='lightcoral', width=0.8, label='1')
# plt.grid(axis="y")
# plt.xlabel("活躍帳戶數量")
# plt.ylabel('信用卡核准率')
# plt.title("活躍帳戶數量&信用卡核准率 - 長條圖")
# plt.savefig("活躍帳戶數量&信用卡核准率 - 長條圖")
#-----------------------------------------------------------------------------#
'''核准通過&信用紀錄不良紀錄 - 長條圖'''
df["信用分組"] = pd.cut(df["reports"], bins=[-np.inf, 0, np.inf], labels=["有", "沒有"])
# Reports_yes = df[df["card"]=="yes"]["信用分組"]
# counts = Reports_yes.value_counts() # 通過者有910位 有信用不良紀錄

# plt.bar(counts.index, counts.values, color='lightcoral')
# plt.xlabel("信用不良紀錄")
# plt.ylabel("人數")
# plt.grid(axis='y')
# plt.title("核准通過&信用紀錄不良紀錄 - 長條圖")
# plt.savefig("核准通過&信用紀錄不良紀錄 - 長條圖")
# #-----------------------------------------------------------------------------#
'''核卡未通過者與其信用紀錄 - 長條圖'''
# Reports_no = df[df["card"]=="no"]["信用分組"]
# counts = Reports_no.value_counts() # 未通過者有151位 沒有信用不良紀錄

# plt.bar(counts.index, counts.values, color='lightcoral')
# plt.xlabel("信用不良紀錄")
# plt.ylabel("人數")
# plt.grid(axis='y')
# plt.title("核卡未通過者與其信用紀錄 - 長條圖")
# plt.savefig("核卡未通過者與其信用紀錄 - 長條圖")
#-----------------------------------------------------------------------------#
'''信用不良紀錄核卡率 - 長條圖'''
# Reports_yes = df[df["card"]=="yes"]["reports"]
# Reports_counts = Reports_yes.value_counts()
# Reports_total = Reports_yes.shape[0]

# Reports_total = df["reports"].value_counts()
# Reports_yes = df[df["card"]== "yes"]["reports"].value_counts()
# yes_rates = Reports_yes / Reports_total *100

# plt.bar(yes_rates.index, yes_rates.values, color='lightcoral')
# plt.xlabel("信用不良紀錄數量")
# plt.ylabel("核准成功率")
# plt.grid(axis='y')
# plt.title("信用不良紀錄 - 長條圖")
# plt.savefig("信用不良紀錄核卡率 - 長條圖")
#-----------------------------------------------------------------------------#
'''消費支出 - 箱型圖及分布圖'''  # 11231321
# Expenditure_mean = df["expenditure"].mean()       #184.9元
# Expenditure_mode = df["expenditure"].mode()[0]    #0元
# Expenditure_median = df["expenditure"].median()   #101.2元
# Expenditure_Q1 = (df['expenditure']).quantile(0.25)     #Q1: 4.58
# Expenditure_Q2 = (df['expenditure']).quantile(0.5)      #Q2: 101.2
# Expenditure_Q3 = (df['expenditure']).quantile(0.75)     #Q3: 248.9

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(df['expenditure'][df["card"]=='yes'], color="skyblue", orient="h")
# plt.xlabel("消費支出")
# plt.xticks([0, 1000, 2000, 3000])
# plt.title("消費支出 - 箱型圖")
# plt.grid(axis='x')

# plt.subplot(1, 2, 2)
# plt.hist(df['expenditure'], bins=20, color="skyblue")
# plt.xlabel("消費支出")
# plt.ylabel("申辦人數")
# plt.xticks([0, 1000, 2000, 3000])
# plt.yticks([0, 200, 400, 600, 800])
# plt.title("消費支出 - 直方圖")
# plt.grid(axis='y')
# plt.savefig("消費支出 - 箱型圖及直方圖")
#-----------------------------------------------------------------------------#
''' 消費支出&信用卡核准率 - 長條圖 '''
# df["消費平均分組"] = pd.cut(df["expenditure"], bins=[-np.inf, 5, 20, 50, 100, 500, 1000, np.inf], 
#                       labels=["<5","5~20","20~50", "50~100", "100~500","500~1000","1000-"])
# Expenditure_total = df["消費平均分組"].value_counts()
# Expenditure_yes = df[df["card"]== "yes"]["消費平均分組"].value_counts()
# yes_rates = (Expenditure_yes / Expenditure_total) * 100 #只有<5 核准率為10.6%

# plt.bar(yes_rates.index, yes_rates.values, color='lightcoral', width=0.8, label='1')
# plt.grid(axis="y")
# plt.xlabel("消費支出")
# plt.ylabel('信用卡核准率')
# plt.title("消費支出&信用卡核准率 - 長條圖")
# plt.savefig("消費支出&信用卡核准率 - 長條圖")
#-----------------------------------------------------------------------------#
''' 信用卡支出占比 - 箱型圖及分布圖 '''
# Share_mean = df["share"].mean()       #6.86%
# Share_mode = df["share"].mode()[0]    #0%
# Share_median = df["share"].median()   #3.87%
# Share_Q1 = (df['share']).quantile(0.25)     #Q1: 22%
# Share_Q2 = (df['share']).quantile(0.5)      #Q2: 3.8%
# Share_Q3 = (df['share']).quantile(0.75)     #Q3: 9.3%

# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# sns.boxplot(df["share"], color="skyblue", orient="h")
# plt.xlabel("信用卡支出占比")
# plt.xticks([0, 20, 40, 60, 80])
# plt.title("信用卡支出占比 - 箱型圖")
# plt.grid(axis='x')

# plt.subplot(1, 2, 2)
# plt.hist(df['share'][df["card"]=='yes'], bins=20, color="skyblue", width=4.5)
# plt.xlabel("信用卡支出占比")
# plt.ylabel("人數")
# plt.xticks([0, 20, 30, 40, 50, 60])
# plt.yticks([0, 100, 200, 300, 400])
# plt.title("信用卡支出占比 - 直方圖")
# plt.grid(axis='y')
# plt.savefig("信用卡支出占比-箱型圖及直方圖")
#-----------------------------------------------------------------------------#
''' 消費支出區間占比影響核准率 - 長條圖'''
# df["信用卡支出占比分組"] = pd.cut(df["share"], bins=[-np.inf, 1, 5, 10, 20, 30, 40, 50,np.inf], 
#                         labels=["<1", "1~5", "5~10","10~20", "20~30", "30~40","40~50","50-"])

# Share_total = df["信用卡支出占比分組"].value_counts()
# Share_yes = df[df["card"]== "yes"]["信用卡支出占比分組"].value_counts()
# yes_rates = (Share_yes / Share_total) * 100 # "<1"的核卡率僅有28% 其餘區間皆有高達100%

# plt.bar(yes_rates.index, yes_rates.values, color='lightcoral', width=0.8, label='1')
# plt.grid(axis="y")
# plt.xlabel("消費支出區間占比")
# plt.ylabel('信用卡核准率')
# plt.title("各消費支出區間占比影響核准率 - 長條圖")
# plt.savefig("消費支出區間占比影響核准率 - 長條圖")
#-----------------------------------------------------------------------------#





















