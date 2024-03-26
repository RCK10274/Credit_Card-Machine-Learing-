import pandas as pd
import numpy as np

def DataProcess(path):

    data = pd.read_csv(path)
    check_null = np.sum(data.isnull(), axis=0)#確認沒有缺失值
    #print(check_null)
    data = data.drop(data[(data["age"]>=0) & (data["age"]<=15)].index, axis=0)
    
    data["reports"] = data.loc[:,"reports"].apply(lambda x: "less than 4" if x < 4 else "equal and greater then 4")
    one_hot_rep = pd.get_dummies(data["reports"], prefix="reports")#---------------------
    
    data['age'] = data['age'].astype(float)
    def age(row):
        if row>=18 and row<30:
            return "18~30"
        elif row>=30 and row<50:
            return "30~50"
        elif row>=50:
            return "50~"
    data['age']=data['age'].apply(age)
    one_hot_age = pd.get_dummies(data["age"], prefix="age")#------------------------
    #print(np.sort(data["age"].unique()))
    #----------------------------------------------------------------------------------
    def ceiling_floor(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        max_within_limits = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit), col].max()
        min_within_limits = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit), col].min()

        df.loc[df[col] < lower_limit, col] = min_within_limits
        df.loc[df[col] > upper_limit, col] = max_within_limits

        return df
    data = ceiling_floor(data, 'income')
    data = ceiling_floor(data, 'expenditure')
    data = ceiling_floor(data, 'share')
    """
    def income(row):
        if row>=4:
            return ">=4"
        else:
            return "<4"
    data["income_category"] = data["income"].apply(income)
    one_hot_income = pd.get_dummies(data["income_category"], prefix="income")
    """

    def z(row):#極值正規化
        return (row-row.mean())/row.std()
    data["income"] = z(data["income"])
    data["expenditure"] = z(data["expenditure"])
    #----------------------------------------------------------------------------------
    one_hot_owner = pd.get_dummies(data["owner"], prefix="owner")#------------------------
    #----------------------------------------------------------------------------------
    one_hot_selfemp = pd.get_dummies(data["selfemp"], prefix="selfemp")#------------------------
    #----------------------------------------------------------------------------------
    def dep(row):
        if row==0 or row==1:
            return "0~1"
        else:
            return "1~"
    data["dependents"] = data["dependents"].apply(dep)
    one_hot_dep = pd.get_dummies(data["dependents"], prefix="dependents")#------------------------
    #----------------------------------------------------------------------------------
    def mon(row):
        if row<50:
            return "~50"
        elif row<50 and 60<row:
            return "50~60"
        else:
            return "60~"
    data["months"] = data["months"].apply(mon)
    one_hot_mon = pd.get_dummies(data["months"], prefix="months")#------------------------
    #----------------------------------------------------------------------------------
    def mc(row):
        if row>=1:
            return ">=1"
        else:
            return "0"
    data["majorcards"] = data["majorcards"].apply(mc)
    one_hot_mc = pd.get_dummies(data["majorcards"], prefix="majorcards")#------------------------
    #----------------------------------------------------------------------------------
    def active(row):
        if row>=7:
            return ">=7"
        else:
            return "<7"
    data["active"] = data["active"].apply(active)
    one_hot_active = pd.get_dummies(data["active"], prefix="active")#------------------------
    #----------------------------------------------------------------------------------

    
    #----------------------------------------------------------------------------------
    feature_df = data.copy()
    feature_df["card"] = feature_df["card"].map({"yes":True, "no":False})
    feature_df = feature_df.drop(["reports", "age", "owner", "selfemp", "dependents", "months", "majorcards", "active"], axis=1)
    feature_df = pd.concat([feature_df, one_hot_rep, one_hot_age, one_hot_owner, one_hot_selfemp, 
                    one_hot_dep, one_hot_mon, one_hot_mc, one_hot_active], axis=1).reset_index(drop=True)
    #-----------------------------------------------------------------------------------
    return feature_df


def process2(data):
    df = data[["Ind_ID", "Propert_Owner", "Annual_income", "Birthday_count", "Family_Members", "label"]]
    #check_null = np.sum(df.isnull(), axis=0)
    df = df.drop(df[df.isnull().any(axis=1)].index)
    #print(np.unique(df["Propert_Owner"]))
    df["Propert_Owner"] = df["Propert_Owner"].map({"Y":1, "N":0})

    def ceiling_floor(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        max_within_limits = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit), col].max()
        min_within_limits = df.loc[(df[col] <= upper_limit) & (df[col] >= lower_limit), col].min()

        df.loc[df[col] < lower_limit, col] = min_within_limits
        df.loc[df[col] > upper_limit, col] = max_within_limits

        return df
    df = ceiling_floor(df, "Annual_income")

    def z(row):#極值正規化
        return (row-row.mean())/row.std()
    df["Annual_income"]=z(df["Annual_income"])
    #print(df["Annual_income"][(df["Annual_income"]>1) | (df["Annual_income"]<-1)].max())
    df["Birthday_count"] = abs(df["Birthday_count"]/365.5)
    def age(row):
        if row>=18 and row<30:
            return "18~30"
        elif row>=30 and row<50:
            return "30~50"
        elif row>=50:
            return "50~"
        
    df["Birthday_count"]=df["Birthday_count"].apply(age)
    one_hot_B = pd.get_dummies(df["Birthday_count"], prefix="age")
    def dep(row):
        if row==0 or row==1:
            return "0~1"
        else:
            return "1~"
        
    df["Family_Members"] = df["Family_Members"].map(dep)

    one_hot_P = pd.get_dummies(df["Propert_Owner"], prefix="owner")
    one_hot_F = pd.get_dummies(df["Family_Members"], prefix="dependents")
    df = df.drop(["Birthday_count", "Family_Members", "Propert_Owner"], axis=1)
    feature_df2 = pd.concat([df, one_hot_B,one_hot_P,one_hot_F], axis=1).reset_index(drop=True)
    feature_df2.to_csv("Data/Data3.csv", index=False)






data = DataProcess("Data/AER_credit_card_data.csv")
data.to_csv("Data/Data2.csv")


'''
#data_name ="Data/AER_credit_card_data.csv"
#feature_df = DataProcess(data_name)
#feature_df.to_csv("Data/Data2.csv", index=False)

data2 = pd.read_csv("Data/Credit_card.csv")
res = pd.read_csv("Data/Credit_card_label.csv")
data2 = pd.concat([data2,res["label"]], axis=1)
print(data2)
process2(data2)
'''