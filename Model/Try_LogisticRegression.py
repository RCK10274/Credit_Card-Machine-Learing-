import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB      
from sklearn.naive_bayes import MultinomialNB     
from sklearn.naive_bayes import BernoulliNB   
from matplotlib import pyplot as plt

def matrix(true, pre):#混淆矩陣Funtion
    f = metrics.f1_score(true, pre)
    pre_score = metrics.precision_score(true, pre)
    re_score = metrics.recall_score(true, pre)
    
    return f, pre_score, re_score
class GridSearch:#網格搜索Funtion
    def __init__(self, model, model_str, param_grid):
        self.model = model
        self.param_grid = param_grid
        self.best_params = {}
        self.best_score = -np.inf
        self.model_name = model_str
        
    def fit(self, X_train, y_train, X_val, y_val):
        param_combinations = list(product(*self.param_grid.values()))
 
        for combination in tqdm(param_combinations, desc=f"GridSearch-{self.model_name}"):
            params = dict(zip(self.param_grid.keys(), combination))

            clf = self.model(**params)
            clf.fit(X_train, y_train)
            score = clf.score(X_val, y_val)

            if score > self.best_score:
                self.best_score = score
                self.best_params = params
        print(f"Best Score: {self.best_score}")
        print(f"Best Parameters: {self.best_params}")

class LogisticR:#邏輯迴歸參照Python機器學習62頁，以Adaline該造成邏輯迴歸
    def __init__(self, learning_rate=0.01, iter=10000, s=None):
        """
        learning_rate:學習率
        iter:跌代次數
        s:指定random seed
        """
        self.learning_rate = learning_rate
        self.iter = iter
        self.s = s
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.s)
        self.weight = rng.randn(n_features)
        self.bias = 0

        for _ in range(self.iter):
            y_pred = self.get_predict(X)
            dw = (1 / n_samples) * np.dot(X.T, y_pred-y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        #return self.weight, self.bias
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_predict(self, X):
        z = np.dot(X, self.weight) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        probabilities = self.get_predict(X)
        return np.array([1 if i > 0.5 else 0 for i in probabilities])

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

def gridtest(X_train,y_train, X_test, y_test):#網格搜索暴力查找最適合的參數
    parameter={"learning_rate":[i for i in np.arange(0.001,0.1,0.01)], "iter":[i for i in np.arange(100,1000,10)]}
    gs = GridSearch(LogisticR, "LogisticR", parameter)
    return gs.fit(X_train,y_train, X_test, y_test)

def main_getdata():#取得特徵資料與目標資料
    data_numeric = pd.read_csv("Data/Data2.csv")
    data_numeric = data_numeric.astype(float)
    features = data_numeric.drop(['card','expenditure', "Unnamed: 0"], axis=1)
    y_true = data_numeric['card']
    return features, y_true

def main():
    features, y_true = main_getdata()
    #===============================
    score = []
    fscore_db = []
    pre_db = []
    re_db = []
    #==================================================================================================
    #跌代test_iters次觀儲存每次跌代的混淆矩陣分數並儲存成CSV
    test_iters = 100
    for i in tqdm(range(test_iters)):
        X_train, X_test, y_train, y_test = train_test_split(features, y_true, test_size=0.2)
        GR = LogisticR(learning_rate=0.030999999999999996, iter=720)
        GR.fit(X_train, y_train)
        pre = GR.predict(X_test)
        sc = GR.score(X_test,y_test)
        score.append(sc)
        f, pre_score, re_score = matrix(y_test, pre)
        fscore_db.append(f)
        pre_db.append(pre_score)
        re_db.append(re_score)
    all_score = pd.DataFrame({"score":score, "Precision":pre_db, "Recall":re_db, "f_measure":fscore_db})
    all_score.describe().to_csv("Data/all_score_describe7.csv")
    #==================================================================================================
    #圖表繪製混淆矩陣分數總跌代次數的折線圖並儲存
    plt.plot(np.arange(1,test_iters+1), all_score.iloc[:,0], label="Score", color="#00ffc8")
    plt.axhline(y=all_score.iloc[:,0].mean(), linestyle="--", color="#00ffc8")
    plt.plot(np.arange(1,test_iters+1), all_score.iloc[:,1], label="Precision", color="#a6cfff")
    plt.axhline(y=all_score.iloc[:,1].mean(), linestyle="--", color="#a6cfff")
    plt.plot(np.arange(1,test_iters+1), all_score.iloc[:,2], label="Recall", color="#ffc9f5")
    plt.axhline(y=all_score.iloc[:,2].mean(), linestyle="--", color="#ffc9f5")
    plt.plot(np.arange(1,test_iters+1), all_score.iloc[:,3], label="F_measure", color="#FF3333")
    plt.axhline(y=all_score.iloc[:,3].mean(), linestyle="--", color="#FFD7AF")
    plt.xlabel("Number of iterations")
    plt.ylabel("Score")
    plt.xlim([0,test_iters+1])
    plt.ylim([0.0,1.00])
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.legend()
    plt.savefig("Data/score7.png")
    

if __name__=="__main__":
    main()