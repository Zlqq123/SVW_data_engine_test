import re
import pandas as pd
import numpy as np
#from efficient_apriori import apriori
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

tb_order = pd.read_csv('订单表.csv',encoding="gbk")

df1 = pd.DataFrame(tb_order, columns=['产品名称', '客户ID'])

hot_encoded_df = df1.groupby(['客户ID','产品名称'])['产品名称'].count().unstack().reset_index().fillna(0).set_index('客户ID')
hot_encoded_df = hot_encoded_df.applymap(encode_units)
frequent_itemsets = apriori(hot_encoded_df, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

print("频繁项集：", frequent_itemsets)
print("关联规则：", rules[(rules['lift'] >= 1) & (rules['confidence'] >= 0.2)])

