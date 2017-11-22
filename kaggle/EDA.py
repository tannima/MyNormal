# coding:utf-8
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

"""
unit sales可能存在负数
unit sales为0时，该行不存在，需要填充
没有库存信息，无法得知unit sales为0处是否是因为缺货
onpromotion 只有“是”、“否”信息，没有促销力度信息；train中有16%为NaN
train中存在商品不在test中，test中也存在商品不在train中
stores中的type和cluster含义不明
items中family从值名字看代表种类：杂货、酒水、清洁用品、农产品。class含义不明，perishable表示是否易腐坏，权重为1.25
holiday中有三百多个，描述中说应注意transfer
其他信息：薪水两周Pay一次，每月15号和最后一天；
April 16, 2016发生地震，此后几周的第一需求商品销量受到影响
oil price：经济会很大程度受到油价影响
transactions：训练数据中的每日交易数
"""
path = '/home/ubuntu/tanxiao/kaggle/'
os.chdir(path+'data/')

stores = pd.read_csv('stores.csv')
stores.city.value_counts()  # Quito城市最多
stores.state.value_counts()  # Pichincha州最多
stores.type.value_counts()  # ABCDE的type都有
stores.cluster.value_counts()  # 没那么集中

items = pd.read_csv('items.csv')
item_cnt = items.shape[0]    # 4100个商品
items.family.value_counts()  # 杂货最多
items['class'].value_counts()  # 大量只出现一次的值，含义不明
items.perishable.sum()  # 有986个易腐坏的

holidays = pd.read_csv('holidays_events.csv')

oil = pd.read_csv('oil.csv')  # 油价，不同年度的变化还是很大的，要看一下总销量和油价关系

transactions = pd.read_csv('transactions.csv')

train = pd.read_csv('train.csv')
# train.shape  # 1亿多行
train.head()
train.unit_sales.mean()
train.unit_sales.median()
train.unit_sales.min()
train.unit_sales.max()
'''
In [74]: train.unit_sales.describe()
Out[74]:
count    1.254970e+08
mean     8.554865e+00
std      2.360515e+01
min     -1.537200e+04
25%      2.000000e+00
50%      4.000000e+00
75%      9.000000e+00
max      8.944000e+04
'''

# 店铺数量变化
nstores = train.groupby('date')['store_nbr'].nunique()
nstores.to_csv(path + 'EDA/nstores.csv')

local_path = 'D:\\kaggle\\EDA\\'
nstores = pd.read_csv(local_path + 'nstores.csv', header=None)
nstores = nstores.rename(columns={0: 'date', 1: 'cnt'})
nstores['date'] = pd.to_datetime(nstores.date)
plt.plot(nstores.date, nstores.cnt)
# 每年第一天数据貌似都有点异常，是这一天关门了吗？
nstores[nstores.date == '2014-01-01']

# 总销售趋势与油价关系

train_sales_by_date = train.groupby('date')['unit_sales'].sum()
train_sales_by_date.to_csv(path + 'EDA/train_sales_by_date.csv')

train_sales_by_date = pd.read_csv(local_path + 'train_sales_by_date.csv', header=None)
train_sales_by_date = train_sales_by_date.rename(columns={0: 'date', 1: 'total_sales'})
train_sales_by_date['date'] = pd.to_datetime(train_sales_by_date.date)

oil = pd.read_csv('D:\\kaggle\\data\\oil.csv')
oil['date'] = pd.to_datetime(oil.date)

# 查看总销售量与油价的关系，貌似并没与太大关系。再细研究一下油价大的shock的影响？
# 总销售量除了每天初外，总体上有一个年度增长趋势以及年内的周期性
# 总销售量的变化：
# 1）有店铺说增加的影响（15-16年店铺增加多，后来不怎么变了）
# 2）有店铺平均增加的影响（13-14年店铺数量不怎么变，总销售额大幅增加）
# 地震影响：April 16总销售量并没有太大变化，可能是销售结构发生了变化（第一需求）
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(train_sales_by_date.date, train_sales_by_date.total_sales, label='total_sales')

ax1.set_xlabel('date')
ax2 = ax1.twinx()
ax2.plot(oil.date, oil.dcoilwtico, label='oil_price', c='darkorange')
ax2.set_ylabel('price')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')


# test预测，如果预测时间长，可能需要几个trend的变量，代表品牌、品类的增长趋势？
# 通过不断调大、小粒度的trend，看提交结果的表现
# 看了一下预测长度是在8月16-31号
# 可能最因为考虑的是这期间的年内周期性，找一下当地最近发生的事情（未来信息）
test = pd.read_csv('test.csv')
test.date.nunique()
test.item_nbr.nunique()     # 3901个



# 后面做数据分析可能不需要将数据按照实际销量分组，而是按照log(y+1)分组？
