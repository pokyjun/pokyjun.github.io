---
layout : single
title: "Shop Customer Data"
categories : "분석"
tag : [python, Pandas]
author_profile: false
sidebar:
  nav: "docs"
---


Kaggle의 Shop Customer Data입니다.
- CustomerID : 고객 아이디
- Gender : 성별
- Age : 나이
- Annual Income($) : 연간 소득
- Spending Score(1-100) : 고객의 소비 점수
- Profession : 직업
- Work Experience : 근무 경력
- Family Size : 가족 구성원 수

### 라이브러리


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글깨짐 방지
plt.rcParams['font.family'] = 'AppleGothic'

data = pd.read_csv('Customers.csv')
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Annual Income ($)</th>
      <th>Spending Score (1-100)</th>
      <th>Profession</th>
      <th>Work Experience</th>
      <th>Family Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15000</td>
      <td>39</td>
      <td>Healthcare</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>35000</td>
      <td>81</td>
      <td>Engineer</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>86000</td>
      <td>6</td>
      <td>Engineer</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>59000</td>
      <td>77</td>
      <td>Lawyer</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>38000</td>
      <td>40</td>
      <td>Entertainment</td>
      <td>2</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



### 전처리
분석만 진행할 예정이니 확인만 하고 넘어가겠습니다.


```python
# 필요없는 CustomerID는 제거하겠습니다.
data.drop('CustomerID', axis=1, inplace=True)
```


```python
# 결측치
data.isnull().sum()
```




    Gender                     0
    Age                        0
    Annual Income ($)          0
    Spending Score (1-100)     0
    Profession                35
    Work Experience            0
    Family Size                0
    dtype: int64



Profession에 결측치가 35개 있습니다


```python
# 중복값
data.duplicated().sum()
```




    0



중복값 없습니다.

이상치는 넘기겠습니다.


```python
# 성별 평균 값
data.groupby('Gender').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Annual Income ($)</th>
      <th>Spending Score (1-100)</th>
      <th>Work Experience</th>
      <th>Family Size</th>
    </tr>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Female</th>
      <td>48.822934</td>
      <td>110553.715008</td>
      <td>50.974705</td>
      <td>4.035413</td>
      <td>3.768128</td>
    </tr>
    <tr>
      <th>Male</th>
      <td>49.159705</td>
      <td>110991.323096</td>
      <td>50.944717</td>
      <td>4.200246</td>
      <td>3.769042</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 직업별 평균 값
data.groupby('Profession').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Annual Income ($)</th>
      <th>Spending Score (1-100)</th>
      <th>Work Experience</th>
      <th>Family Size</th>
    </tr>
    <tr>
      <th>Profession</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Artist</th>
      <td>49.004902</td>
      <td>108776.580065</td>
      <td>52.678105</td>
      <td>4.215686</td>
      <td>3.653595</td>
    </tr>
    <tr>
      <th>Doctor</th>
      <td>46.621118</td>
      <td>111573.217391</td>
      <td>51.900621</td>
      <td>4.304348</td>
      <td>3.670807</td>
    </tr>
    <tr>
      <th>Engineer</th>
      <td>55.094972</td>
      <td>111161.240223</td>
      <td>48.966480</td>
      <td>3.955307</td>
      <td>3.581006</td>
    </tr>
    <tr>
      <th>Entertainment</th>
      <td>51.162393</td>
      <td>110650.333333</td>
      <td>52.940171</td>
      <td>3.500000</td>
      <td>3.888889</td>
    </tr>
    <tr>
      <th>Executive</th>
      <td>46.601307</td>
      <td>113770.130719</td>
      <td>49.901961</td>
      <td>4.248366</td>
      <td>3.967320</td>
    </tr>
    <tr>
      <th>Healthcare</th>
      <td>47.843658</td>
      <td>112574.041298</td>
      <td>50.516224</td>
      <td>4.002950</td>
      <td>3.905605</td>
    </tr>
    <tr>
      <th>Homemaker</th>
      <td>45.366667</td>
      <td>108758.616667</td>
      <td>46.383333</td>
      <td>6.133333</td>
      <td>4.050000</td>
    </tr>
    <tr>
      <th>Lawyer</th>
      <td>47.753521</td>
      <td>110995.838028</td>
      <td>48.859155</td>
      <td>3.528169</td>
      <td>3.619718</td>
    </tr>
    <tr>
      <th>Marketing</th>
      <td>45.823529</td>
      <td>107994.211765</td>
      <td>48.717647</td>
      <td>4.305882</td>
      <td>3.729412</td>
    </tr>
  </tbody>
</table>
</div>



### 시각화


```python
sns.countplot(data, x=data['Gender'])
plt.title('회원 수')
```




    Text(0.5, 1.0, '회원 수')




    
![png](https://user-images.githubusercontent.com/103250467/233824986-592da65e-f643-416d-bf33-4ab5db637b4a.png)
    


아무래도 남자보다 여자의 비율이 더 높다.


```python
# 회원이 가장 많은 나이 TOP 10
age = data['Age'].value_counts()
age_index = age.head(10).index
age_values = age.head(10).values

sns.barplot(x=age_index, y=age_values, order=age_index) # order가 없으면 x축이 오름차순으로 그려짐
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('회원이 가장 많은 나이 TOP 10')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824987-42eb54bc-cc20-4c95-8442-831a2f048442.png)
    


전체 회원 중 가장 많은 나이의 4순위가 91세다.  
좀 이상하다..?


```python
# 회원이 가장 적은 나이 TOP 10
age = data['Age'].value_counts()
age_index = age.tail(10).index
age_values = age.tail(10).values

sns.barplot(x=age_index, y=age_values, order=age_index[::-1])
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('회원이 가장 적은 TOP 10')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824989-e34be182-9497-4922-bcde-fb1d9f1e168b.png)
    



```python
# 남자 소비점수 탑 100의 직업
man = data[data['Gender'] == 'Male']
man_top_100 = man.sort_values('Spending Score (1-100)', ascending=False).head(100)
plt.figure(figsize=(6,4))
ax = sns.countplot(man_top_100, x='Profession', order=man_top_100['Profession'].value_counts().index)
ax.bar_label(ax.containers[0], fmt='%.0f')
plt.xticks(rotation=45)
plt.title('남자 소비점수 탑 100의 직업')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824990-89eddad7-96e9-465d-836c-082d72e3c42e.png)
    



```python
# 여자 소비점수 탑 100의 직업
woman = data[data['Gender'] == 'Female']
woman_top_100 = woman.sort_values('Spending Score (1-100)', ascending=False).head(100)
plt.figure(figsize=(6,4))
ax = sns.countplot(woman_top_100, x='Profession', order=woman_top_100['Profession'].value_counts().index)
ax.bar_label(ax.containers[0], fmt='%.0f')
plt.xticks(rotation=45)
plt.title('여자 소비점수 탑 100의 직업')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824992-ebcadd03-7fde-4d3b-9f87-b4126e00c7f3.png)
    


2023-04-19   
1달러 : 1331.84


```python
# 남자 top 100명의 평균 연봉
average = man_top_100['Annual Income ($)'].sum() / 100
one_dollar = 1331.84
print(f"남자 top 100명의 평균 연봉 : {average * one_dollar:,.0f}원")
```

    남자 top 100명의 평균 연봉 : 145,327,318원



```python
# 여자 top 100명의 평균 연봉
average = woman_top_100['Annual Income ($)'].sum() / 100
one_dollar = 1331.84
print(f"여자 top 100명의 평균 연봉 : {average * one_dollar:,.0f}원")
```

    여자 top 100명의 평균 연봉 : 157,714,135원



```python
ax = sns.countplot(man_top_100, x=man_top_100['Family Size'])
ax.bar_label(ax.containers[0])
plt.title('남자 탑 100 가족 구성원 수')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824993-aaa1d424-ec15-4729-b23e-fa5359f6e6e8.png)
    



```python
# man_top_100을 제외한 나머지 데이터
man_714 = pd.DataFrame(man.loc[~man.index.isin(man_top_100.index)])

'''
- 'isin'은 인자로 받은 값이 해당 시리즈, 데이터프레임에 포함되어 있으면 True 포함되어 있지 않으면 False를 뱉어준다.
- '~' 은 부정을 의미한다.
->  isin으로 man_top_100이 true가 나왔지만 ~로 인해 False 바뀐다. 
    man의 전체 데이터에서 False로 바뀐 man_top_100을 제외한 나머지 값들이 man_714에 들어감
'''

ax = sns.countplot(man_714, x=man_714['Family Size'])
ax.bar_label(ax.containers[0])
plt.title('남자 탑 100을 제외한 데이터의 가족 구성원 수')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824994-320d8553-3b74-44eb-b1b7-9557c6f2e1d0.png)
    



```python
ax = sns.countplot(woman_top_100, x=woman_top_100['Family Size'])
ax.bar_label(ax.containers[0])
plt.title('여자 탑 100 가족 구성원 수')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824995-1d0d604b-2513-4744-b250-dbe224160c3b.png)
    



```python
woman_1086 = pd.DataFrame(woman.loc[~woman.index.isin(woman_top_100.index)])
ax = sns.countplot(woman_1086, x=woman_1086['Family Size'])
ax.bar_label(ax.containers[0])
plt.title('여자 탑 100을 제외한 데이터의 가족 구성원 수')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824996-109d8622-93db-4196-9e40-bd619f3d74f6.png)
    


man_top_100이랑 거의 7배 차이가 나서 ÷7을 했지만 가족 구성원 수는 비슷하다.  
woman_top_100이랑 10배 조금 넘게 차이 난다. ÷10을 했지만 그렇게 많이 차이가 나지 않는다.  
  
가족이 많아서 소비 점수가 높은 줄 알았는데 아니다.  

그러면 직업과 관련이 있을 것으로 생각이 든다.  


```python
ax = sns.countplot(man, y=man['Profession'], order=man['Profession'].value_counts().index)
ax.bar_label(ax.containers[0], fmt='%.0f')
plt.title('남자 전체 데이터 직업')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824997-18a7478d-3583-4857-a88b-2d60b9df2b5c.png)
    



```python
colors = ['#ed6868', '#ffc28c', '#fffd8c', '#deff8c', '#9fff8c', '#8cfff0', '#8cbcff', '#9b8cff', '#ff8cf7']

wedgeprops = {'linewidth': 1, 'edgecolor': 'black'}

plt.pie(man['Profession'].value_counts(), labels=man['Profession'].value_counts().index, autopct='%1.1f%%', colors=colors, wedgeprops=wedgeprops, explode=[0.05] * 9)
plt.title('Male Profession Percentage')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233824999-95d973d3-3216-4077-b9cf-b4222a19f6c1.png)
    



```python
ax = sns.countplot(woman, y=woman['Profession'], order=woman['Profession'].value_counts().index)
ax.bar_label(ax.containers[0], fmt='%.0f')
plt.title('여자 전체 데이터 직업')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233825000-2c5da65f-6a9f-43a7-a131-9a0d800671f6.png)
    



```python
colors = ['#ed6868', '#ffc28c', '#fffd8c', '#deff8c', '#9fff8c', '#8cfff0', '#8cbcff', '#9b8cff', '#ff8cf7']

wedgeprops = {'linewidth': 1, 'edgecolor': 'black'}

plt.pie(woman['Profession'].value_counts(), labels=woman['Profession'].value_counts().index, autopct='%1.1f%%', colors=colors, wedgeprops=wedgeprops, explode=[0.05] * 9)
plt.title('Female Profession Percentage')
plt.show()
```


    
![png](https://user-images.githubusercontent.com/103250467/233825001-050b5bd5-0fae-42f9-854b-61a2432c8f17.png)
    


위에서 확인해 보았던 top 100의 남, 여 둘 다 Artist가 압도적으로 많았다.   
혹시나 해서 전체 직업도 확인을 해보았더니 남, 여 둘 다 1/4 이상이 Artist 직업을 가지고 있다.   
데이터가 몇 년도 데이터인지는 모르겠지만 딱 봤을 때 Artist가 많은 걸 보니 Youtube, 인터넷 방송 등이 인기를 끌면서 많은 사람들이 Artist의 직업을 갖게 된 것으로 보인다.  
거기에 고객 소비 점수도 Artist가 압도적으로 많은 것을 볼 수 있다.   
아무래도 카메라, 조명, 스피커 등 장비들이 가격대가 있어서 소비 점수가 많이 올라간 것으로 보인다.
