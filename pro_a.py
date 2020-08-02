import re
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
headers={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
urlp="http://car.bitauto.com/xuanchegongju/?mid=8&page="
v_name,tpdz,jgwf=[],[],[]
for i in range(1,4):
    url=urlp+str(i)
    html=requests.get(url,headers=headers,timeout=10)
    content = html.text
    soup = BeautifulSoup(content, 'html.parser', from_encoding='utf-8')
    b=soup.body.contents
    b1=b[23].contents
    b2=b1[7].contents
    b3=b2[5].contents#"search-result-list-item"搜索结果内容
    for j in range(len(b3)):
        if b3[j]!="\n":
            v1=b3[j].a
            v2=v1.contents
            v_name.append(v2[3].contents[0])
            tpdz.append(v2[1].attrs['src'])
            jgwf.append(v2[5].contents[0])
max_price,min_price,unit=[],[],[]
for a in jgwf:
    if a=="暂无":
        max_price.append(None)
        min_price.append(None)
        unit.append(None)
    else:
        u=re.sub("[0-9\.\-]", "", a)
        unit.append(u)
        if '-' in a:
            p_list=a.split('-',1)
            min_price.append(float(p_list[0]))
            max_price.append(float(re.sub(u,'',p_list[1])))
        else:
            max_price.append(float(re.sub(u,'',a)))
            min_price.append(float(re.sub(u,'',a)))
df=pd.DataFrame({'name':v_name, 'min price':min_price, 'max price': max_price,'price unit': unit, 'photo links':tpdz})
df.to_csv("vehicle_price_from_bitauto.csv", encoding='utf_8_sig', index=False, sep=',')