#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:20:34 2017

@author: jasper
"""
import re
import pandas as pd
from mlxtend.frequent_patterns import apriori#, association_rules
from mlxtend.preprocessing import OnehotTransactions
import json
import glob

a = []

for file in glob.glob("data/*.json"):
    data = json.load(open(file))
    try:
        hashtags = re.findall(r"#(\w+)", data["caption"])
        hashtags = [h.lower() for h in hashtags]
        a.append(hashtags)
    except KeyError:
        break

oht = OnehotTransactions()
oht_ary = oht.fit(a).transform(a)
df = pd.DataFrame(oht_ary, columns=oht.columns_)
print("im here")

apr = apriori(df,min_support=0.06,use_colnames=True)

