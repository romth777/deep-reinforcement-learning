# -*- coding: utf-8 -*-
from pathlib import Path

log_path = Path("./console_research.log")

with log_path.open("r") as f:
    data = f.readlines()

tag = ["normal", "cauchy", "exponential", "gamma", "t", "uniform", "binomial", "beta", "chisquare", "else"]

rets = {}
ret = []

for item in data[17:]:
    item = item.strip()
    if item in tag:
        print(item)
    else:
        if item != "":
            item = item.split()
            print(item[6])
