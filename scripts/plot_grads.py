#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("filename")
args = parser.parse_args()

df = pd.read_csv(args.filename, index_col="i")

fig, ax = plt.subplots()
sns.lineplot(data=df, x="i", y="weight_grad", ax=ax)
sns.lineplot(data=df, x="i", y="visible_grad", ax=ax)
sns.lineplot(data=df, x="i", y="hidden_grad", ax=ax)
plt.show()
