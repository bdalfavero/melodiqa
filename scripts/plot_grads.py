#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("filename", help="Filename for data input.")
parser.add_argument("--output", help="Filename for image output.")
parser.add_argument("--show", action="store_true", help="Shows the plot in a window.")
args = parser.parse_args()

df = pd.read_csv(args.filename, index_col="i")

fig, ax = plt.subplots()
sns.lineplot(data=df, x="i", y="weight_grad", ax=ax)
sns.lineplot(data=df, x="i", y="visible_grad", ax=ax)
sns.lineplot(data=df, x="i", y="hidden_grad", ax=ax)
if args.output:
    plt.savefig(args.output)
if args.show:
    plt.show()
