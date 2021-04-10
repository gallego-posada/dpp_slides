import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


CLR = {'dark': '#222523',
          'dgray': '#767B78',
          'gray': '#D9D9D9',
          'dblue': '#17375E',
          'blue': '#418deb',
          'purple': '#904895',
          'green': '#32735B',
          'gold': '#CDA63C',
          'lgold': '#EEB822',
          'red': '#BF0603',
         }

import seaborn as sns
# sns.set_context("talk", font_scale=1.)
sns.set(style='white', context='talk', font_scale=1.)

sns.set_palette(sns.color_palette(list(CLR.values())))

import utils