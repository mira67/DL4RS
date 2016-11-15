import os
import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

#each year a plot, and calcualte a yearly mean to compare with nature
print("visualize results")
con = MySQLdb.connect(host="localhost",port=3306,user="mira67",passwd="1234",db="nsidcgt")
sql = """SELECT year,month,day,nrow,ncol,qc,
cloud,mpf,icef,wf
FROM esiber1115_dm
WHERE esiber1115_dm.year < 2012 AND esiber1115_dm.cloud = 0
"""

df = pd.read_sql(sql, con)
print df.shape
#group by year/month to compute MPF/IF/WF mean

grouped = df.groupby(['month','day','nrow','ncol'],as_index=False)
print("Number of Groups: ", len(grouped))
dfmean = grouped.aggregate(np.mean)
#print dfmean.head(n=5)
ngroup = dfmean.groupby(['month','day','nrow','ncol'],as_index=False)

con.close()
#os.system('espeak "done"')

#daily average plot across years, and compare with figure data in literatures
plt.rcParams['figure.figsize'] = (10,10) # Make the figures a bit bigger
plt.rcParams['axes.labelsize'] = 30

import matplotlib.dates as mdates
import datetime
from pylab import *

xl = pd.ExcelFile("figure_data.xlsx")
xl.sheet_names
df1 = xl.parse("MODIS MPF 2000-2011 mean")
df2 = xl.parse("CICE 2001-2013 mean MPF")

mgrouped = df.groupby(['month','day'],as_index=False)
print("Number of Groups: ", len(mgrouped))
mdfmean = mgrouped.aggregate(np.mean)
#print dfmean.head(n=5)
mngroup = mdfmean.groupby(['month','day'],as_index=False)
#visualize trend by month for many years
oldkey = 5

years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
yearsFmt = mdates.DateFormatter('%m-%d')
day = []
mpf = []
gn = 0
fig, ax = plt.subplots()
for key, grp in mngroup:#fix last key
    gn = gn + 1#count number of groups
    if (key[0] != oldkey) or (gn == len(mngroup)):#if year not change or last group
        #mpf = smooth(np.asarray(mpf),3,'hanning')
        ax.plot(day, mpf, label=int(oldkey))
        day = []
        mpf = []
        oldkey = key[0]
    #cur_day = datetime.date(int(grp['year'].as_matrix()[0]),int(grp['month'].as_matrix()[0]),int(grp['day'].as_matrix()[0]))
    cur_day = datetime.date(1900,int(grp['month'].as_matrix()[0]),int(grp['day'].as_matrix()[0]))
    day.append(cur_day)
    mpf.append(grp['mpf'].as_matrix()[0])

ax.plot(df1['Julian Day'],df1['MPF']*100, label='ANN MODIS')
ax.plot(df2['Julian Day'],df2['MPF'], label = 'Nature CICE')

ax.xaxis.set_major_formatter(yearsFmt)
# format the ticks
#ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
fig.autofmt_xdate()
ax.grid(True)
ax.legend(loc='upper right')
plt.show()
os.system('espeak "done"')
