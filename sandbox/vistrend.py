"""
Load predicted mpf/if/wf from Database
Plot mean trend over 2000-2011
Smooth trend using savitzky_golay filter

Author: Qi Liu, 11/15/2016
"""
import os
import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dsphelper import savitzky_golay

plt.style.use(u'seaborn-paper')

#each year a plot, and calcualte a yearly mean to compare with nature
print("visualize results")
con = MySQLdb.connect(host="localhost",port=3306,user="mira67",passwd="1234",db="nsidcgt")
sql = """SELECT year,month,day,nrow,ncol,qc,
cloud,mpf,icef,wf
FROM canadian1115_dm
WHERE canadian1115_dm.year < 2012 AND canadian1115_dm.cloud = 0
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
import matplotlib.dates as mdates
import datetime
from pylab import *

plt.rcParams['figure.figsize'] = (5,5) # Make the figures a bit bigger
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['font.family'] = 'Times New Roman'
fs = 11

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
yearsFmt = mdates.DateFormatter('%m-%d')#format without year to elimate 1900 in ticks
day = []
mpf = []
mpf_all = []
day_all = []
gn = 0
fig, ax = plt.subplots()
for key, grp in mngroup:#fix last key
    gn = gn + 1#count number of groups
    if (key[0] != oldkey) or (gn == len(mngroup)):#if year not change or last group
        #ax.plot(day, mpf, label=int(oldkey))
        mpf_all.extend(mpf)#concatenate mpf from each month to a list
        day_all.extend(day)
        day = []
        mpf = []
        oldkey = key[0]
    cur_day = datetime.date(1900,int(grp['month'].as_matrix()[0]),int(grp['day'].as_matrix()[0]))
    day.append(cur_day)
    mpf.append(grp['mpf'].as_matrix()[0])

mpf_all_f = savitzky_golay(mpf_all, 21, 3)#smooth with sg filter
ax.plot(day_all, mpf_all_f, label='ST-DNN Model',color='#ff6600')
ax.plot(df1['Julian Day'],df1['MPF']*100, label='A.R Model',color='#666633')
ax.plot(df2['Julian Day'],df2['MPF'], label = 'CICE Model',color='#0099cc')
ax.xaxis.set_major_formatter(yearsFmt)
# format the ticks
#ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)

plt.ylabel('Melt Pond Fraction (%)',fontsize=fs, color='black')
plt.xlabel('Date (mm-dd)',fontsize=fs, color='black')
plt.tick_params(axis='both', which='major', labelsize=fs)
plt.title('Canadian Sea MPF Trend',fontsize=fs)
fig.autofmt_xdate()
ax.grid(True)
ax.legend(loc='upper right',prop={'size':fs})
plt.show()
#done
os.system('espeak "done"')
