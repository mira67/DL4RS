"""
File Transformation
Author: Qi Liu
Date: Oct.27.2016
"""

#import NSIDC groundtruth data
def nsidcgt(datapath):
    print "Write NSIDC GT"
    for tf in flist:
        df = pd.read_csv(tf, sep=",", skiprows=1, names = ["FID","Id","b1","b2","b3","b4","b5","b6","b7","b8","Rowid","MP_F","I_F","W_F",])
        nsplit = os.path.basename(tf).split('_')
        df_len = len(df['FID'])
        site_name = np.chararray((df_len), itemsize=6)
        site_name[:] = nsplit[0]
        year = np.zeros(df_len)
        year[:] = int(nsplit[1][1:5])
        day = np.zeros(df_len)
        day[:] = int(nsplit[1][5:8])
        day_label = np.chararray(df_len, itemsize=1)
        if len(nsplit[1]) == 9:
            day_label[:] = nsplit[1][8]
        else:
            day_label[:] = 'N'
            df['site'] = pd.Series(site_name, index=df.index)
            df['year'] = pd.Series(year, index=df.index)
            df['day'] = pd.Series(day, index=df.index)
            df['day_label'] = pd.Series(day_label, index=df.index)
            #record data to database
            df.to_sql(con=con, name=site_name[0], if_exists='append', flavor='mysql')
            con.close()
