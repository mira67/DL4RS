"""
MySQL Database Access Module
Author: Qi Liu
Date: Oct.27.2016
"""

import pandas as pd
import os
from pandas.io import sql
import MySQLdb
import logging

#general read with a sql to pandas format
def pdread(sql):
    con = MySQLdb.connect(host="localhost",port=3306,user="mira67",passwd="1234",db="nsidcgt")
    try:
        data = pd.read_sql(sql, con)
        con.close()
        return data
    except Exception,e:
        print str(e)
        # Rollback in case there is any error
        con.rollback()

#general write with a sql from pandas format
def pdwrite(sql,table_name,data):
    con = MySQLdb.connect(host="localhost",port=3306,user="mira67",passwd="1234",db="nsidcgt")
    try:
        data.to_sql(con=con, name=table_name, if_exists='append', flavor='mysql')
        con.close()
    except Exception,e:
        print str(e)
        # Rollback in case there is any error
        con.rollback()

#write csv to database without using pandas
def sqlwrite(path, filename, sql):
    con = MySQLdb.connect(host="localhost",port=3306,user="mira67",passwd="1234",db="nsidcgt")
    try:
         cursor = con.cursor()
         #create table first
         sqltable = """
            CREATE TABLE {}
            ( pid int(11),
              mpf double,
              icef double,
              wf double
            );
         """
         print sqltable.format(filename[:-4])
         cursor.execute(sqltable.format(filename[:-4]))
         #record results to file
         sqldata = sql.format(path+filename,filename[:-4])
         print sqldata
         cursor.execute(sqldata)
         con.commit()
    except Exception,e:
         print str(e)
         # Rollback in case there is any error
         con.rollback()

    con.close()
