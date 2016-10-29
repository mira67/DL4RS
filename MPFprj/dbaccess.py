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

con = MySQLdb.connect(host="localhost",port=3306,user="mira67",passwd="1234",db="nsidcgt")

#general read with a sql to pandas format
def pdread(sql):
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
    try:
        data.to_sql(con=con, name=table_name, if_exists='append', flavor='mysql')
    except Exception,e:
        print str(e)
        # Rollback in case there is any error
        con.rollback()
    con.close()

#write csv to database without using pandas
def sqlwrite(filename, sql):
    try:
         cursor = con.cursor()
         cursor.execute(sql.format(filename,filename))
         con.commit()
    except Exception,e:
         print str(e)
         # Rollback in case there is any error
         con.rollback()
