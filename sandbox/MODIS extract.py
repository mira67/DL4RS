# Author: Yawen Zhang
# Date: Oct.23, 2016
# Use: Extract values from MODIS 7 bands and QC, cloud mask (.tif file) to from .csv file

# add gdal module to Python 2.7
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")

import gdal
import glob
from gdalconst import *
import datetime
import numpy as np
import csv

# Form a row: Year, Month, Day, nrow, ncol, QC, cloud, b1, b2, b3, b4, b5, b6, b7
arr = np.empty((0, 14), float)
file_date = 0

# Read .tif file
for i in range(len(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif"))):
    filename = glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i]

    # if it is a QC file
    if filename[116:118] == 'QC':
        # get date from filename
        print i / 9
        file_date = filename[63:70]
        file_date = datetime.datetime.strptime(file_date,'%Y%j').strftime('%Y%m%d')
        file_year = int(file_date[0:4])
        file_month = int(file_date[4:6])
        file_day = int(file_date[6:9])

        # read dataset
        dataset_QC = gdal.Open(filename, GA_ReadOnly)
        dataset_cloud = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 1], GA_ReadOnly)
        dataset_b1 = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 2], GA_ReadOnly)
        dataset_b2 = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 3], GA_ReadOnly)
        dataset_b3 = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 4], GA_ReadOnly)
        dataset_b4 = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 5], GA_ReadOnly)
        dataset_b5 = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 6], GA_ReadOnly)
        dataset_b6 = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 7], GA_ReadOnly)
        dataset_b7 = gdal.Open(glob.glob("/Users/yawen/ladsweb.nascom.nasa.gov/orders/501083075/*.tif")[i + 8], GA_ReadOnly)

        cols = dataset_QC.RasterXSize
        rows = dataset_QC.RasterYSize

        # get QC and band data value
        band_QC = dataset_QC.GetRasterBand(1)
        data_QC = band_QC.ReadAsArray(0, 0, cols, rows)

        band_cloud = dataset_cloud.GetRasterBand(1)
        data_cloud = band_cloud.ReadAsArray(0, 0, cols, rows)
        
        band_b1 = dataset_b1.GetRasterBand(1)
        data_b1 = band_b1.ReadAsArray(0, 0, cols, rows)

        band_b2 = dataset_b2.GetRasterBand(1)
        data_b2 = band_b2.ReadAsArray(0, 0, cols, rows)

        band_b3 = dataset_b3.GetRasterBand(1)
        data_b3 = band_b3.ReadAsArray(0, 0, cols, rows)

        band_b4 = dataset_b4.GetRasterBand(1)
        data_b4 = band_b4.ReadAsArray(0, 0, cols, rows)

        band_b5 = dataset_b5.GetRasterBand(1)
        data_b5 = band_b5.ReadAsArray(0, 0, cols, rows)

        band_b6 = dataset_b6.GetRasterBand(1)
        data_b6 = band_b6.ReadAsArray(0, 0, cols, rows)

        band_b7 = dataset_b7.GetRasterBand(1)
        data_b7 = band_b7.ReadAsArray(0, 0, cols, rows)
               
        for r in range(rows):
            for c in range(cols):
                QC = data_QC[r][c] & 3
                cloud = data_cloud[r][c] & 3
                b1 = data_b1[r][c] * 0.0001
                b2 = data_b2[r][c] * 0.0001
                b3 = data_b3[r][c] * 0.0001
                b4 = data_b4[r][c] * 0.0001
                b5 = data_b5[r][c] * 0.0001
                b6 = data_b6[r][c] * 0.0001
                b7 = data_b7[r][c] * 0.0001
                arr_append = np.array([[file_year, file_month, file_day, r, c, QC, cloud, b1, b2, b3, b4, b5, b6, b7]])
                arr = np.append(arr, arr_append, axis = 0)

# write arr to csv file            
with open('/Users/yawen/ladsweb.nascom.nasa.gov/orders/arr.csv','wb') as f:
    # write header to csv file
    writer = csv.DictWriter(f, fieldnames = ["year", "month", "day", "nrow", "ncol", "QC", "cloud", "b1", "b2", "b3", "b4", "b5", "b6", "b7"], delimiter = ',')
    writer.writeheader()
    np.savetxt(f, arr, fmt = '%.5f', delimiter=',')



                

            
    
    

