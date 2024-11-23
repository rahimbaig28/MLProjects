from Plot_AQI import avg_data_2013, avg_data_2014, avg_data_2015, avg_data_2016
import requests
import sys
import pandas as pd
from bs4 import BeautifulSoup
import os
import csv

def met_data(month, year):
    file_html = open('air_quality_index/Data/Html_Data/{}/{}.html'.format(year, month), 'rb')
    plain_text = file_html.read()

    tempD = []
    finalD = []

    soup = BeautifulSoup(plain_text, "lxml")
    for table in soup.findAll('table', {'class': 'medias mensuales numspan'}):
        for tbody in table:
            for tr in tbody:
                a = tr.get_text()
                tempD.append(a)

    rows = len(tempD) / 15

    for times in range(round(rows)):
        newtempD = []
        for i in range(15):
            newtempD.append(tempD[0])
            tempD.pop(0)
        finalD.append(newtempD)

    length = len(finalD)

    finalD.pop(length - 1)
    finalD.pop(0)

    for a in range(len(finalD)):
        finalD[a].pop(6)
        finalD[a].pop(13)
        finalD[a].pop(12)
        finalD[a].pop(11)
        finalD[a].pop(10)
        finalD[a].pop(9)
        finalD[a].pop(0)

    return finalD

def data_combine(year, cs):
    combined_data = []
    for a in pd.read_csv(f'air_quality_index/Data/Real-Data/real_{year}.csv', chunksize=cs):
        combined_data.extend(a.values.tolist())
    return combined_data

if __name__ == "__main__":
    if not os.path.exists("air_quality_index/Data/Real-Data"):
        os.makedirs("air_quality_index/Data/Real-Data")
    
    for year in range(2013, 2017):
        final_data = []
        with open(f'air_quality_index/Data/Real-Data/real_{year}.csv', 'w', newline='') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            wr.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])
        
        for month in range(1, 13):
            temp = met_data(month, year)
            final_data.extend(temp)
        
        pm = getattr(sys.modules[__name__], f'avg_data_{year}')()

        if len(pm) == 364:
            pm.insert(364, '-')

        for i in range(len(final_data) - 1):
            final_data[i].insert(8, pm[i])

        with open(f'air_quality_index/Data/Real-Data/real_{year}.csv', 'a', newline='') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            for row in final_data:
                if all(elem != "" and elem != "-" for elem in row):
                    wr.writerow(row)
                    
    data_2013 = data_combine(2013, 600)
    data_2014 = data_combine(2014, 600)
    data_2015 = data_combine(2015, 600)
    data_2016 = data_combine(2016, 600)
     
    total = data_2013 + data_2014 + data_2015 + data_2016
    
    with open('air_quality_index/Data/Real-Data/Real_Combine.csv', 'w', newline='') as csvfile:
        wr = csv.writer(csvfile, dialect='excel')
        wr.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])
        wr.writerows(total)
        
df = pd.read_csv('air_quality_index/Data/Real-Data/Real_Combine.csv')
print(df.head())
