from Plot_AQI import avg_data_2013, avg_data_2014, avg_data_2015, avg_data_2016
import requests
import sys
import pandas as pd
from bs4 import BeautifulSoup
import os
import csv

def met_data(month, year):
    file_path = 'AQI_Prediction/Data/Html_Data/{}/{}.html'.format(year, month)
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, 'rb') as file_html:
        plain_text = file_html.read()

    soup = BeautifulSoup(plain_text, "html.parser")

    tempD = []
    finalD = []

    for table in soup.findAll('table', {'class': 'medias mensuales numspan'}):
        for tbody in table:
            for tr in tbody:
                tempD.append(tr.get_text())

    rows = len(tempD) / 15

    for _ in range(round(rows)):
        newtempD = [tempD.pop(0) for _ in range(15)]
        finalD.append(newtempD)

    finalD = finalD[1:-1]

    for entry in finalD:
        entry.pop(6)
        entry.pop(13)
        entry.pop(12)
        entry.pop(11)
        entry.pop(10)
        entry.pop(9)
        entry.pop(0)

    return finalD

def data_combine(year, cs):
    combined_data = []
    for chunk in pd.read_csv(f'AQI_Prediction/Data/Real-Data/real_{year}.csv', chunksize=cs):
        combined_data.extend(chunk.values.tolist())
    return combined_data

if __name__ == "__main__":
    if not os.path.exists("AQI_Prediction/Data/Real-Data"):
        os.makedirs("AQI_Prediction/Data/Real-Data")
    
    for year in range(2013, 2017):
        final_data = []
        with open(f'AQI_Prediction/Data/Real-Data/real_{year}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])

        for month in range(1, 13):
            temp = met_data(month, year)
            final_data.extend(temp)
        
        pm_data = getattr(sys.modules[__name__], f'avg_data_{year}')()

        if len(pm_data) == 364:
            pm_data.insert(364, '-')

        for i in range(len(final_data) - 1):
            final_data[i].insert(8, pm_data[i])

        with open(f'AQI_Prediction/Data/Real-Data/real_{year}.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect='excel')
            for row in final_data:
                if all(elem != "" and elem != "-" for elem in row):
                    writer.writerow(row)
                    
    data_2013 = data_combine(2013, 600)
    data_2014 = data_combine(2014, 600)
    data_2015 = data_combine(2015, 600)
    data_2016 = data_combine(2016, 600)
     
    total = data_2013 + data_2014 + data_2015 + data_2016
    
    with open('AQI_Prediction/Data/Real-Data/Real_Combine.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, dialect='excel')
        writer.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])
        writer.writerows(total)
        
df = pd.read_csv('AQI_Prediction/Data/Real-Data/Real_Combine.csv')
print(df.head())
