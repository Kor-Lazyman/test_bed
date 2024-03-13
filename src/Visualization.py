from config import *
import matplotlib.pyplot as plt

Plt=[]
Wip=[]
Mat=[]
Product=[]
def visualization(export_Daily_Report):
    for id in I.keys():
        if export_Daily_Report[id*SIM_TIME][2]=="Material":
            temp=[]
            for x in range(SIM_TIME):
                temp.append(export_Daily_Report[id*SIM_TIME+x][6])
            Mat.append(temp)

            
        elif export_Daily_Report[id*SIM_TIME][2]=="WIP":
            temp=[]
            for x in range(SIM_TIME):
                temp.append(export_Daily_Report[id*SIM_TIME+x][6])
            Wip.append(temp)

        elif export_Daily_Report[id*SIM_TIME][2]=="Product":
            temp=[]
            for x in range(SIM_TIME):
                temp.append(export_Daily_Report[id*SIM_TIME+x][6])
            Product.append(temp)
    
