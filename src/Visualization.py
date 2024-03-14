from config import *
import matplotlib.pyplot as plt


def visualization(export_Daily_Report):
    Visual_Dict={'Material':[],
                 'WIP':[],
                 'Product':[],
                 'Keys':{'Material':[],
                         'WIP':[],
                         'Product':[]}}
    Key=['Material','WIP','Product']
    
    for id in I.keys():
        temp=[]
        for x in range(SIM_TIME):
            temp.append(export_Daily_Report[id*SIM_TIME+x][6])
        Visual_Dict[export_Daily_Report[id*SIM_TIME+x][2]].append(temp)
        Visual_Dict['Keys'][export_Daily_Report[id*SIM_TIME+x][2]].append(export_Daily_Report[id*SIM_TIME+x][1])
        
    if VISUALIAZTION=='Material':
        cont=0
        for lst in Visual_Dict[VISUALIAZTION]:
            plt.plot(lst,label=Visual_Dict['Keys'][VISUALIAZTION][cont])
            plt.legend()
            cont+=1
    elif VISUALIAZTION=='WIP':
        cont=0 
        for lst in Visual_Dict[VISUALIAZTION]:
            plt.plot(lst,label=Visual_Dict['Keys'][VISUALIAZTION][cont])
            plt.legend()
            cont+=1
    elif VISUALIAZTION=='Product':
        cont=0 
        for lst in Visual_Dict[VISUALIAZTION]:
            plt.plot(lst,label=Visual_Dict['Keys'][VISUALIAZTION][cont])
            plt.legend()
            cont+=1
    if VISUALIAZTION=='ALL':
        for x in range(3):
            plt.subplot(int(f"31{x+1}"))
            cont=0
            for lst in Visual_Dict[Key[x]]:
                plt.plot(lst,label=Visual_Dict['Keys'][Key[x]][cont])
                plt.legend()
                cont+=1
    
    
    plt.show()    
