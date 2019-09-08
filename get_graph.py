import numpy as np
import matplotlib.pyplot as plt, mpld3


def get_graph(ticker, startTime, endTime):
    financialReturnsScore_arr = []
    growthScore_arr = []
    multipleScore_arr = []
    integratedScore_arr = []
    data = np.load('ticker_data_file.npy')
    print(data)
    name = ""
    for row in data:
        if row[1] == ticker:
            name = row[0]
            financialReturnsScore_arr.append(row[5])
            growthScore_arr.append(row[6])
            multipleScore_arr.append(row[7])
            integratedScore_arr.append(row[8])
    
    plt.plot(financialReturnsScore_arr, label='Financial Return Score')
    plt.plot(growthScore_arr, label='Growth Score')
    plt.plot(multipleScore_arr, label='Multiple Score')
    plt.plot(integratedScore_arr, label='Integrated Score')
    plt.title("Metrics for " +  name + " from " + startTime + " to " + endTime)
    plt.legend()
    plt.ylabel("Score")
    plt.xlabel("Time")
    plt.savefig(ticker+"_data.png")


get_graph('LULU', '2012-01-01', '2018-01-01')
