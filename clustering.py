import requests
import json
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt, mpld3
import datetime as dt
import matplotlib.dates as mdates
from iexfinance.stocks import get_historical_data
from mpl_toolkits.mplot3d import Axes3D
import itertools
import pylab
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

def get_gs_data():
    clusters = 4
    start = "2012-01-01"
    end = "2018-01-01"

    auth_data = {
        "grant_type"    : "client_credentials",
        "client_id"     : "0cf5a96f6ad745b585c6fe637cdbb7f0",
        "client_secret" : "8d094e5c81329b12c562684ec2a301338cf47cf21815e2d53b0e457d79220b59",
        "scope"         : "read_product_data"
    }

    # create session instance
    session = requests.Session()

    auth_request = session.post("https://idfs.gs.com/as/token.oauth2", data = auth_data)
    access_token_dict = json.loads(auth_request.text)
    access_token = access_token_dict["access_token"]

    # update session headers with access token
    session.headers.update({"Authorization":"Bearer "+ access_token})

    request_url = "https://api.marquee.gs.com/v1/data/USCANFPP_MINI/query"

    request_query = {
                        "startDate": start,
                        "endDate": end
                }

    request = session.post(url=request_url, json=request_query)
    results = json.loads(request.text)

    gsids = set()
    for d in results['data']:
        gsids.add(d['gsid'])

    request_url = "https://api.marquee.gs.com/v1/assets/data/query"

    request_query = {
        "where": {
            "gsid": list(gsids)
        },
        "limit": 300,
        "fields": ["gsid", "ticker", "name"]
    }

    data = []

    mapping = {}
    request = session.post(url=request_url, json=request_query)
    response = json.loads(request.text)

    for dictionary in response['results']:
        name = dictionary['name']
        ticker = dictionary['ticker']
        gsid = dictionary['gsid']
        mapping[gsid] = [name, ticker]

    tickers = []

    ticker_dict = {}

    for d in results['data']:
        name = mapping[d['gsid']][0]
        ticker = mapping[d['gsid']][1]
        tickers.append(ticker)
        if 'multipleScore' not in d:
            multipleScore = None
        else:
            multipleScore = float(d['multipleScore'])
        
        if 'integratedScore' not in d:
            integratedScore = None
        else:
            integratedScore = float(d['integratedScore'])

        if 'financialReturnsScore' not in d:
            financialReturnsScore = None
        else:
            financialReturnsScore = float(d['financialReturnsScore'])

        if 'growthScore' not in d:
            growthScore = None
        else:
            growthScore = float(d['growthScore'])

        temp = [name, ticker, d['gsid'], d['date'], d['updateTime'], financialReturnsScore, growthScore, multipleScore, integratedScore]

        if ticker not in ticker_dict:
            ticker_dict[ticker] = []

        ticker_dict[ticker].append(temp)

        data.append(temp)

    data = np.array(data)

    np.save('ticker_data_file.npy', data)

    dataset = []
    for key in ticker_dict:
        last_five = ticker_dict[key][-5:]
        avgFinancialReturnsScore = 0
        avgGrowthScore = 0
        avgMultipleScore = 0
        avgIntegratedScore = 0
        
        for row in last_five:
            avgFinancialReturnsScore += row[5]/5
            avgGrowthScore += row[6]/5
            avgMultipleScore += row[7]/5
            avgIntegratedScore += row[8]/5
        dataset.append([avgFinancialReturnsScore, avgGrowthScore, avgMultipleScore, avgIntegratedScore])
    

    dataset = np.array(dataset)
    S = (1/4)*np.matmul(dataset.T, dataset)
    eigVals, eigVec = np.linalg.eig(S)
    sorted_index = eigVals.argsort()[::-1] 
    eigVals = eigVals[sorted_index]
    eigVec = eigVec[:,sorted_index]
    eigVec = eigVec[:,:3]
    transformed = np.matmul(dataset, eigVec)


    # Performing k-means
    iterations = 1000
    centers_ind = np.random.choice(range(len(transformed)), clusters)
    centers = []
    for i in centers_ind:
        centers.append(transformed[i])
    centers = np.array(centers)

    for iteration in range(iterations):
        # Update responsibilities
        r = np.zeros((len(transformed), clusters))
        for i in range(len(transformed)):
            p = transformed[i]
            closest = float('inf')
            arg_min = 0
            for j in range(len(centers)):
                c = centers[j]
                if np.linalg.norm(np.array(p)-np.array(c)) < closest:
                    closest = np.linalg.norm(np.array(p)-np.array(c))
                    arg_min = j
            r[i][arg_min] = 1

        # Update means
        for j in range(len(centers)):
            count = 0
            responsible_sum = np.array([0,0,0])
            for i in range(len(transformed)):
                if r[i][j] == 1:
                    responsible_sum = responsible_sum + transformed[i]
                    count += 1        
            centers[j] = np.divide(responsible_sum, count)
    
    grouped_data = []
    for j in range(clusters):
        xs = []
        ys = []
        zs = []
        for i in range(len(transformed)):
            if r[i][j] == 1:
                xs.append(transformed[i][0])
                ys.append(transformed[i][1])
                zs.append(transformed[i][2])
        grouped_data.append((xs, ys, zs))
    grouped_data = tuple(grouped_data)
    colors = ('r', 'b', 'g', 'y')
    groups = tuple(["Cluster " + str(i+1) for i in range(clusters)])


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor="1.0")
    ax = fig.gca(projection='3d')

    for data, color, group in zip(grouped_data, colors, groups):
        x, y, z = data
        ax.scatter(x, y, z, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('k-Means Clustering of PCA Dimensionality Reduction')
    plt.savefig("k_means_pca.png")

    cluster_outputs = [[] for i in range(clusters)]
    for j in range(clusters):
        for i in range(len(transformed)):
            if r[i][j] == 1:
                cluster_outputs[j].append(list(ticker_dict.keys())[i])
    
    colors = ('Red', 'Blue', 'Green', 'Yellow')
    i = 0
    for l in cluster_outputs:
        print()
        print(colors[i] + ": " + str(l))
        i += 1

    
get_gs_data()



