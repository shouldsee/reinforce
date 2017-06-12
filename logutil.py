from matplotlib import pyplot as plt
import csv
import numpy as np

def readcsv(name, usedict = 0):
    with open(name, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        first = 1;
        for row in spamreader:
            if first:
                fieldind = [i for i,x in enumerate(row) if not x.replace('.','').replace('-','').isdigit()]
    #             print(fieldind);
                header = [row[i] for i in fieldind];
    #             data = {row[i]:[] for i in fieldind};
                print(header)
                data = [[] for i in fieldind];
                first = 0;
            for i,n in enumerate(fieldind):
                data[i].append(row[n+1]);
    if usedict:
        data = {header[i]:data[i] for i,fi in enumerate(fieldind)};
    return(data)



def quickax(siz = [8,8] ):
    fig = plt.figure(figsize=siz);
    ax1 = fig.add_subplot(221)
    return ax1
LogName = 'Models/ac1.log';

winwid = 10;
movmean = lambda x,winwid:np.convolve(x,np.ones((winwid,))/winwid,'valid' );

def visualise(ax, AgentName, winwid = 10):
    LogFile = 'Models/'+AgentName+'.log';
    x = np.array(readcsv(LogFile,True)['Score']).astype('float');
    movm_x = movmean(x,winwid);
    ax.plot(movm_x, label = AgentName);