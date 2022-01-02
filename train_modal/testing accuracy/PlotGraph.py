### Import ###
import pandas as pd
import os
import matplotlib.pyplot as plt

### Variables ###
y_axis = []
X_axis = []

### Functions ###

## Get File List
def getFile():
    fileList = []
    for file in os.listdir("."):
        if file.endswith(".log"):
            fileList.append(file)
    fileList.sort()       
    print(fileList)
    return fileList

## Get Accuracy Columns
def readFileAndGetAccuracy(fileName):
    df = pd.read_csv(fileName)
    columnData = df['accuracy'].tolist()
    average = sum(columnData)/len(columnData)
    return average

## Get X-axis & Y-axis values
def getXaxisAndYaxisValues(fileList):
    data = {}
    for file in fileList:
        epochesName = int((file.split("_"))[1])
        #X_axis.append(epochesName[1])
        accuracy = readFileAndGetAccuracy(file)
        data[epochesName] = accuracy
        #y_axis.append(accuracy)
    return dict(sorted(data.items()))

## Draw Graph
def plotgraph(graphData):
    #for i in graphData:
        #print(i,graphData[i])
        #plt.plot(i,graphData[i],marker='o', markersize=7,label = str(i))
    print(graphData.keys())
    print(graphData.values())
    plt.plot(graphData.keys(),graphData.values(),marker='o', markersize=7)
    plt.ylabel('Accuracy')  
    plt.xlabel('Epoch')
    plt.title('Accuracy Graph (Training)')
    # show a legend on the plot
    #plt.legend()
    # function to show the plot
    plt.show()




### Main ###

# Get File List
fileList = getFile()
# Get Y&X axis values
graphData = getXaxisAndYaxisValues(fileList)
# Plot Graph
plotgraph(graphData)


    



