def describeData(dataSetToDescribe):
    # Get an overview of the various statistics of the dataset
    print(dataSetToDescribe.describe())

    # Identify the missing entries for the various indices in order to better assess what data needs to be cleansed
    #print(dataSetToDescribe.info())
