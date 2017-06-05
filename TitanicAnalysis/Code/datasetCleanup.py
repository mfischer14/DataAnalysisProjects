import numpy
# Remove passengers that didn't embark (i.e. resulting in survivor rate higher than expected)
def clean_unembarkedPassengers(df):
    numPassengers = len(df)
    df = df.drop(df.Embarked.notnull())
    print('Removed {} passengers that have not embarked'.format(numPassengers - len(df)))
    return df

def performCleanup(df):
    #df = clean_unembarkedPassengers(df)
    del df['Name']
    del df['Ticket']
    del df['Cabin']
    #df['Age'] = df['Age'].fillna(df['Age'].mean)
    #df['Fare'] = df['Fare'].fillna(df['Fare'].mean)
    del df['Age']
    del df['Fare']
    df['Sex'] = (df['Sex'] == "male") * 1
    df['Embarked'] = df['Embarked'].fillna(0)
    df['Embarked'] = (df['Embarked'] != 0) * 1
    return df

def splitFeaturesFromLabels(df):
    labels = df['Survived']
    del df['Survived']
    return df, labels

def generateTestSubSample(features_train, labels_train):
    return (features_train.loc[:round(len(features_train)*0.9,0)],
            labels_train.loc[:round(len(labels_train)*0.9)],
            features_train.loc[round(len(features_train)*0.9,0)+1:len(features_train)],
            labels_train.loc[round(len(labels_train)*0.9,0)+1:len(labels_train)])
