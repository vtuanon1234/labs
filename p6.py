import pandas as pd

data = pd.read_csv('dataset.csv')

totalExamples=len(data) 
positiveExamples = len(data.loc[data[data.columns[-1]]=='Yes'])
negativeExamples = totalExamples-positiveExamples

training = data.sample(frac=0.75,replace=False)
test = pd.concat([data, training]).drop_duplicates(keep=False)
prob={}

for col in training.columns[:-1]:
    prob[col] = {}
    vals = set(data[col])
    for val in vals:
        temp=training.loc[training[col]==val]
        pe=len(temp.loc[temp[temp.columns[-1]]=='Yes'])
        ne=len(temp)-pe
        prob[col][val]=[pe/positiveExamples,ne/negativeExamples]
print(prob)

prediction=[]
right_prediction=0

for i in range(len(test)):
    row=test.iloc[i,:]
    fpp=positiveExamples/totalExamples
    fpn=negativeExamples/totalExamples
    for col in test.columns[:-1]:
        fpp*=prob[col][row[col]][0]
        fpn*=prob[col][row[col]][1]
    if fpp>fpn:
        prediction.append('Yes')
    else:
        prediction.append('No')
    if prediction[-1]==row[-1]:
        right_prediction+=1

print('\nActual Values : ',list(test[test.columns[-1]]))
print('Predicted : ',prediction)
print('Accuracy : ',right_prediction/len(test))