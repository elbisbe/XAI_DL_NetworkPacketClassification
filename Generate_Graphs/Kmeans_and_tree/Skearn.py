# %%
# Imports

import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
# %%
df_allPackets_train = pd.read_csv("train_data.csv",names=["PacketType", "PacketLength"])
df_allPackets_test = pd.read_csv("test_data.csv",names=["PacketType", "PacketLength"])

df_allPackets_train = df_allPackets_train.replace({'Chat': 0, 'Audio': 1, 'Email': 2,'FTPS': 3,'Video': 4})
df_allPackets_test = df_allPackets_test.replace({'Chat': 0, 'Audio': 1, 'Email': 2,'FTPS': 3,'Video': 4})

df_allPackets = pd.concat([df_allPackets_train,df_allPackets_test], axis=0, sort=False)
#df_allPackets = df_allPackets.replace({'Chat': 0, 'Audio': 1, 'Email': 2,'FTPS': 3,'Video': 4})
# %%


trainLengthData = df_allPackets_train['PacketLength'].to_numpy().reshape(-1, 1).astype('int')
trainType = df_allPackets_train['PacketType'].to_numpy().astype('int')

testLengthData = df_allPackets['PacketLength'].to_numpy().reshape(-1, 1).astype('int')
testType = df_allPackets['PacketType'].to_numpy().astype('int')

finalLengthData = df_allPackets['PacketLength'].to_numpy().reshape(-1, 1).astype('int')
finalType = df_allPackets['PacketType'].to_numpy().astype('int')

print(finalLengthData.dtype)

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(finalLengthData, finalType)


# %%

clf.predict_proba([[150]])
# %%
tree.plot_tree(clf,filled=True)
plt.savefig('tree1.pdf',format='pdf',bbox_inches = "tight")
# %%

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(testLengthData, testType)

predictedTestType = clf.predict(testLengthData)
acc = accuracy_score(testType, predictedTestType)

print(acc)
# %%