import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


data=pd.read_csv("C:\\Users\\buğra\\Desktop\\train.csv")
data.fillna(0,inplace=True )
print(data.columns)



le=LabelEncoder()
data['Age'].fillna(data['Age'].mean(), inplace=True)

data['Sex']=le.fit_transform(data['Sex'])

data['Ticket']=le.fit_transform(data['Ticket'])

x=data[['Parch','Fare','SibSp','Ticket','Sex']]
y=data['Survived']


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.3 ,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


accuracy=accuracy_score(y_test, y_pred)
print(f"acunariy: {accuracy}")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


uuı=f1_score(y_test,y_pred, average='micro')
print(f"f1_score: {uuı}")

"""  
plt.pie(data['Sex'].value_counts(), labels=data['Sex'].unique(), autopct="%1.1f%%", startangle=90)
plt.show()

sns.countplot(x='Embarked',data=data)
plt.show()

sns.lineplot(x='Survived',y='Pclass', hue='Sex',data=data)
plt.show()

sns.lineplot(x='Survived',y='Parch',data=data, hue='Sex')
plt.show()
"""