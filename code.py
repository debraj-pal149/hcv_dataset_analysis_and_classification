import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
import seaborn as sns

pd.set_option('display.max_rows',200)
pd.set_option('display.max_columns',200)
#loading dataset
df=pd.read_csv('hcvdat0.csv').dropna()
ab=df.loc[:,['ALB', 'ALP','ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA',
             'GGT', 'PROT']].apply(lambda x: (x-np.mean(x))/np.std(x))#normalizing the variables 
df[['ALB', 'ALP','ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]=ab
df.head()

plt.hist(df.loc[:,'Category_num'])


#chi square test to check if there is dependence between categorical vars: Category of patient and Sex
cont_table=pd.crosstab(df.Category,df.Sex)
stat,p,dof,expected=chi2_contingency(cont_table)
print('The expected frequency table:',expected)
print('The p-value for chi2 test:',p)
if p>0.05:
    print('''The p value is more than 0.05 therefore we accept the null hypothesis''')
    print('No relation betweeen variables')
elif p<=0.05:
    print('''The p value is less than 0.05 therefore we reject the null hypothesis''')
    print('Variables are related')
    
  
#for analysis: prediction of Category (Y) using the other variables (covariates X)
#we assign variables and split them into training and testing sets
Y=df.Category_num
X=df.loc[:,['ALB', 'ALP','ALT', 'AST','BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
Y.unique()



#we shall be using the Logistic Regression model now for the classification task at hand
#creating the multinomial logistic regression model 
logreg=LogisticRegression(multi_class='multinomial',random_state=0)
   
  
#doing feature selection using recursive feature elimantion RFE 
rfe=RFE(logreg, n_features_to_select=6)
rfe.fit(x_train,y_train)
bo=rfe.support_ #this outputs an boolean array of which variables should be included in the model and which should not
x_train2=x_train.loc[:,bo]
x_test2=x_test.loc[:,bo]



#fitting the logistic regrssion model
logregf=logreg.fit(x_train2,y_train)
#fitting the predicted y values based on testing set of x
y_pred=logregf.predict(x_test2)


#calculating accuracy of fitted model based on difference
#between actual y values and predicted y values from our model
print('The accuracy of our fitted model is:',logregf.score(x_test2,y_test))
# The accuracy of our fitted model is: 0.940677966101695

#printing confusion matrix to visually see accuracy of model
cm=metrics.confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True,cmap='Blues_r')
plt.ylabel('Actual label');
plt.xlabel('Predicted label')
