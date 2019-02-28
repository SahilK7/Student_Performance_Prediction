#Students Performance
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#math_dataset = pd.read_csv('student-mat.csv', sep=';')
por_dataset = pd.read_csv('student-por.csv', sep=';')

por_dataset = pd.get_dummies(por_dataset,drop_first=True)

X = por_dataset.iloc[:, :].values
y = por_dataset.iloc[:, -27].values

X = pd.DataFrame(X)


#df = pd.concat([math_dataset, por_dataset])
por_dataset.info()


# No null value
math_dataset.info()
df.info()

#df.isnull()
por_dataset.isnull().any()

summary =por_dataset.describe()

df.columns

'''df.columns = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime',
       'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc',
       'Walc', 'health', 'absences', 'G1', 'G2', 'G3']'''


df.columns = ['school','sex','age','address','family_size','parents_status','mother_education',
              'father_education','mother_job','father_job','reason','guardian','commute_time',
              'study_time','failures','school_support','family_support','paid_classes','activities',
              'nursery','desire_higher_edu','internet','romantic','family_quality','free_time',
              'go_out','weekday_alcohol_usage','weekend_alcohol_usage','health','absences',
              'period1_score','period2_score','period3_score']

# New column 
df['final_score_grade'] = 'na'


df.loc[(df.period3_score >=0) & (df.period3_score <=9), 'final_score_grade'] = 'poor'
df.loc[(df.period3_score >=10) & (df.period3_score <=14), 'final_score_grade'] = 'fair'
df.loc[(df.period3_score >=15) & (df.period3_score <=20), 'final_score_grade'] = 'good'


#Checking the distribution of target class
plt.figure(figsize=(10,8))
sns.countplot("G3", data=math_dataset)
plt.title('Final Score (Target Class) Distribution')
plt.xlabel('Number of Students', fontsize=16)


plt.figure(figsize=(8,6))
sns.countplot(df.final_score_grade, order=["poor","fair","good"], palette='Set1')
plt.title('Final Score Grade - Number of Students',fontsize=20)
plt.xlabel('Final Grade', fontsize=16)
plt.ylabel('Number of Student', fontsize=16)


corr = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, fmt='.1g')
plt.title('Correlation Heatmap', fontsize=20)
plt.savefig('heatmap.jpg')



romance_good_bad = pd.crosstab(index=df.final_score_grade, columns=df.romantic)
alcohol_wkdays_good_bad = pd.crosstab(index=df.final_score_grade, columns=df.weekday_alcohol_usage)
alcohol_wkends_good_bad = pd.crosstab(index=df.final_score_grade, columns=df.weekend_alcohol_usage)



good = df.loc[df.final_score_grade == 'good']
good['good_alcohol_usage']=good.weekend_alcohol_usage


poor = df.loc[df.final_score_grade == 'poor']
poor['poor_alcohol_usage']=poor.weekend_alcohol_usage

good['good_student_father_education'] = good.father_education


plt.figure(figsize=(8,6))
sns.countplot(df.address)
plt.title('Urban Vs Rural  Student count')
plt.xlabel('Living Area', fontsize=12)
plt.ylabel('Number of Students', fontsize=12)

X = df.iloc[:,:].values

#-------------------------------------------------------------------------------------
# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder = LabelEncoder()
#df['address'] = labelencoder.fit_transform(df['address'])
#df['sex'] = labelencoder.fit_transform(df['sex'])
#onehotencoder = OneHotEncoder(categorical_features =df['address'] )
#df['address'] = onehotencoder.fit_transform(df['address']).toarray()
#X[3] = X[3].astype('int64')
#X=pd.DataFrame(X)
#-------------------------------------------------------------------------------------
#sns.barplot(x='school',y='commute_time',data=df)
#plt.savefig('math_por_school.jpg')

#-------------------------------------------------------------------------------------
# Count girls and boys 
math_dataset['sex'].value_counts()
por_dataset['sex'].value_counts()
df['sex'].value_counts()

df['age'].value_counts()



def result(score):
    new=[]
    for i in score:
        if (i<8):
            i=0
        else:
            i=1
        new.append(i)
    return new
por_dataset['G3']=result(por_dataset['G3'])

por_dataset = pd.get_dummies(por_dataset,drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(por_dataset.drop(['G3'],axis=1), por_dataset['G3'], test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression() 
regressor.fit(X_train,y_train) 
regressor.score(X_test,y_test)

y_pred = regressor.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#-------------------------------------------------------------------------------------

X = pd.get_dummies(X,drop_first=True)
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((649,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,
              26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41]]
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()

