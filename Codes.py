

import pickle
import pandas as pd
df_atp = pd.read_csv('data/Data.csv')
df_atp.head()
#Different features used in our prediction
'''
Date: date of the match
Series: name of ATP tennis series (we kept the four main current categories namely Grand Slams, Masters Surface: type of surface (clay, hard or grass)
Round: round of match (from first round to the final)
Best of: maximum number of sets playable in match (Best of 3 or Best of 5)
WRank: ATP Entry ranking of the match winner as of the start of the tournament
LRank: ATP Entry ranking of the match loser as of the start of the tournament
'''
# The output variable is binary. The better player has higher rank by definition.
# win=1, if higher ranked player wins, win=0, if higher ranked player loses
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
sns.set_style("darkgrid")
import pylab as pl
#After loading the dataset we proceed as following:
#Keep only completed matches i.e. eliminate matches with injury withdrawals and walkovers.
#For convenience we rename Best of to Best_of
#Choose the features listed above
#Drop NaN entries
#Consider the two final years only (to avoid comparing different categories of tournaments which existed #We note that this choice is somewhat arbitrary and can be changed if needed.
#Choose only higher ranked players for better accuracy
df_atp['Date'] = pd.to_datetime(df_atp['Date'])
# Restricing dates
df_atp = df_atp.loc[(df_atp['Date'] > '2014-11-09') & (df_atp['Date'] <= '2016-11-09')]
# Keeping only completed matches
df_atp = df_atp[df_atp['Comment'] == 'Completed'].drop("Comment",axis = 1)
# Rename Best of to Best_of
df_atp.rename(columns = {'Best of':'Best_of'},inplace=True)
# Choosing features
cols_to_keep = ['Date','Series','Surface', 'Round','Best_of', 'WRank','LRank']
# Dropping NaN
df_atp = df_atp[cols_to_keep].dropna()
# Dropping errors in the dataset and unimportant entries (e.g. there are very few entries for Masters df_atp = df_atp[(df_atp['LRank'] != 'NR') & (df_atp['WRank'] != 'NR') & (df_atp['Series'] != 'Masters df_atp.head()
# Transform strings into numerical values
1
df_atp[['Best_of','WRank','LRank']] = df_atp[['Best_of','WRank','LRank']].astype(int)
df_atp.head()
#Creating an extra columns for the variable win described above using an auxiliary function win(x)
def win(x):
if x > 0:
return 0
elif x <= 0:
return 1
df_atp['win'] = (df_atp['WRank'] - df_atp['LRank']).apply(win)
df_atp.head()
# A previous analysis by some researchers as Corral and Prieto-Rodriguez (2010), we are restricting # to top players as including everyone decreases the predictive power
df_new = df_atp.copy()
df2_new = df_new[(df_new['WRank'] <= 150) & (df_new['LRank'] <= 150)]
df2_new.head()
# Restricting our analysis to matches of Best_of = 5. Since only Grand Slams have 5
# sets we can drop the new Series column. The case of Best_of = 3 will be considered later.
df3 = df2_new.copy()
df3 = df3[df3['Best_of'] == 5]
# Drop Best_of and Series columns
df3.drop("Series",axis = 1,inplace=True)
df3.drop("Best_of",axis = 1,inplace=True)
df3.head()
print(df3['win'].sum(axis=0))
print(len(df3.index))
df3['win'].sum(axis=0)/float(len(df3.index))
# Looking at the wins before stratification surface wise
win_by_Surface_before_strat = pd.crosstab(df3.win, df3.Surface).apply( lambda x: x/x.sum(), axis =
win_by_Surface_before_strat = pd.DataFrame( win_by_Surface_before_strat.unstack() ).reset_index()
win_by_Surface_before_strat.columns = ["Surface", "win", "total" ]
fig2 = sns.barplot(win_by_Surface_before_strat.Surface, win_by_Surface_before_strat.total, hue = win_fig2.figure.set_size_inches(8,5)
# Looking at the wins before stratification rounds wise
win_by_round_before_strat = pd.crosstab(df3.win, df3.Round).apply( lambda x: x/x.sum(), axis = 0 )
win_by_round_before_strat = pd.DataFrame(win_by_round_before_strat.unstack() ).reset_index()
win_by_round_before_strat.columns = ["Round", "win", "total" ]
fig2 = sns.barplot(win_by_round_before_strat.Round, win_by_round_before_strat.total, hue = win_by_round_fig2.figure.set_size_inches(8,5)
#dataset is uneven in terms of frequency of wins, to balance this using stratified sampling procedure
y_0 = df3[df3.win == 0]
y_1 = df3[df3.win == 1]
n = min([len(y_0), len(y_1)])
y_0 = y_0.sample(n = n, random_state = 0)
y_1 = y_1.sample(n = n, random_state = 0)
df_strat = pd.concat([y_0, y_1])
X_strat = df_strat[['Date', 'Surface', 'Round','WRank', 'LRank']]
y_strat = df_strat.win
X_strat.head()
y_strat.head()
2
X_strat_1=X_strat.copy()
X_strat_1['win']=y_strat
X_strat_1.head()
# Rank1 > Rank2
df = X_strat_1.copy()
df["P1"] = df[["WRank", "LRank"]].max(axis=1)
df["P2"] = df[["WRank", "LRank"]].min(axis=1)
df.head()
#------------------------------------------------------------------------------------------------------------
# Looking at wins surface wise
win_by_Surface = pd.crosstab(df.win, df.Surface).apply( lambda x: x/x.sum(), axis = 0 )
win_by_Surface
win_by_Surface = pd.DataFrame( win_by_Surface.unstack() ).reset_index()
win_by_Surface.columns = ["Surface", "win", "total" ]
fig2 = sns.barplot(win_by_Surface.Surface, win_by_Surface.total, hue = win_by_Surface.win )
fig2.figure.set_size_inches(8,5)
# results: Clay court upsets are more often than the other two surfaces
# Looking at wins rounds wise
win_by_round = pd.crosstab(df.win, df.Round).apply( lambda x: x/x.sum(), axis = 0 )
win_by_round
win_by_round = pd.DataFrame(win_by_round.unstack() ).reset_index()
win_by_round.columns = ["Round", "win", "total" ]
fig2 = sns.barplot(win_by_round.Round, win_by_round.total, hue = win_by_round.win )
fig2.figure.set_size_inches(8,5)
# A slight trend of higher ranked player winning emerges from semifinals and finals
#To keep the dataframe cleaner we transform the Round entries into numbers. We then transform rounds df1 = df.copy()
def round_number(x):
if x == '1st Round':
return 1
elif x == '2nd Round':
return 2
elif x == '3rd Round':
return 3
elif x == '4th Round':
return 4
elif x == 'Quarterfinals':
return 5
elif x == 'Semifinals':
return 6
elif x == 'The Final':
return 7
df1['Round'] = df1['Round'].apply(round_number)
dummy_ranks = pd.get_dummies(df1['Round'], prefix='Round')
df1 = df1.join(dummy_ranks.loc[:, 'Round_2':])
df1[['Round_2', 'Round_3', 'Round_4', 'Round_5', 'Round_6', 'Round_7']] = df1[['Round_2', 'Round_3'
df1.head()
3
#Performing the same for surface
dummy_ranks = pd.get_dummies(df1['Surface'], prefix='Surface')
dummy_ranks.head()
df_2 = df1.join(dummy_ranks.loc[:, 'Surface_Grass':])
df_2.drop("Surface",axis = 1,inplace=True)
df_2[['Surface_Grass','Surface_Hard']] = df_2[['Surface_Grass','Surface_Hard']].astype('int_')
df_2.drop("Round",axis = 1,inplace=True)
df_2.head()
df4 = df_2.copy()
df4['P1'] = np.log2(df4['P1'].astype('float64'))
df4['P2'] = np.log2(df4['P2'].astype('float64'))
df4['D'] = df4['P1'] - df4['P2']
df4['D'] = np.absolute(df4['D'])
df4.head()
import numpy as np
import matplotlib.pyplot as plt
# Logit Function
def logit(x):
return np.exp(x) / (1 + np.exp(x))
x = np.linspace(-6,6,50, dtype=float)
y = logit(x)
plt.plot(x, y)
plt.ylabel("Probability")
plt.show()
df4.columns.tolist()
feature_cols = ['Round_2', 'Round_3', 'Round_4', 'Round_5', 'Round_6', 'Round_7', 'Surface_Grass',
dfnew = df4.copy()
dfnew[feature_cols].head()
with pd.option_context('mode.use_inf_as_na', True):
dfnew = dfnew.dropna(subset=['D'], how='all')
X = dfnew[feature_cols]
y = dfnew.win
# Test_train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#---------------------Data Procesing Ends------------------------------------------------------------------------
# 1. LOGISTIC REGRESSION
filename = 'lr.sav'
loaded_model = pickle.load(open(filename, 'rb'))
y_pred_class = loaded_model.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
print('True:', y_test.values[0:40])
print('Pred:', y_pred_class[0:40])
y_pred_prob = loaded_model.predict_proba(X_test)[:, 1]
auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
auc_score
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
fig = plt.plot(fpr, tpr,label='ROC curve (area = %0.2f)' % auc_score )
plt.plot([0, 1], [0, 1], 'k--')
4
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for win classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.grid(True)
import statsmodels.api as sm
X = dfnew[feature_cols]
X = sm.add_constant(X)
y = dfnew['win']
lm = sm.Logit(y, X)
result = lm.fit()
result.summary()
filename = 'lr_cv.sav'
loaded_model = pickle.load(open(filename, 'rb'))
pred_probs = loaded_model.predict_proba(dfnew[["D"]])
plt.scatter(dfnew["D"], pred_probs[:,1])
plt.title('Win Probability for 5 sets matches')
plt.xlabel('D')
plt.ylabel('Win Probability for 5 sets matches')
plt.legend(loc="lower right")
plt.grid(True)
#------------------------------------------------------------------------------
# 2. DECISION TREES
X = dfnew[feature_cols].dropna()
y = dfnew['win']
filename1 = 'dt.sav'
model = pickle.load(open(filename1, 'rb'))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print('AUC {}, Average AUC {}'.format(scores, scores.mean()))
#------------------------------------------------------------------------------
# 3. RANDOM FORESTS
from sklearn.model_selection import cross_val_score
X = dfnew[feature_cols].dropna()
y = dfnew['win']
filename2 = 'rf.sav'
model = pickle.load(open(filename2, 'rb'))
features = X.columns
feature_importances = model.feature_importances_
features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
features_df.sort_values('Importance Score', inplace=True, ascending=False)
features_df
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
#feature_importances.sort()
feature_importances.plot(kind="barh", figsize=(7,6))
#Performing cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='roc_auc', cv= 5)
5
print('AUC {}, Average AUC {}'.format(scores, scores.mean()))
6
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 4 02:40:27 2019
@author: karth
"""
# Help taken to understand and implement some functionalities
#general google search,
#https://github.com/cuffery/cs229fall2017
#https://www.kaggle.com/edouardthomas/beat-the-bookmakers-with-machine-learning-tennis/notebook#Table-#https://www.kaggle.com/kerneler/starter-a-large-tennis-dataset-for-atp-14f6b5ff-9
import pickle
#Results for the men's ATP tour
import pandas as pd
df_atp = pd.read_csv('data/Data.csv')
df_atp.head()
#Different features used in our prediction
'''
Date: date of the match
Series: name of ATP tennis series (we kept the four main current categories namely Grand Slams, Masters Surface: type of surface (clay, hard or grass)
Round: round of match (from first round to the final)
Best of: maximum number of sets playable in match (Best of 3 or Best of 5)
WRank: ATP Entry ranking of the match winner as of the start of the tournament
LRank: ATP Entry ranking of the match loser as of the start of the tournament
'''
# The output variable is binary. The better player has higher rank by definition.
# win=1, if higher ranked player wins, win=0, if higher ranked player loses
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
sns.set_style("darkgrid")
import pylab as pl
#After loading the dataset we proceed as following:
#Keep only completed matches i.e. eliminate matches with injury withdrawals and walkovers.
#For convenience we rename Best of to Best_of
#Choose the features listed above
#Drop NaN entries
#Consider the two final years only (to avoid comparing different categories of tournaments which existed #We note that this choice is somewhat arbitrary and can be changed if needed.
#Choose only higher ranked players for better accuracy
# tested with all the matches which gives a logistic regression accuracy of 0.66, DT accuracy of 0.58 # which is comparitively less than the accuracy achieved by restricting dates
df_atp['Date'] = pd.to_datetime(df_atp['Date'])
1
# Restricing dates
df_atp = df_atp.loc[(df_atp['Date'] > '2014-11-09') & (df_atp['Date'] <= '2016-11-09')]
# Keeping only completed matches
df_atp = df_atp[df_atp['Comment'] == 'Completed'].drop("Comment",axis = 1)
# Rename Best of to Best_of
df_atp.rename(columns = {'Best of':'Best_of'},inplace=True)
# Choosing features
# Selecting all the features further gives us an accuracy on logistic regression of 0.6259 which is cols_to_keep = ['Date','Series','Surface', 'Round','Best_of', 'WRank','LRank']
# Dropping NaN
df_atp = df_atp[cols_to_keep].dropna()
# Dropping errors in the dataset and unimportant entries (e.g. there are very few entries for Masters df_atp = df_atp[(df_atp['LRank'] != 'NR') & (df_atp['WRank'] != 'NR') & (df_atp['Series'] != 'Masters df_atp.head()
# Transform strings into numerical values
df_atp[['Best_of','WRank','LRank']] = df_atp[['Best_of','WRank','LRank']].astype(int)
df_atp.head()
#Creating an extra columns for the variable win described above using an auxiliary function win(x)
def win(x):
if x > 0:
return 0
elif x <= 0:
return 1
df_atp['win'] = (df_atp['WRank'] - df_atp['LRank']).apply(win)
df_atp.head()
# A previous analysis by some researchers as Corral and Prieto-Rodriguez (2010), we are restricting # to top players as including everyone decreases the predictive power
df_new = df_atp.copy()
df2_new = df_new[(df_new['WRank'] <= 150) & (df_new['LRank'] <= 150)]
df2_new.head()
# Restricting our analysis to matches of Best_of = 5. Since only Grand Slams have 5
# sets we can drop the new Series column. The case of Best_of = 3 will be considered later.
df3 = df2_new.copy()
df3 = df3[df3['Best_of'] == 5]
# Drop Best_of and Series columns
df3.drop("Series",axis = 1,inplace=True)
df3.drop("Best_of",axis = 1,inplace=True)
df3.head()
print(df3['win'].sum(axis=0))
print(len(df3.index))
df3['win'].sum(axis=0)/float(len(df3.index))
# Looking at the wins before stratification surface wise
win_by_Surface_before_strat = pd.crosstab(df3.win, df3.Surface).apply( lambda x: x/x.sum(), axis =
win_by_Surface_before_strat = pd.DataFrame( win_by_Surface_before_strat.unstack() ).reset_index()
win_by_Surface_before_strat.columns = ["Surface", "win", "total" ]
fig2 = sns.barplot(win_by_Surface_before_strat.Surface, win_by_Surface_before_strat.total, hue = win_fig2.figure.set_size_inches(8,5)
# Looking at the wins before stratification rounds wise
win_by_round_before_strat = pd.crosstab(df3.win, df3.Round).apply( lambda x: x/x.sum(), axis = 0 )
win_by_round_before_strat = pd.DataFrame(win_by_round_before_strat.unstack() ).reset_index()
win_by_round_before_strat.columns = ["Round", "win", "total" ]
2
fig2 = sns.barplot(win_by_round_before_strat.Round, win_by_round_before_strat.total, hue = win_by_round_fig2.figure.set_size_inches(8,5)
#dataset is uneven in terms of frequency of wins, to balance this using stratified sampling procedure
y_0 = df3[df3.win == 0]
y_1 = df3[df3.win == 1]
n = min([len(y_0), len(y_1)])
y_0 = y_0.sample(n = n, random_state = 0)
y_1 = y_1.sample(n = n, random_state = 0)
df_strat = pd.concat([y_0, y_1])
X_strat = df_strat[['Date', 'Surface', 'Round','WRank', 'LRank']]
y_strat = df_strat.win
X_strat.head()
y_strat.head()
X_strat_1=X_strat.copy()
X_strat_1['win']=y_strat
X_strat_1.head()
# For P1,P2 Rank1 > Rank2
df = X_strat_1.copy()
df["P1"] = df[["WRank", "LRank"]].max(axis=1)
df["P2"] = df[["WRank", "LRank"]].min(axis=1)
df.head()
#------------------------------------------------------------------------------------------------------------
# Looking at wins surface wise
win_by_Surface = pd.crosstab(df.win, df.Surface).apply( lambda x: x/x.sum(), axis = 0 )
win_by_Surface
win_by_Surface = pd.DataFrame( win_by_Surface.unstack() ).reset_index()
win_by_Surface.columns = ["Surface", "win", "total" ]
fig2 = sns.barplot(win_by_Surface.Surface, win_by_Surface.total, hue = win_by_Surface.win )
fig2.figure.set_size_inches(8,5)
# results: Clay court upsets are more often than the other two surfaces
# Looking at wins rounds wise
win_by_round = pd.crosstab(df.win, df.Round).apply( lambda x: x/x.sum(), axis = 0 )
win_by_round
win_by_round = pd.DataFrame(win_by_round.unstack() ).reset_index()
win_by_round.columns = ["Round", "win", "total" ]
fig2 = sns.barplot(win_by_round.Round, win_by_round.total, hue = win_by_round.win )
fig2.figure.set_size_inches(8,5)
# A slight trend of higher ranked player winning emerges from semifinals and finals
#To keep the dataframe cleaner we transform the Round entries into numbers. We then transform rounds df1 = df.copy()
def round_number(x):
if x == '1st Round':
return 1
elif x == '2nd Round':
return 2
elif x == '3rd Round':
return 3
elif x == '4th Round':
return 4
elif x == 'Quarterfinals':
return 5
3
elif x == 'Semifinals':
return 6
elif x == 'The Final':
return 7
df1['Round'] = df1['Round'].apply(round_number)
dummy_ranks = pd.get_dummies(df1['Round'], prefix='Round')
df1 = df1.join(dummy_ranks.loc[:, 'Round_2':])
df1[['Round_2', 'Round_3', 'Round_4', 'Round_5', 'Round_6', 'Round_7']] = df1[['Round_2', 'Round_3'
df1.head()
#Performing the same for surface
dummy_ranks = pd.get_dummies(df1['Surface'], prefix='Surface')
dummy_ranks.head()
df_2 = df1.join(dummy_ranks.loc[:, 'Surface_Grass':])
df_2.drop("Surface",axis = 1,inplace=True)
df_2[['Surface_Grass','Surface_Hard']] = df_2[['Surface_Grass','Surface_Hard']].astype('int_')
df_2.drop("Round",axis = 1,inplace=True)
df_2.head()
df4 = df_2.copy()
df4['P1'] = np.log2(df4['P1'].astype('float64'))
df4['P2'] = np.log2(df4['P2'].astype('float64'))
df4['D'] = df4['P1'] - df4['P2']
df4['D'] = np.absolute(df4['D'])
df4.head()
import numpy as np
import matplotlib.pyplot as plt
# Logit Function
def logit(x):
return np.exp(x) / (1 + np.exp(x))
x = np.linspace(-6,6,50, dtype=float)
y = logit(x)
plt.plot(x, y)
plt.ylabel("Probability")
plt.show()
df4.columns.tolist()
feature_cols = ['Round_2', 'Round_3', 'Round_4', 'Round_5', 'Round_6', 'Round_7', 'Surface_Grass',
dfnew = df4.copy()
dfnew[feature_cols].head()
with pd.option_context('mode.use_inf_as_na', True):
dfnew = dfnew.dropna(subset=['D'], how='all')
X = dfnew[feature_cols]
y = dfnew.win
# Performing test_train split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#---------------------Data Procesing Ends------------------------------------------------------------------------
# 1. LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
filename = 'lr.sav'
4
pickle.dump(logreg, open(filename, 'wb'))
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
print('True:', y_test.values[0:40])
print('Pred:', y_pred_class[0:40])
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
auc_score
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
fig = plt.plot(fpr, tpr,label='ROC curve (area = %0.2f)' % auc_score )
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for win classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.grid(True)
import statsmodels.api as sm
X = dfnew[feature_cols]
X = sm.add_constant(X)
y = dfnew['win']
lm = sm.Logit(y, X)
result = lm.fit()
result.summary()
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()
logreg.fit(dfnew[["D"]],dfnew["win"])
pred_probs = logreg.predict_proba(dfnew[["D"]])
filename = 'lr_cv.sav'
pickle.dump(logreg, open(filename, 'wb'))
plt.scatter(dfnew["D"], pred_probs[:,1])
plt.title('Win Probability for 5 sets matches')
plt.xlabel('D')
plt.ylabel('Win Probability for 5 sets matches')
plt.legend(loc="lower right")
plt.grid(True)
#------------------------------------------------------------------------------
# 2. DECISION TREES
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
X = dfnew[feature_cols].dropna()
y = dfnew['win']
model.fit(X, y)
#Performing cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print('AUC {}, Average AUC {}'.format(scores, scores.mean()))
model = DecisionTreeClassifier(
max_depth = 10,
5
min_samples_leaf = 8)
model.fit(X, y)
filename1 = 'dt.sav'
pickle.dump(model, open(filename1, 'wb'))
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print('CV AUC {}, Average AUC {}'.format(scores, scores.mean()))
#------------------------------------------------------------------------------
# 3. RANDOM FORESTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X = dfnew[feature_cols].dropna()
y = dfnew['win']
model1 = RandomForestClassifier(n_estimators = 200)
model1.fit(X, y)
filename2 = 'rf.sav'
pickle.dump(model1, open(filename2, 'wb'))
features = X.columns
feature_importances = model1.feature_importances_
features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
features_df.sort_values('Importance Score', inplace=True, ascending=False)
features_df
feature_importances = pd.Series(model1.feature_importances_, index=X.columns)
#feature_importances.sort()
feature_importances.plot(kind="barh", figsize=(7,6))
#Performing cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model1, X, y, scoring='roc_auc', cv= 5)
print('AUC {}, Average AUC {}'.format(scores, scores.mean()))
for n_trees in range(1, 200, 10):
model = RandomForestClassifier(n_estimators = n_trees)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print('n trees: {}, CV AUC {}, Average AUC {}'.format(n_trees, scores, scores.mean()))
#-------------------------------------------------------------------------------------------------------------------------
#Restricting analysis to best of 3
# ------------------------------------------------------------------------------------------------------------------------
import pandas as pd
df_atp = pd.read_csv('Data.csv')
df_atp['Date'] = pd.to_datetime(df_atp['Date'])
# Restricing dates
df_atp = df_atp.loc[(df_atp['Date'] > '2014-11-09') & (df_atp['Date'] <= '2016-11-09')]
# Keeping only completed matches
df_atp = df_atp[df_atp['Comment'] == 'Completed'].drop("Comment",axis = 1)
# Rename Best of to Best_of
df_atp.rename(columns = {'Best of':'Best_of'},inplace=True)
# Choosing features
cols_to_keep = ['Date','Series','Surface', 'Round','Best_of', 'WRank','LRank']
# Dropping NaN
df_atp = df_atp[cols_to_keep].dropna()
# Dropping errors in the dataset and unimportant entries (e.g. there are very few entries for Masters df_atp = df_atp[(df_atp['LRank'] != 'NR') & (df_atp['WRank'] != 'NR') & (df_atp['Series'] != 'Masters df_atp[['Best_of','WRank','LRank']] = df_atp[['Best_of','WRank','LRank']].astype(int)
6
def win(x):
if x > 0:
return 0
elif x <= 0:
return 1
df_atp['win'] = (df_atp['WRank'] - df_atp['LRank']).apply(win)
df_atp.head()
newdf = df_atp.copy()
newdf2 = newdf[(newdf['WRank'] <= 100) & (newdf['LRank'] <= 100)]
newdf2.head()
df3 = newdf2.copy()
df3 = df3[df3['Best_of'] == 3]
# Drop Best_of and Series columns
df3.drop("Series",axis = 1,inplace=True)
df3.drop("Best_of",axis = 1,inplace=True)
df3.head()
y_0 = df3[df3.win == 0]
y_1 = df3[df3.win == 1]
n = min([len(y_0), len(y_1)])
y_0 = y_0.sample(n = n, random_state = 0)
y_1 = y_1.sample(n = n, random_state = 0)
#Without stratification
win_by_Surface = pd.crosstab(df3.win, df3.Surface).apply( lambda x: x/x.sum(), axis = 0 )
win_by_Surface
win_by_Surface = pd.DataFrame( win_by_Surface.unstack() ).reset_index()
win_by_Surface.columns = ["Surface", "win", "total" ]
fig2 = sns.barplot(win_by_Surface.Surface, win_by_Surface.total, hue = win_by_Surface.win )
fig2.figure.set_size_inches(8,5)
#without stratification
win_by_round = pd.crosstab(df.win, df.Round).apply( lambda x: x/x.sum(), axis = 0 )
win_by_round
win_by_round = pd.DataFrame(win_by_round.unstack() ).reset_index()
win_by_round.columns = ["Round", "win", "total" ]
fig2 = sns.barplot(win_by_round.Round, win_by_round.total, hue = win_by_round.win )
fig2.figure.set_size_inches(8,5)
# Adding stratification
df_strat = pd.concat([y_0, y_1])
X_strat = df_strat[['Date', 'Surface', 'Round','WRank', 'LRank']]
y_strat = df_strat.win
X_strat_1=X_strat.copy()
X_strat_1['win']=y_strat
X_strat_1.head()
df = X_strat_1.copy()
df["P1"] = df[["WRank", "LRank"]].max(axis=1)
df["P2"] = df[["WRank", "LRank"]].min(axis=1)
df.head()
df1 = df.copy()
7
def round_number(x):
if x == '1st Round':
return 1
elif x == '2nd Round':
return 2
elif x == '3rd Round':
return 3
elif x == '4th Round':
return 4
elif x == 'Quarterfinals':
return 5
elif x == 'Semifinals':
return 6
elif x == 'The Final':
return 7
df1['Round'] = df1['Round'].apply(round_number)
dummy_ranks = pd.get_dummies(df1['Round'], prefix='Round')
df1 = df1.join(dummy_ranks.loc[:, 'Round_2':])
df1[['Round_2', 'Round_3',
'Round_4', 'Round_5', 'Round_6', 'Round_7']] = df1[['Round_2', 'Round_3','Round_4', 'Round_5'
df1.head()
dummy_ranks = pd.get_dummies(df1['Surface'], prefix='Surface')
df_2 = df1.join(dummy_ranks.loc[:, 'Surface_Grass':])
df_2.drop("Surface",axis = 1,inplace=True)
df_2[['Surface_Grass','Surface_Hard']] = df_2[['Surface_Grass','Surface_Hard']].astype('int_')
df_2.drop("Round",axis = 1,inplace=True)
df_2.head()
df4 = df_2.copy()
df4['P1'] = np.log2(df4['P1'].astype('float64'))
df4['P2'] = np.log2(df4['P2'].astype('float64'))
df4['D'] = df4['P1'] - df4['P2']
df4['D'] = np.absolute(df4['D'])
df4.head()
#---------------------------Data processing ends --------------------------------------------------------------------
# 1. LOGISTIC REGRESSION
df4.columns.tolist()
feature_cols = ['D','Surface_Hard','Surface_Grass','Round_6','Round_5','Round_3']
dfnew = df4.copy()
with pd.option_context('mode.use_inf_as_na', True):
dfnew = dfnew.dropna(subset=['D'], how='all')
X = dfnew[feature_cols]
y = dfnew.win
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
8
print(auc_score)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
auc_score = metrics.roc_auc_score(y_test, y_pred_prob)
auc_score
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
fig = plt.plot(fpr, tpr,label='ROC curve (area = %0.2f)' % auc_score )
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for win classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend(loc="lower right")
plt.grid(True)
from sklearn.model_selection import cross_val_score
cross_val_score(logreg, X, y, cv=5, scoring='roc_auc').mean()
logreg.fit(dfnew[["D"]],dfnew["win"])
pred_probs = logreg.predict_proba(dfnew[["D"]])
plt.scatter(dfnew["D"], pred_probs[:,1])
plt.title('Win Probability for 3 sets matches')
plt.xlabel('D')
plt.ylabel('Win Probability for 3 sets matches')
plt.legend(loc="lower right")
plt.grid(True)
#------------------------------------------------------------------------------
# 2, DECISION TREES
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
X = dfnew[feature_cols].dropna()
y = dfnew['win']
model.fit(X, y)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print('AUC {}, Average AUC {}'.format(scores, scores.mean()))
model = DecisionTreeClassifier(
max_depth = 10,
min_samples_leaf = 8)
model.fit(X, y)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=5)
print('CV AUC {}, Average AUC {}'.format(scores, scores.mean()))
#------------------------------------------------------------------------------
# 3. RANDOM FORESTS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X = dfnew[feature_cols].dropna()
y = dfnew['win']
model = RandomForestClassifier(n_estimators = 200)
model.fit(X, y)
features = X.columns
feature_importances = model.feature_importances_
features_df = pd.DataFrame({'Features': features, 'Importance Score': feature_importances})
features_df.sort_values('Importance Score', inplace=True, ascending=False)
9
features_df
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
#feature_importances.sort()
feature_importances.plot(kind="barh", figsize=(7,6))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='roc_auc', cv = 5)
print('AUC {}, Average AUC {}'.format(scores, scores.mean()))
for n_trees in range(1, 200, 10):
model = RandomForestClassifier(n_estimators = n_trees)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv= 5)
print('n trees: {}, CV AUC {}, Average AUC {}'.format(n_trees, scores, scores.mean()))
10