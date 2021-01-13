# -- coding: utf-8 --
"""
Created on Wed Dec 11 12:10:46 2019

@author: noagr
"""
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import stats

from sklearn.decomposition import PCA
import seaborn as sns
from scipy.optimize import nnls as nnls
from sklearn.metrics import mean_squared_error as MSE


DATA_PATH="C:/Users/Noa/Downloads/lab7 (2) (1)"

def read_election_results(year, analysis):
    df_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per ' + analysis + ' ' + year + '.csv'),
                             encoding='iso-8859-8', index_col='שם ישוב').sort_index()
    if year == '2019b':
        df = df_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
    else:
        df = df_raw

    df = df[df.index != 'מעטפות חיצוניות']  # removing external envelops from the data
    if analysis == 'city':
        first_col = 5
    else:  # ballot
        first_col = 6
    df = df[df.columns[first_col:]]  # removing "metadata" columns

    return df, df_raw


# Read data for differnt elections
df_sep, df_sep_raw = read_election_results('2019b', 'ballot')
df_april, df_april_raw = read_election_results('2019a', 'ballot')

# Drop all the nonparties colouns
df_sep = df_sep.drop("כשרים", axis=1)
df_sep = df_sep.drop("מצביעים", axis=1)
df_sep = df_sep.drop("פסולים", axis=1)

#Taking only the biggest 10 parties from sep elections.
p_sep = df_sep.sum().div(df_sep.sum().sum()).sort_values(ascending=False)
big_parties_sep = p_sep[0:10].keys()
df_sep = df_sep[big_parties_sep]

#Taking the biggest 14 parties from April elections
p_April=df_april.sum().div(df_april.sum().sum()).sort_values(ascending=False)
big_parties_april = p_April[0:14].keys()
df_april=df_april[big_parties_april]


def adapt_df(df, parties, ballot_number_field_name=None):
    df['ballot_id'] = df['סמל ישוב'].astype(str) + '__' + df[ballot_number_field_name].astype(str)
    df = df.set_index('ballot_id')
    df = df[parties]
    df = df.reindex(sorted(df.columns), axis=1)
    return df

#Adding a uniuqe column for each df as an index column
df_sep_new = adapt_df(df_sep_raw,df_sep.columns, 'קלפי')
df_april_new = adapt_df(df_april_raw,df_april.columns, 'מספר קלפי')

#Finding the shared rows from the two df.
df_shared=df_sep_new.index.intersection(df_april_new.index)
Nb =(df_sep_new.loc[df_shared])
Na =(df_april_new.loc[df_shared])

#Vectors for parties names
April_names =['טב','ז','נ','ג','שס','מחל','כ','ל','פה','נר','אמת','מרצ','דעם','ום']
Sep_names =['כף','טב','ג','שס','מחל','ל','פה','אמת','מרצ','ודעם']

#Creating a new Data Frames, including only parties on shared ballots from both dataframes.
Na = Na[April_names]
Nb = Nb[Sep_names]

#Transposeing the DFS
Nb_t = pd.DataFrame.transpose(Nb)
Na_t= pd.DataFrame.transpose(Na)

#Converting the dfs to matrixs in order to use math funcations.
Nb_t = pd.DataFrame.as_matrix(Nb_t)
Na_t = pd.DataFrame.as_matrix(Na_t)

Nb =pd.DataFrame.as_matrix(Nb)
Na =pd.DataFrame.as_matrix(Na)

#Multiple NA_transpose by NA
NATNA =np.linalg.inv(Na_t.dot(Na))

#Calculating the predicted M matrix - the trasition matrix. (Nb_transpose*NA*NA_transpose*NA)
M = Nb_t.dot(Na).dot(NATNA)

#Creating vectors of explicit parties names
full_names_ap=["איחוד מפלגות הימין","זהות","הימין החדש","יהדות התורה","שס","הליכוד","כולנו","ישראל ביתנו","כחול לבן","גשר","עבודה","מרץ","תעל","חדש"]
full_names_sep=["עוצמה יהודית","ימינה","יהדות התורה","שס","הליכוד","ישראל ביתנו","כחול לבן","עבודה-גשר","מרץ","הרשימה המשותפת"]

#Reversing names (to correct initial problem with the data)
rev_names_ap = [name[::-1] for name in list(full_names_ap)]
rev_names_sep = [name[::-1] for name in list(full_names_sep)]

M= pd.DataFrame(M,index=rev_names_sep, columns = rev_names_ap)
M= pd.DataFrame.transpose(M)
#Ploting the M matrix as an Heatmap.
ax = plt.axes()
sns.heatmap(M, annot=True, cmap="BuPu",ax=ax)
ax.set_title('Heat Map of the transition matrix - M^')
plt.show()

#Fixing the matrix: removing small values and changing the rows sum to be equal to 1.
##1.b
M[M<0.05]=0
M_fixed = M.div(M.sum(axis = 1), axis = 0)
#Plotting the fixed matrix as an Heatmap.
ax = plt.axes()
sns.heatmap(M_fixed, annot=True, cmap="BuPu",ax=ax)
ax.set_title('Heat Map of M^ - fixed version')
plt.show

##Q.2
#Adding to the matrix a column of "not voted" 
Not_voted_sep =pd.DataFrame(df_sep_raw["בזב"]-df_sep_raw["מצביעים"])
Not_voted_april =pd.DataFrame(df_april_raw["בזב"]-df_april_raw["מצביעים"])

Not_voted_sep["סמל ישוב"]=df_sep_raw["סמל ישוב"]
Not_voted_april["סמל ישוב"]=df_april_raw["סמל ישוב"]
Not_voted_sep["קלפי"]=df_sep_raw["קלפי"]
Not_voted_april["קלפי"]=df_april_raw["מספר קלפי"]

Not_voted_april["id"] = Not_voted_april["סמל ישוב"].map(str).str.cat(Not_voted_april["קלפי"].map(str),sep="__")
Not_voted_sep["id"] = Not_voted_sep["סמל ישוב"].map(str).str.cat(Not_voted_sep["קלפי"].map(str),sep="__")

Not_voted_april.index= Not_voted_april["id"]
Not_voted_sep.index= Not_voted_sep["id"]

#Taking only the shared values from both dfs.
Not_voted_april=Not_voted_april.loc[df_shared]
Not_voted_sep=Not_voted_sep.loc[df_shared]

Na_2 = pd.DataFrame(Na)
Nb_2 =pd.DataFrame(Nb)

Na_2["Not_voted"]=Not_voted_april[0].values
Nb_2["Not_voted"]=Not_voted_sep[0].values

#Converting the DF to matrix in order to use math funcations.
Na_2 = pd.DataFrame.as_matrix(Na_2)
Nb_2 = pd.DataFrame.as_matrix(Nb_2)

NATNA_2 = np.linalg.inv(Na_2.transpose().dot(Na_2))
M_2 = Nb_2.transpose().dot(Na_2).dot(NATNA_2)

april_names =['טב','ז','נ','ג','שס','מחל','כ','ל','פה','נר','אמת','מרצ','דעם','ום',"לא הצביעו"]
sep_names =['כף','טב','ג','שס','מחל','ל','פה','אמת','מרצ','ודעם', "לא הצביעו"]

full_names_ap=["איחוד מפלגות הימין","זהות","הימין החדש","יהדות התורה","שס","הליכוד","כולנו","ישראל ביתנו","כחול לבן","גשר","עבודה","מרץ","תעל","חדש","לא הצביעו"]
full_names_sep=["עוצמה יהודית","ימינה","יהדות התורה","שס","הליכוד","ישראל ביתנו","כחול לבן","עבודה-גשר","מרץ","הרשימה המשותפת","לא הצביעו"]

rev_names_ap = [name[::-1] for name in list(full_names_ap)]
rev_names_sep = [name[::-1] for name in list(full_names_sep)]

M_2= pd.DataFrame(M_2,index=rev_names_sep, columns = rev_names_ap)
M_2= pd.DataFrame.transpose(M_2)
#Ploting the new matrix with the not voted column.
ax = plt.axes()
sns.heatmap(M_2, annot=True, cmap="BuPu",ax=ax)
ax.set_title('matrix M^ with the unvoted rates')
ax.set_ylabel('April Elections')
ax.set_xlabel('September Elections')
plt.show

M_2[M_2<0.05]=0
M2_fixed = M_2.div(M_2.sum(axis = 1), axis = 0)

ax = plt.axes()
sns.heatmap(M2_fixed, annot=True, cmap="BuPu",ax=ax)
ax.set_title("matrix M^ with the unvoted rates - fixed version")
ax.set_ylabel('April Elections')
ax.set_xlabel('September Elections')
plt.show

##Q3
Nb =(df_sep_new.loc[df_shared])
Na =(df_april_new.loc[df_shared])

april_names =['טב','ז','נ','ג','שס','מחל','כ','ל','פה','נר','אמת','מרצ','דעם','ום']
sep_names =['כף','טב','ג','שס','מחל','ל','פה','אמת','מרצ','ודעם']
Na = Na[april_names]
Nb = Nb[sep_names]
Na.columns = M.index
Nb.columns = M.columns

#Q3
#Calculating the new M matrix using NNLS from Scipy library.
M_nnls = M*0
for i in M_nnls.columns:
    tmp = nnls(Na, Nb[str(i)])
    tmp = np.asarray(tmp[0])
    M_nnls[str(i)] = tmp
#Ploting the new M 
ax = sns.heatmap(M_nnls, annot=True, cmap="BuPu")
ax.set_title('Heat-Map Non Negative least squares')
ax.set_ylabel('April Elections')
ax.set_xlabel('September Elections')
plt.show()

M_nnls[M_nnls <= 0.05] = 0
M_nnls = M_nnls.div(M_nnls.sum(axis = 1), axis = 0)
ax = sns.heatmap(M_nnls, annot=True, cmap="BuPu")
ax.set_title('Heat-Map Non Negative least squares - fixed version')
ax.set_ylabel('April Elections')
ax.set_xlabel('September Elections')
plt.show()


#Q4
#creating the Residuals matrix.
Nb_2_df = pd.DataFrame(Nb_2, columns = M_2.columns, index = df_shared)
#Creating the MSE matrix.
Y_hat = Na_2.dot(pd.DataFrame.as_matrix(M_2))
y_hat_df= pd.DataFrame(Y_hat, columns = M_2.columns, index = df_shared)
mse_parties={}
for i in M_2.columns:
    tmp = MSE(Nb_2_df[str(i)],y_hat_df[str(i)])
    mse_parties[i] = tmp

mse_parties = pd.DataFrame.from_dict(mse_parties ,orient= "index")
mse_parties.columns =  ["rate"]

def parties_bar(df):
    n = len(mse_parties)  # number of parties
    names = mse_parties.index
    fig, ax = plt.subplots()  # plt.subplots()
    party_bar = ax.bar(np.arange(n),list(mse_parties["rate"]), color='purple')
    ax.set_ylabel('MSE rates')
    ax.set_xlabel('Parties Names')
    ax.set_title('MSE rates per party')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(names, rotation = 90)
    plt.show()
    return fig, ax
#ploting the MSE bar plot.
parties_bar(mse_parties)