import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt




DATA_PATH="C:/Users/Noa/Downloads/lab7 (2) (1)"




names_of_parties = {"פה": "כחול לבן", "מחל": "הליכוד", "ודעם": "הרשימה המשותפת", "שס": "שס", "ל": "ישראל ביתנו", "ג": "יהדות התורה",
           "טב": "ימינה", "אמת": "העבודה גשר", "מרצ": "המחנה הדמוקרטי", "כף": "עוצמה יהודית"}

full_names=[ "כחול לבן",  "הליכוד",  "הרשימה המשותפת",  "שס",  "ישראל ביתנו",  "יהדות התורה",
            "ימינה",  "העבודה גשר",  "המחנה הדמוקרטי",  "עוצמה יהודית"]


df_hev = pd.read_csv(os.path.join(DATA_PATH, r'HevratiCalcaliYeshuvim.csv'), encoding = 'iso-8859-8', index_col='רשות מקומית').sort_index()  # , encoding='utf-8')


analysis = 'city'  # ballot
df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per '+analysis+' 2019b.csv'), \
                         encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep = df_sep_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
df_sep = df_sep[df_sep.index != 'מעטפות חיצוניות']
if analysis == 'city':
    first_col = 5
else:
    first_col = 9
df_sep = df_sep[df_sep.columns[first_col:]]  # removing "metadata" columns
df_sep_raw2 = df_sep_raw[df_sep_raw.index != 'מעטפות חיצוניות']


#Builind a df for shared cities in both votes and social data frame
shared_cities =list(df_hev.index.intersection(df_sep.index))
df_merged=pd.concat([df_hev.loc[shared_cities],df_sep.loc[shared_cities]],axis=1)
print("number of shared_cities",str(len(shared_cities)))

def ten_parties_votes(df):
    par = df.sum().sort_values(ascending=False)
    return par[:10]
names = ten_parties_votes(df_sep).keys()
top_ten = df_sep[names]

df_merged_parties=df_merged[names]
p_real=top_ten.sum().div(top_ten.sum().sum())
p_merged=df_merged_parties.sum().div(df_merged_parties.sum().sum())


#Plotting the parties bars, full data vs merged cities
def parties_bar(real,merged):
    width = 0.3

    n=len(real) # number of parties
    rev_names = [name[::-1] for name in list(names)]
    fig, ax = plt.subplots()  # plt.subplots()

    merged_bar = ax.bar(np.arange(n)+width, list(merged), width, color='r')
    real_bar =  ax.bar(np.arange(n), list(real), width, color='b')

    ax.set_ylabel('Votes percent')
    ax.set_xlabel('Parties Names')
    ax.set_title('Votes percent real vs merged')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(rev_names)
    ax.legend((merged_bar, real_bar), ('merged',"real" ))
    plt.show()

    return fig, ax

parties_bar(p_real,p_merged)



def eshkol(df):
    P_dem = np.zeros([10,10])  # P_dem[i][j] is voting frequency for party j at eshkol i
    for i in range(10):
    # Find all cities in current eshkol. You can use np.where (similar to R which)
        eshk=i+1
        cur_cities_T_f=df_merged['מדד חברתי-']==str(eshk)
        cur_cities=df_merged[cur_cities_T_f]
        cur_cities=cur_cities[names]

    # Compute parties frequencies for cities in current eshkol
        P_dem[i,]  = cur_cities.sum().div(cur_cities.sum().sum())

    return P_dem

p_eshkol=eshkol(df_merged)
p_eshkol = pd.DataFrame(p_eshkol)
# Bar Plot votes for each eshkol in a different subplot. Use plt.subplots
p_eshkol.columns=full_names
p_eshkol.index=[1,2,3,4,5,6,7,8,9,10]

def plots(df,real):
    width=0.5
    n = len(real)  # number of parties
    rev_names = [name[::-1] for name in list(full_names)]
    fig, ax = plt.subplots(2, 5, figsize=(50, 20))
  #  ax = ax.T.flatten()
    counter=0
    for i in range(2):
        for j in range(5):

            cur_eshkol=p_eshkol.iloc[counter]
            cur_bar = ax[i,j].bar(np.arange(n) + width, list(cur_eshkol), width, color='r')
            real_bar = ax[i,j].bar(np.arange(n), list(real), width, color='b')

            ax[i,j].set_ylabel('percent')
            ax[i,j].set_xlabel( 'P_Names')
            counter += 1
            cur= 'real vs eshcol',str(counter)
            ax[i,j].set_title(cur)
            ax[i,j].set_xticks(np.arange(n))
            ax[i,j].set_xticklabels(rev_names,rotation = 90)
            ax[i,j].legend((cur_bar, real_bar), ('e', "r"))
    plt.show()

    return fig, ax


#Plotting every Eskol (socio economical level) voting for diffarent parties.
plots(p_eshkol,p_real)



def plots_parties(df,real):
    width=0.5
    n = len(real)  # number of parties
    rev_names = [name[::-1] for name in list(full_names)]
    fig, ax = plt.subplots(2, 5, figsize=(50, 20))
  #  ax = ax.T.flatten()
    counter=0
    for i in range(2):
        for j in range(5):

            cur_eshkol=p_eshkol[full_names[counter]]
            cur_bar = ax[i,j].bar(np.arange(n) + width, list(cur_eshkol), width, color='g')

            ax[i,j].set_ylabel('percent')
            xlabel=str(rev_names[counter])
            ax[i,j].set_xlabel( xlabel)
            counter += 1
            cur= 'party',xlabel
     #       ax[i,j].set_title(cur)
            ax[i,j].set_xticks(np.arange(n))
            ax[i,j].set_xticklabels(df.index,)
            ax[i,j].legend((cur_bar), (cur))
    plt.show()

    return fig, ax

#Ploting each party distribution per Eskol
plots_parties(p_eshkol,p_real)




analysis = 'ballot'
df_sep_raw = pd.read_csv(os.path.join(DATA_PATH, r'votes per '+analysis+' 2019b.csv'), encoding = 'iso-8859-8', index_col='שם ישוב').sort_index()
df_sep_bal = df_sep_raw.drop('סמל ועדה', axis=1)  # new column added in Sep 2019
df_sep_bal = df_sep_bal[df_sep_bal.index != 'מעטפות חיצוניות']
if analysis == 'city':
    first_col = 5
else:
    first_col = 9
df_sep_bal = df_sep_bal[df_sep_bal.columns[first_col:]]  # removing "metadata" columns

H = pd.DataFrame({'City':shared_cities, 'Het':np.zeros(len(shared_cities)), \
                  'Size':np.zeros(len(shared_cities))}).set_index('City')

df_merged_bar=df_merged[df_merged.columns[13:]]

 ###calculation
df_sep_bal2 = df_sep_bal.loc[df_merged.index]
p_ballot = df_sep_bal.apply(lambda x: x/x.sum(),axis=1)
p_city = df_merged_bar.apply(lambda x: x/x.sum(),axis=1)
df_hev_shared = df_hev.loc[df_merged.index]

#Creating a df for the hetroginisity on the cities
def d_hetero(shared_cities_df, p_ballot, p_city, df_hev_shared):
    ind = shared_cities_df.index
    final = pd.DataFrame(np.zeros((len(ind), 1)))
    final = final.set_index(ind)

    for i in range(len(ind)):
        ballot_j = p_ballot.loc[ind[i]]
        ballot_city_dist = (ballot_j - p_city.loc[ind[i]])
        dist_transpose = np.transpose(ballot_city_dist)
        ans = (ballot_city_dist.as_matrix() @ dist_transpose.as_matrix())

        # if result matrix is 1x1 dimension
        if ans.shape != ():
            ans = ans.diagonal()
            ans = sum(ans) / len([ans])

        final.loc[ind[i]] = ans
    gini = df_hev_shared["מדד ג'יני[2]"]
    population = df_hev_shared["אוכלוסייה[1]"].str.replace('\W', '').astype(int)
    kmr = df_hev_shared['דירוג.2'].str.replace('\W', '').astype(int)
    final_df = pd.concat([final, gini, population, kmr], axis=1)
    final_df.columns = ['dist', 'gini', 'pop', 'kmr']
    return final_df


hetro_df = d_hetero(df_merged,p_ballot,p_city,df_hev_shared)

#A scatter plot for the GINI vs Hetroginisity
def hetero_scatter_plot(df):
    cm = plt.cm.get_cmap('viridis_r')
    scat = plt.scatter(x=df['dist'], y=df['gini'],
                       c=np.log(df['pop']), cmap=cm,alpha=.7, s=(df['kmr'])/2)
    plt.colorbar(scat)
    plt.title("Heterogeneity vs. GINI")
    plt.xlabel("Heterogeneity")
    plt.ylabel("GINI")
    plt.legend(scatterpoints=1,loc='upper right',  ncol=1, fontsize=8,title="Size = population/kmr\nColor: Log(Population)")
    plt.show()
#print plot
hetero_scatter_plot(hetro_df)

#Spearman Correlation check.
print(hetro_df.corr(method='spearman'))
   

    
    
