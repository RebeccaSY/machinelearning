import pandas as pd
import numpy as np
import sys
import json
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns

train_csv = sys.argv[1]
validate_csv = sys.argv[2]

train_data = pd.read_csv(train_csv)
validate_data = pd.read_csv(validate_csv)

#train_data = pd.read_csv('training.csv')
#validate_data = pd.read_csv('validation.csv')

'''
# overall info, to see how much lost data there is
print(train_data.info())
print(validate_data.info())
'''

max_revenue_train = max(train_data['revenue'])
high_revenue = 0.2 * max_revenue_train  # a relatively high revenue


# object: cast, crew, genres, keywords, production_companies, production_countries, spoken_languages

# get release year
def get_yearmonth(df):
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['year'] = df['release_date'].dt.year
    #df['month'] = df['release_date'].dt.month
    
# get ratio between budget and year 
def get_budgetyear(df):
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['year'] = df['release_date'].dt.year
    # adjust the ratio because the scale of budget is much larger than that of year
    df['budget_year'] = (df['budget']) ** (1/3) / (df['year'] - 1900)
    

# get first genre
def get_genres(df):
    df['genre1'] = [0 for i in range(len(df['genres']))]
    #df['genre2'] = [0 for i in range(len(df['genres']))]
    for i in range(len(df['genres'])):
        genres = df['genres'][i]   # list of genres (Object)
        str_genres = json.loads(genres)       # str of genres
        if str_genres != []:            
            genre1 = str_genres[0]                     # dict of 1st genre
            df['genre1'][i] = list(genre1.values())[0] # id of 1st genre
            '''
            if len(str_genres) < 2:
                genre2 = str_genres[0]         # dict of still the 1st genre
                df['genre2'][i] = list(genre2.values())[0] # still id of 1st genre
            else:
                genre2 = str_genres[1]         # dict of 2nd genre
                df['genre2'][i] = list(genre2.values())[0] # id of 2nd genre
            '''

# get the first six cast (name & order)
def get_cast(df):
    df['cast1'] = [0 for i in range(len(df['cast']))]
    df['cast2'] = [0 for i in range(len(df['cast']))]
    #df['cast3'] = [0 for i in range(len(df['cast']))]
    #df['cast4'] = [0 for i in range(len(df['cast']))]
    #df['cast5'] = [0 for i in range(len(df['cast']))]
    #df['cast6'] = [0 for i in range(len(df['cast']))]
    for i in range(len(df['cast'])):

        cast = df['cast'][i]        # list of cast
        str_cast = json.loads(cast) # convert to str
        if str_cast != []:
            cast1 = str_cast[0]
            df['cast1'][i] = list(cast1.values())[5]
            #df['cast1'][i] = list(cast1.values())[5] + ' ' + str(list(cast1.values())[6])    # str: name & order
            if len(str_cast) < 2:
                df['cast2'][i] = df['cast1'][i]
            else:
                cast2 = str_cast[1]
                df['cast2'][i] = list(cast2.values())[5]
                #df['cast2'][i] = list(cast2.values())[5] + ' ' + str(list(cast2.values())[6])    # str: name & order
            '''
            if len(str_cast) < 3:
                df['cast3'][i] = df['cast2'][i]
            else:
                cast3 = str_cast[2]
                df['cast3'][i] = list(cast3.values())[5] + ' ' + str(list(cast3.values())[6])    # str: name & order
            
            if len(str_cast) < 4:
                df['cast4'][i] = df['cast3'][i]
            else:
                cast4 = str_cast[3]
                df['cast4'][i] = list(cast4.values())[5] + ' ' + str(list(cast4.values())[6])    # str: name & order
            
            if len(str_cast) < 5:
                df['cast5'][i] = df['cast4'][i]
            else:
                cast5 = str_cast[4]
                df['cast5'][i] = list(cast5.values())[5] + ' ' + str(list(cast5.values())[6])    # str: name & order
                
            if len(str_cast) < 6:
                df['cast6'][i] = df['cast5'][i]
            else:
                cast6 = str_cast[5]
                df['cast6'][i] = list(cast6.values())[5] + ' ' + str(list(cast6.values())[6])    # str: name & order
            '''
    df['cast1'] = LabelEncoder().fit_transform(df['cast1'])
    df['cast2'] = LabelEncoder().fit_transform(df['cast2'])
    #df['cast3'] = LabelEncoder().fit_transform(df['cast3'])
    #df['cast4'] = LabelEncoder().fit_transform(df['cast4'])
    #df['cast5'] = LabelEncoder().fit_transform(df['cast5'])
    #df['cast6'] = LabelEncoder().fit_transform(df['cast6'])

# get director
def get_director(df):
    df['director'] = [0 for i in range(len(df['crew']))]
    for i in range(len(df['crew'])):
        crew = df['crew'][i]        # list of crew
        str_crew = json.loads(crew) # convert to str
        for j in range(len(str_crew)):
            if list(str_crew[j].values())[4] == "Director":
                # get the id of director
                df['director'][i] = list(str_crew[j].values())[3]


# get number of production companies
def get_num_companies(df):
    df['num_co'] = [0 for i in range(len(df['production_companies']))]
    for i in range(len(df['production_companies'])):
        pro_companies = df['production_companies'][i]   # list of production companies (Object)
        str_companies = json.loads(pro_companies)       # str of production companies
        df['num_co'][i] = len(str_companies)

# get the ratio between budget and number of production companies
def get_large_com(df):
    get_num_companies(df)
    df['budget_com'] = [0 for i in range(len(df['budget']))]
    for i in range(len(df['budget'])):
        if df['num_co'][i] != 0:
            df['budget_com'][i] = df['budget'][i] / df['num_co'][i]    # average budget per companies
        else:
            df['budget_com'][i] = 0
    df = df.drop(columns=['num_co'])

# get number of spoken languages
def get_num_sl(df):
    df['num_sl'] = [0 for i in range(len(df['spoken_languages']))]
    for i in range(len(df['spoken_languages'])):
        sl = df['spoken_languages'][i]      # list of spoken_languages (Object)
        str_sl = json.loads(sl)             # str of spoken_languages
        df['num_sl'][i] = len(str_sl)

# get length of tagline
def get_len_tag(df):
    df['len_tag'] = [0 for i in range(len(df['tagline']))]
    for i in range(len(df['tagline'])):
        if df['tagline'][i] == '':
            df['len_tag'][i] = 0
        else:
            df['tagline'][i] = str(df['tagline'][i])
            df['len_tag'][i] = len(df['tagline'][i])

# factorize homepage
def get_hp(df):
    df['hp'] = [0 for i in range(len(df['homepage']))]
    for i in range(len(df['homepage'])):
        if df['homepage'][i] == '':
            df['hp'][i] = 0
        else:
            df['hp'][i] = len(str(df['homepage'][i]))

# factorize overview
def get_overview(df):
    df['view'] = [0 for i in range(len(df['overview']))]
    for i in range(len(df['overview'])):
        if df['homepage'][i] == '':
            df['view'][i] = 0
        else:
            df['view'][i] = len(str(df['overview'][i]))

# get first production companies' id
def get_two_companies(df):
    '''
    df['first_company'] = [0 for i in range(len(df['production_companies']))]
    df['sec_company'] = [0 for i in range(len(df['production_companies']))]
    for i in range(len(df['production_companies'])):
        pro_companies = df['production_companies'][i]   # list of production companies (Object)
        str_companies = json.loads(pro_companies)       # str of production companies
        if str_companies != []:            
            first_co = str_companies[0]                     # dict of 1st company
            # print(i, first_co, len(str_companies))
            df['first_company'][i] = list(first_co.values())[1] # id of 1st company
            if len(str_companies) < 2:
                sec_co = str_companies[0]         # dict of still the 1st company
                df['sec_company'][i] = list(sec_co.values())[1] # still id of 1st company
            else:
                sec_co = str_companies[1]         # dict of 2nd company
                df['sec_company'][i] = list(sec_co.values())[1] # id of 2nd company
    '''
    df['first_company'] = [0 for i in range(len(df['production_companies']))]
    for i in range(len(df['production_companies'])):
        pro_companies = df['production_companies'][i]   # list of production companies (Object)
        str_companies = json.loads(pro_companies)       # str of production companies
        if str_companies != []:            
            first_co = str_companies[0]                     # dict of 1st company
            # print(i, first_co, len(str_companies))
            df['first_company'][i] = list(first_co.values())[1] # id of 1st company


# make a col for cast popularity 
# all info from train dataset
def popular_act(df, df_validate):    
    # make a list for "popular" actors id
    popular_list = []
    for i in range(len(df['cast'])):
        cast = df['cast'][i]        # list of cast
        str_cast = json.loads(cast) # convert to str
        if str_cast != [] and df['revenue'][i] > high_revenue:
            # add the id of main cast to the list
            for j in range(min(len(str_cast), 3)):
                if list(str_cast[j].values())[4] not in popular_list:
                    act_id = list(str_cast[j].values())[4]
                    popular_list.append(act_id)
    
    # create a col to indicate how "popular" the cast is
    # for both datasets
    df['contain_pop'] = [0 for i in range(len(df['cast']))]
    for i in range(len(df['cast'])):
        cast = df['cast'][i]        # list of cast
        str_cast = json.loads(cast) # convert to str
        if str_cast != []:
            for j in range(len(str_cast)):
                if list(str_cast[j].values())[4] in popular_list:
                    df['contain_pop'][i] = df['contain_pop'][i] + 1

    df_validate['contain_pop'] = [0 for i in range(len(df_validate['cast']))]
    for i in range(len(df_validate['cast'])):
        cast = df_validate['cast'][i]       # list of cast
        str_cast = json.loads(cast)         # convert to str
        if str_cast != []:
            for j in range(len(str_cast)):
                if list(str_cast[j].values())[4] in popular_list:
                    df_validate['contain_pop'][i] = df_validate['contain_pop'][i] + 1


# make a col for crew popularity 
# all info from train dataset
def popular_crew(df, df_validate):    
    # make a list for "popular" crew members id
    popular_list = []
    # consider main roles in the crew
    main_jobs = ["Producer", "Director", "Writer", "Production Supervisor", "Editor"]
    for i in range(len(df['crew'])):
        crew = df['crew'][i]        # list of crew
        str_crew = json.loads(crew) # convert to str
        if str_crew != [] and df['revenue'][i] > high_revenue:
            # add the id of main crew to the list
            crew_list = []
            for j in range(len(str_crew)):
                crew_list = list(str_crew[j].values())
                if crew_list[4] in main_jobs and crew_list[3] not in popular_list:
                    crew_id = crew_list[3]
                    popular_list.append(crew_id)
    
    # create a col to indicate how "popular" the crew is
    # for both datasets
    df['crew_pop'] = [0 for i in range(len(df['crew']))]
    for i in range(len(df['crew'])):
        crew = df['crew'][i]        # list of crew
        str_crew = json.loads(crew) # convert to str
        if str_crew != []:
            for j in range(len(str_crew)):
                if list(str_crew[j].values())[3] in popular_list:
                    df['crew_pop'][i] = df['crew_pop'][i] + 1

    df_validate['crew_pop'] = [0 for i in range(len(df_validate['crew']))]
    for i in range(len(df_validate['cast'])):
        crew = df_validate['crew'][i]       # list of crew
        str_crew = json.loads(crew)         # convert to str
        if str_crew != []:
            for j in range(len(str_crew)):
                if list(str_crew[j].values())[3] in popular_list:
                    df_validate['crew_pop'][i] = df_validate['crew_pop'][i] + 1


# make a col for popular genres
# all info from train dataset
def popular_genres(df, df_validate):    
    # make a list for "popular" genres id
    popular_list = []
    for i in range(len(df['genres'])):
        genres = df['genres'][i]        # list of genres
        str_genres = json.loads(genres) # convert to str
        if str_genres != [] and df['revenue'][i] > high_revenue:
            # add the id of first genre to the list
            genre_list = list(str_genres[0].values())
            if genre_list[0] not in popular_list:
                genre_id = genre_list[0]
                popular_list.append(genre_id)

    
    # create a col to indicate how "popular" the genre is
    # for both datasets
    df['genre_pop'] = [0 for i in range(len(df['genres']))]
    for i in range(len(df['genres'])):
        genres = df['genres'][i]        # list of genres
        str_genres = json.loads(genres) # convert to str
        if str_genres != []:
            for j in range(len(str_genres)):
                if list(str_genres[j].values())[0] in popular_list:
                    df['genre_pop'][i] = df['genre_pop'][i] + 1

    df_validate['genre_pop'] = [0 for i in range(len(df_validate['genres']))]
    for i in range(len(df_validate['genres'])):
        genres = df_validate['genres'][i]   # list of genres
        str_genres = json.loads(genres)     # convert to str
        if str_genres != []:
            for j in range(len(str_genres)):
                if list(str_genres[0].values())[0] in popular_list:
                    df_validate['genre_pop'][i] = df_validate['genre_pop'][i] + 1


# make a col for popular keywords
# all info from train dataset
def popular_keywords(df, df_validate):    
    # make a list for "popular" keywords id
    popular_list = []
    for i in range(len(df['keywords'])):
        keywords = df['keywords'][i]        # list of keywords
        str_keywords = json.loads(keywords) # convert to str
        if str_keywords != [] and df['revenue'][i] > high_revenue:
            # add the id of main crew to the list
            genre_list = []
            for j in range(len(str_keywords)):
                genre_list = list(str_keywords[j].values())
                if genre_list[0] not in popular_list:
                    genre_id = genre_list[0]
                    popular_list.append(genre_id)
    
    # create a col to indicate how "popular" the keywords are
    # for both datasets
    df['kw_pop'] = [0 for i in range(len(df['keywords']))]
    for i in range(len(df['keywords'])):
        keywords = df['keywords'][i]        # list of keywords
        str_keywords = json.loads(keywords) # convert to str
        if str_keywords != []:
            for j in range(len(str_keywords)):
                if list(str_keywords[j].values())[0] in popular_list:
                    df['kw_pop'][i] = df['kw_pop'][i] + 1

    df_validate['kw_pop'] = [0 for i in range(len(df_validate['keywords']))]
    for i in range(len(df_validate['keywords'])):
        keywords = df_validate['keywords'][i]   # list of keywords
        str_keywords = json.loads(keywords)     # convert to str
        if str_keywords != []:
            for j in range(len(str_keywords)):
                if list(str_keywords[j].values())[0] in popular_list:
                    df_validate['kw_pop'][i] = df_validate['kw_pop'][i] + 1
    

# get cast with year
def cast_year(df):
    df['cast1_year'] = [0 for i in range(len(df['cast']))]
    df['cast2_year'] = [0 for i in range(len(df['cast']))]
    get_cast(df)
    get_yearmonth(df)
    df['cast1_year'] = df['cast1'] / df['year']
    df['cast2_year'] = df['cast2'] / df['year']
    df = df.drop(columns=['cast1','cast2','year'])

# get producer
def get_producer(df):
    df['producer'] = [0 for i in range(len(df['crew']))]
    for i in range(len(df['crew'])):
        crew = df['crew'][i]        # list of crew
        str_crew = json.loads(crew) # convert to str
        for j in range(len(str_crew)):
            if list(str_crew[j].values())[4] == "Producer":
                # get the id of producer
                df['producer'][i] = list(str_crew[j].values())[3]

# get writer
def get_writer(df):
    df['writer'] = [0 for i in range(len(df['crew']))]
    for i in range(len(df['crew'])):
        crew = df['crew'][i]        # list of crew
        str_crew = json.loads(crew) # convert to str
        for j in range(len(str_crew)):
            if list(str_crew[j].values())[4] == "Writer":
                # get the id of writer
                df['writer'][i] = list(str_crew[j].values())[3]

# get editor
def get_editor(df):
    df['editor'] = [0 for i in range(len(df['crew']))]
    for i in range(len(df['crew'])):
        crew = df['crew'][i]        # list of crew
        str_crew = json.loads(crew) # convert to str
        for j in range(len(str_crew)):
            if list(str_crew[j].values())[4] == "Editor":
                # get the id of editor
                df['editor'][i] = list(str_crew[j].values())[3]


train_data['original_language'] = LabelEncoder().fit_transform(train_data['original_language'])
validate_data['original_language'] = LabelEncoder().fit_transform(validate_data['original_language'])



##############################
#           PART 1           #
##############################

#get_cast(train_data)
#get_cast(validate_data)
get_yearmonth(train_data)
get_yearmonth(validate_data)
get_budgetyear(train_data)
get_budgetyear(validate_data)
#get_genres(train_data)
#get_genres(validate_data)
get_director(train_data)
get_director(validate_data)
#get_num_companies(train_data)
#get_num_companies(validate_data)
get_large_com(train_data)
get_large_com(validate_data)
#get_two_companies(train_data)
#get_two_companies(validate_data)
popular_act(train_data, validate_data)
popular_crew(train_data, validate_data)
popular_genres(train_data, validate_data)
popular_keywords(train_data, validate_data)
get_num_sl(train_data)
get_num_sl(validate_data)
get_hp(train_data)
get_hp(validate_data)
get_producer(train_data)
get_producer(validate_data)
get_editor(train_data)
get_editor(validate_data)


# drop unnecessary columns
drop_cols = ['homepage', 'tagline', 'production_countries', 'spoken_languages', 'status', 'cast', 'crew', 'genres', 'production_companies',
        'original_title', 'overview', 'release_date', 'keywords']

train_data = train_data.drop(columns=['original_language'])
validate_data = validate_data.drop(columns=['original_language'])

train_data1 = train_data.drop(columns=drop_cols)
validate_data1 = validate_data.drop(columns=drop_cols)

train_data1 = train_data1.drop(columns=['rating','movie_id'])
validate_data1 = validate_data1.drop(columns=['rating','movie_id'])

y_train = train_data1['revenue'].values
x_train = train_data1.drop(columns=['revenue'])

y_test = validate_data1['revenue'].values
x_test = validate_data1.drop(columns=['revenue'])

print(train_data1.head())

# set up the gbrt model
model_gbrt = GradientBoostingRegressor()
model_gbrt.fit(x_train, y_train)
prediction = model_gbrt.predict(x_test)

'''
# set up the linear model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)

'''
'''
# set up the logic model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
'''

# print the result
MSE = round(mean_squared_error(y_test, prediction), 2)
print("MSE: ", MSE)
correlation = pearsonr(y_test, prediction)[0]
print("Pearson", correlation)


# summary of the evaluation metrics (MSE, correlation) for the model trained
summary1 = pd.DataFrame()
summary1['zid'] = ['z5190769']
summary1['MSE'] = [MSE]
summary1['correlation'] = [round(correlation,2)]
summary1.to_csv('z5190769.PART1.summary.csv', index=False)

# store the predicted revenues for all of the movies in the evaluation dataset
output1 = pd.DataFrame()
output1['movie_id'] = validate_data['movie_id']
output1['predicted_revenue'] = prediction
output1 = output1.sort_values('movie_id')
output1.to_csv('z5190769.PART1.output.csv', index=False)


##############################
#           PART 2           #
##############################

cast_year(train_data)
cast_year(validate_data)
popular_act(train_data, validate_data)
popular_crew(train_data, validate_data)
popular_keywords(train_data, validate_data)
get_num_sl(train_data)
get_num_sl(validate_data)
get_len_tag(train_data)
get_len_tag(validate_data)

drop_cols = ['homepage', 'tagline', 'production_countries', 'spoken_languages', 'status', 'cast', 'crew', 'genres', 'production_companies',
        'original_title', 'overview', 'release_date', 'keywords']
drop_cols = drop_cols + ['budget','year','num_co','budget_year','cast1','cast2','director','editor']

train_data2 = train_data.drop(columns=drop_cols)
validate_data2 = validate_data.drop(columns=drop_cols)
train_data2 = train_data2.drop(columns=['revenue','movie_id'])
validate_data2 = validate_data2.drop(columns=['revenue','movie_id'])

y_train2 = train_data2['rating'].values
x_train2 = train_data2.drop(columns=['rating'])

y_test2 = validate_data2['rating'].values
x_test2 = validate_data2.drop(columns=['rating'])

# train
knn = KNeighborsClassifier()
knn.fit(x_train2, y_train2)
prediction2 = knn.predict(x_test2)
'''
dt = DecisionTreeClassifier()
dt.fit(x_train2, y_train2)
prediction2 = dt.predict(x_test2)
'''
# predict
precision = precision_score(y_test2, prediction2, average='macro')
recall = recall_score(y_test2, prediction2, average='macro')
accuracy = accuracy_score(y_test2, prediction2)
print("precision ", precision)
print("recall ", recall)
print("accuracy ", accuracy)

# summary of the evaluation metrics (average_precision, average_recall, accuracy - the unweighted mean ) 
summary2 = pd.DataFrame()
summary2['zid'] = ['z5190769']
summary2['average_precision'] = [round(precision,2)]
summary2['average_recall'] = [round(recall,2)]
summary2['accuracy'] = [round(accuracy,2)]
summary2.to_csv('z5190769.PART2.summary.csv', index=False)

# store the predicted revenues for all of the movies in the evaluation dataset
output2 = pd.DataFrame()
output2['movie_id'] = validate_data['movie_id']
output2['predicted_rating'] = prediction2
output2 = output2.sort_values('movie_id')
output2.to_csv('z5190769.PART2.output.csv', index=False)


# check correlation by plot (PART 1)
Correlation = pd.DataFrame(train_data1)
colormap = plt.cm.rainbow
plt.figure(figsize=(12,7))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(Correlation.astype(float).corr(),fmt=".2f",linewidths=0.05,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()


# check correlation by plot (PART 2)
Correlation = pd.DataFrame(train_data2)
colormap = plt.cm.rainbow
plt.figure(figsize=(12,7))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(Correlation.astype(float).corr(),fmt=".2f",linewidths=0.05,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()

#mask=Correlation.astype(float).corr()<0.01
