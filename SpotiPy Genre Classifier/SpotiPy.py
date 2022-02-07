#!/usr/bin/env python
# coding: utf-8

# # Final Project 
# ### Madeline, Ethan, Alex

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler, LabelEncoder, label_binarize, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# ## Gathering the Data 
# - To create dataset, pull data using the Spotipy Library to access the Spotify API.
# - 1,000 songs from each of the following genres were pulled: **Pop, Rock, Country, EDM, Rap, and Classical**
# - For each song, features such as "danceability", "energy", "loudness", "speechiness", and "acousticness" are recorded into a dataframe.
# - Dataframe is saved to csv so that it can be loaded directly.

# In[48]:


# Spotify Client ID needed for API
#Omitted for security purposes.


# In[ ]:


# Create dictionary of features that will be pulled from each song
features = {
            "genre" : [],
            "artist_name": [],
            "track_name": [],
            "track_id": [],
            "popularity": [],
            "danceability": [],
            "energy": [],
            "key": [],
            'loudness': [],
            'mode': [],
            'speechiness': [],
            'acousticness': [],
            'instrumentalness': [],
            'liveness': [],
            'valence': [],
            'tempo': [],
            'duration_ms': []
            }

genres = ['pop', 'rock', 'country', 'EDM', 'rap', 'classical']

# Search Spotify by genre and add feature values of each song to the features dictionary
for genre in genres:
    genreS = "genre:" + genre
    for i in range (0,1000,50):
        results = sp.search(q=genreS, type='track', limit=50,offset=i)

        for track in results['tracks']['items']:
            features['artist_name'].append(track['artists'][0]['name'])
            features['track_name'].append(track['name'])
            features['track_id'].append(track['id'])
            features['popularity'].append(track['popularity'])
            features['genre'].append(genre)
            audio_features = sp.audio_features(track['id'])
            features['danceability'].append(audio_features[0]["danceability"])
            features['energy'].append(audio_features[0]["energy"])
            features['key'].append(audio_features[0]["key"])
            features['loudness'].append(audio_features[0]["loudness"])
            features['mode'].append(audio_features[0]["mode"])
            features['speechiness'].append(audio_features[0]["speechiness"])
            features['acousticness'].append(audio_features[0]["acousticness"])
            features['instrumentalness'].append(audio_features[0]["instrumentalness"])
            features['liveness'].append(audio_features[0]["liveness"])
            features['valence'].append(audio_features[0]["valence"])
            features['tempo'].append(audio_features[0]["tempo"])
            features['duration_ms'].append(audio_features[0]["duration_ms"])
            
# Create a dataframe from the dictionary values           
class_df = pd.DataFrame.from_dict(features)


# In[ ]:


class_df.drop_duplicates(subset = ["track_id"], inplace=True)


# In[104]:


# Save as csv to avoid pulling the data again
#class_df.to_csv("class_data.csv", index=False)


# ## Exploratory Data Analysis
# 
# - Observe patterns in potential predictor variables by using `df.corr()` to look at correlation between every numeric feature.

# - **Acousticness** and **Instrumentalness** are highly correlated.
# 
# - **Energy**, **loudness**, and **popularity**, and to a lesser extent **danceability**, are all positively correlated. 
#     - These four features are very strongly *negatively* correlated to the two features mentioned above. 

# In[2]:


data = pd.read_csv("class_data.csv")
non_numerics = ["genre", "artist_name", "track_name", "track_id"]

sns.heatmap(data.drop(columns=non_numerics).corr(), square=True)


# In[3]:


data.drop(columns=non_numerics).corr()


# - Observe how these features are represented in each genre by first converting them to z-scores, then taking the mean from each genre.

# In[4]:


scaler = StandardScaler()
cols = data.drop(columns=["artist_name", "track_name", "track_id", "genre"]).columns
data_s = scaler.fit_transform(data[cols])

data_s = pd.DataFrame(data_s, columns=cols)
data_s["genre"] = data["genre"]
data_s


# In[5]:


genres = data_s.groupby(by="genre").mean()
genres


# - Establish one of these genres, **Rock**, as a base level. 
# - Identify and interpret the difference between Rock and the other genres.

# In[6]:


for genre in genres.index:
    genres.loc[genre] = genres.loc[genre] - genres.iloc[5]

genres


# * *Classical music* has the most differences between the rest of the genres by far-- the magnitude of its bars tells that much. 
# * In contrast, the small bars of *country* mean it will likely be difficult to differentiate rock and country. It looks like duration or acousticness will be the most useful in telling them apart. 
# * *Rap* is set apart from every other genre by its high speechiness and danceability. 
# * *Pop's* defining traits appear to be danceability and speechiness as well, though not to the same extent as rap. It also has the highest popularity of the genres.
# * *EDM* has the highest instrumentalness after classical music, the highest loudness and energy, and much lower popularity than the other genres. 
# * Other differences are not quite as obvious: we must keep these observations in mind while training the classifier.

# In[7]:


plt.figure(figsize=(16, 10))

sns.barplot(data=genres.reset_index().melt("genre"), x="genre", y="value", hue="variable", palette="muted")
plt.ylabel("Difference in z-score from Rock", size=16)
plt.xlabel("Genre", size=16)
plt.title("Z-scores of Rock vs other Genres", size=20)


# In the process of our model creation, we found that pop was regularly confused for other genres, to the point of unusability. We decided to remove pop songs from our genre classifier for that reason. Why, specifically, is that the case? Let's look at a comparison of its Z-scores against the other genres, like we did for rock.

# In[11]:


genres = data_s.groupby(by="genre").mean()

pop = genres.loc["pop"]

genres = genres - pop


# In[9]:


plt.figure(figsize=(16, 10))

sns.barplot(data=genres.reset_index().melt("genre"), x="genre", y="value", hue="variable", palette="muted")
plt.ylabel("Difference in z-score from Pop", size=16)
plt.xlabel("Genre", size=16)
plt.title("Z-scores of Pop vs other Genres", size=20)


# Let's break down the differences between pop's characteristics and those of the other genres, excluding classical from most of this discussion due to its massive differences.
# * Pop has the highest popularity, but rap is only slightly lower.
# * Pop's danceability is middle-of-the-road, higher than some and lower than others.
# * A similar story with energy, though pop's energy may be on the lower end.
# * Key is mostly irrelevant in our opinion. The differences are very small.
# * Pop's loudness also does not differ very much from most of the genres.
# * Except for country, the mode (major/minor) of pop also does not differ much from the others.
# * Pop's speechiness is rather high, except when compared to rap.
# * Acousticness is slightly higher than the others, but not significantly.
# * Instrumentalness varies very little by genre (except classical, of course).
# * Liveness appears to be on the lower end, but not by much.
# * Valence also seems to be middle of the road.
# * Tempo is slightly lower than the others.
# * With the exception of rock, duration is on the higher end.
# 
# Almost all of pop's characteristics land in the middle of the other genres, or only set themselves apart by a small margin. This makes it especially hard for a classifier such as ours to make distinctions between pop and other genres. Rock and country are a similar story, but they have some identifying traits which differ intensely from the others, such as tempo. 

# -------------------------

# ## Creating a Genre Classifier
# - The goal of the classifier is to **predict the genre of a given song**.
# - The **features** used to predict genre include:
#     - Danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, and duration of a song.
# - All features are scaled using `MinMaxScaler()` to be used in the classification models
# - Y labels are encoded to numerical format using `LabelEncoder()`

# In[105]:


# Pop is removed for classification as it contains overlap with other genres
data = data[data["genre"] != "pop"]
data['genre'].value_counts()


# In[106]:


features = ['danceability', 'energy', 'loudness', 'speechiness', 
           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', "duration_ms"]

genres = ['rock','country', 'EDM','rap', 'classical']

le = LabelEncoder()

# Transform y labels into numerical categories using LabelEncoder()
X = data[features]
y = le.fit_transform(data['genre'])
le.transform(['rock','country', 'EDM','rap', 'classical'])

genres_re = ["EDM","classical","country","rap","rock"]


# In[108]:


# Create train, validation, and test sets in a 70%, 20%, 10% split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size = .9, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=.77, random_state=1)


# In[109]:


# Scale all x dataframes to be used with classifier
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns = X.columns)
X_val = pd.DataFrame(scaler.transform(X_val), columns = X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)


# ### Evaluation Environment
# - The evaluation environment takes as input:
#     - Model Name
#     - Set being evaluated (training, validation, or test)
#     - The ground-truth y-labels for the set.
#     - The predicted labels for the set.
# - The evaluation environment returns:
#     - Accuracy
#     - F1 Score
#     - Confusion Matrix

# In[113]:


# Evaluation environment
# Return accuracy, F1 score, and Confusion Matrix for given model
def evalEnv(model, set_type, y_label, y_pred, cm = False):
    
    print("\nModel: " + model)
    print("\n" + set_type + " Accuracy = " + str(accuracy_score(y_label, y_pred)))
    print(set_type + " F1 Score = " + str(f1_score(y_label, y_pred, average= "weighted")))
    
    #Plot Confusion Matrix
    if cm:
        cm = confusion_matrix(y_label, y_pred)
        fig, ax = plt.subplots(figsize=(8,8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(model + " Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(genres))
        plt.xticks(tick_marks, genres_re, rotation=45)
        plt.yticks(tick_marks, genres_re)
        # Plot numbers onto grid
        for (j,i),label in np.ndenumerate(cm):
            ax.text(i,j,label,ha='center',va='center')


# In[114]:


# Runs model on training set and returns predictions for the set given by the last parameter
def runModel(model, X_tr, y_tr, X_v):
    model.fit(X_tr,y_tr)
    y_pred = model.predict(X_v)
    return y_pred


# ### Testing Models
# - Using the evaluation environment, we can quickly test a variety of classifier models to see which performs best.
# - Two baselines are created to compare results with.
#     - **Baseline 1**: Randomly guess one of the five genres
#     - **Baseline 2**: Guess the most common genre in the dataset (EDM)

# In[97]:


models = {  "Nearest Neighbors": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Linear SVM": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Neural Network": MLPClassifier(max_iter=1000),
            "Naive Bayes": GaussianNB(),}

for modelName in models:
    preds = runModel(models[modelName], X_train, y_train, X_val)
    evalEnv(modelName, "Validation Set", y_val, preds)

baseline_rand = np.random.randint(0,5,len(y_val))
baseline_mode = np.zeros(len(y_val))

evalEnv("Random Guess Baseline", "", y_val, baseline_rand)
evalEnv("Guess Mode Baseline", "", y_val, baseline_mode)


# ### Preliminary Results
# - All classifiers greatly outperform the baseline results.
# - The top classifiers are Random Forest, Neural Network, and Liner SVM.
# 
# 
# **Random Forest Classifier**
# 
# - The Random Forest Classifier performs the best of any classifier when given default parameters, with an accuracy in the 75-76 percent range over multiple runs.
# - The confusion matrix gives us information about which genres the classifier performed well and poorly on.
#     - Nearly all classical songs were classified correctly
#     - Rock and country were the most commonly confused genres
#     - EDM was often classified as country, rap, or rock

# In[100]:


rf = RandomForestClassifier()

rf_pred_val = runModel(rf, X_train, y_train, X_val)

evalEnv("Random Forest Classifier", "Validation Set", y_val, rf_pred_val, cm = True)


# ### Hyperparameter Tuning
# - Since the Random Forest Classifier performed the best of any classifier with default hyperparameters, we can now tune hyperparameters to see if we can improve performance.
# - There are 18 hyperparameters for the classifier, so we can take a subset of the most important ones and test various values.
# - This is done using `RandomizedSearchCV`
# - From the results, we see that the model performs better with a greater number of n_estimators and max_depth. The tradeoff is that the model takes longer to train with more estimators.

# In[25]:


parameters = {  'n_estimators': [100, 500, 1000],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [1, 2, 4],
                'min_samples_split': [2, 5, 10] }
grid1 = RandomizedSearchCV(rf, parameters, cv=3, n_iter=20, random_state=42)
grid1.fit(X_train, y_train)
results1 = grid1.cv_results_
df1 = pd.DataFrame(results)
bestParams1 = df[["param_n_estimators", "param_max_depth", "param_max_features", "param_min_samples_leaf", "param_min_samples_split", 'mean_test_score']]
bestParams1


# ### Grid Search CV
# - After finding the best values for the n_estimators and max_depth parameters, we use `GridSearchCV` to test values for the other parameters.
# - This has an advantage over `RandomizedSearchCV` as it tests all possible values so that we can see exactly which parameters perform best.
# - Note: Random Forest Classifier performs slightly differently on each run but the GridSearch takes the mean score over all cross validation sets, so it still should return the model with the best parameters.

# In[100]:


parameters = {  'max_features' : ['auto', 'sqrt', 1],
                'min_samples_split' : [2, 5, 10, 15],
                'min_samples_leaf' : [1, 2, 5, 10] }
base = RandomForestClassifier(n_estimators = 1000, max_depth = 90)
grid = GridSearchCV(base, parameters, cv=3)
grid.fit(X_train, y_train)
results = grid.cv_results_
df = pd.DataFrame(results)
bestParams = df[["param_max_features", "param_min_samples_leaf", "param_min_samples_split", 'mean_test_score']]
bestParams


# In[30]:


grid.best_params_


# ### Final Tuned Model
# - Using the best parameters from grid search, we create a new tuned Random Forest Classifer.
# - This performs slightly better on the training set compared to the default model, receiving an accuracy of **77.1%**.
# - The model scores slightly lower on the test set compared to the training set with **72.7%** accuracy.
# - High accuracy on the training set indicates that the model may be overfitting the data. However, accuracy on the validation set still increased even as the model was overfitting.

# In[128]:


tunedRF = RandomForestClassifier(n_estimators = 1000, max_depth = 90, max_features = 1, min_samples_split = 10, min_samples_leaf=2)

trf_pred_test = runModel(tunedRF, X_train, y_train, X_test)
evalEnv("Random Forest", "Test Set", y_test, trf_pred_test, cm = False)

trf_pred_train = runModel(tunedRF, X_train, y_train, X_train)
evalEnv("Random Forest", "Training Set", y_train, trf_pred_train, cm = False)

trf_pred_val = runModel(tunedRF, X_train, y_train, X_val)
evalEnv("Tuned Random Forest CLF", "Validation Set", y_val, trf_pred_val, cm = True)

