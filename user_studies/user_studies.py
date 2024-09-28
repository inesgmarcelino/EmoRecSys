# ----- IMPORTS ----- #
from sklearn.metrics.pairwise import euclidean_distances
from ipywidgets import HBox, VBox, Image as WidgetImage
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import SpectralClustering
from surprise import Reader, Dataset, NMF
from sklearn.decomposition import PCA
from collections import defaultdict
from IPython.display import display
from scipy import stats
from PIL import Image
import pandas as pd
import numpy as np
import warnings
import random
import time
import math
import io

### 1. GETTING NEW DATA USER

# --------------------------------- #
# Call from the server to get this  #
#    new line to insert into the    #
#             dataset.              #
# --------------------------------- #
# (but for now, let's keep it like  #
#               this)               #
# --------------------------------- #

def processing_userstudies_newdata(participant_number):
    """
    Processes data related to a participant's survey, ratings, hobbies, diseases, and visual acuities. 
    It transforms the data to facilitate further analysis or modeling.

    Args:
    participant_number (int): The participant number used to locate their specific data files.

    Returns:
    new_survey (pandas.DataFrame): The transformed survey data including hobby, disease, and visual acuity features.
    new_ratings (pandas.DataFrame): The new ratings data for the participant.

    """
    main_path = f'./participants_data/participant_{str(participant_number)}'
    new_ratings = pd.read_csv(f"{main_path}/new_ratings{str(participant_number)}.csv")

    new_survey = pd.read_csv(f"{main_path}/new_surveys{str(participant_number)}.csv", encoding="utf-8")
    new_survey.columns = ['id_survey', 'age', 'populational_aff', 'gender', 'education', 'city', 'country_residence', 'date_survey', 'consented', 'hobby_other']

    # -- hobbies
    new_hobbies = pd.read_csv(f"{main_path}/new_survey_hobbies{str(participant_number)}.csv")
    hobbies_grouped = new_hobbies.groupby('id_survey')['id_hobby'].apply(list).reset_index()
    new_survey = pd.merge(new_survey, hobbies_grouped, on='id_survey', how='left')

    for hobby in list(set(new_hobbies['id_hobby'].tolist())):
        new_survey["hobby_"+str(hobby)] = new_survey['id_hobby'].apply(lambda x: 1 if isinstance(x, list) and hobby in x else 0)

    new_survey.drop(columns=['id_hobby'], inplace=True)

    # -- diseases
    new_diseases = pd.read_csv(f"{main_path}/new_survey_diseases{str(participant_number)}.csv")
    diseases_grouped = new_diseases.groupby('id_survey')['id_disease'].apply(list).reset_index()
    new_survey = pd.merge(new_survey, diseases_grouped, on='id_survey', how='left')

    for disease in list(set(new_diseases['id_disease'].tolist())):
        new_survey["disease_"+str(disease)] = new_survey['id_disease'].apply(lambda x: 1 if isinstance(x, list) and disease in x else 0)
    
    new_survey.drop(columns=['id_disease'], inplace=True)

    # -- visual acuities
    new_visual_acuities = pd.read_csv(f"{main_path}/new_survey_visual_acuities{str(participant_number)}.csv")
    visual_acuities_grouped = new_visual_acuities.groupby('id_survey')['id_visual_acuities'].apply(list).reset_index()
    new_survey = pd.merge(new_survey, visual_acuities_grouped, on='id_survey', how='left')
    
    for visual_acuity in list(set(new_visual_acuities['id_visual_acuities'].tolist())):
        new_survey["visual_acuity"+str(visual_acuity)] = new_survey['id_visual_acuities'].apply(lambda x: 1 if isinstance(x, list) and visual_acuity in x else 0)
    
    new_survey.drop(columns=['id_visual_acuities'], inplace=True)

    return new_survey, new_ratings

dataset_demo = pd.read_csv("../data/csvs/demographic.csv", encoding="utf-8")
dataset_cf = pd.read_csv("../data/csvs/ratings.csv")

new_demo, new_cf = processing_userstudies_newdata(participant_number=0)
new_demo = new_demo.reindex(columns=dataset_demo.columns, fill_value=0)

new_dataset_demo = pd.concat([dataset_demo, new_demo], ignore_index=True)
new_dataset_cf = pd.concat([dataset_cf, new_cf], ignore_index=True)

dataset_photos = pd.read_csv("../data/csvs/photos.csv")
dataset_photos.set_index('id', inplace=True)


############################################################################
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
############################################################################


### 2. PREPROCESSING

# preprocessing for Demographic dataset
user_profiles = new_dataset_demo.iloc[:, -27:].copy()

# one-hot encoder
for col in ['hobby_other', 'country_residence', 'city', 'education', 'gender', 'populational_aff', 'age']:
    dummies = pd.get_dummies(new_dataset_demo[col], prefix=col)
    user_profiles = pd.concat([dummies, user_profiles], axis=1)

user_profiles.dropna(inplace=True)
user_profiles = user_profiles.applymap(lambda x: 1 if x is True else 0 if x is False else x)

# applying PCA
pca = PCA(n_components=50)
pca.fit(user_profiles)
tve = 0 # total variance explained
for i, ve in enumerate(pca.explained_variance_ratio_):
    tve += ve
    print("PC%d - Variance explained: %7.4f - Total Variance: %7.4f" % (i+1, ve, tve))

# keep 32 principal components, since we get a total explained variance of 90.13%.
X_pca = pca.transform(user_profiles)[:, :32]

# see dataframe
user_profiles_train = pd.DataFrame(data=X_pca, columns=['PCA'+str(i) for i in range(1, X_pca.shape[1]+1)], index=new_dataset_demo['id_survey'])

## as we just removed outliers, we will check that we are only using valid data points
new_dataset_cf = new_dataset_cf[new_dataset_cf['id_survey'].isin(user_profiles_train.index.tolist())]

## now we will be creating a ratings_matrix, in this case using the `like_bool` as the rating
ratings_matrix_like = new_dataset_cf.pivot_table(index='id_survey', columns='id_photo', values='like_bool').reset_index(drop=True)


############################################################################
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
############################################################################


### 3. RECOMMENDATION SYSTEMS

# --------------------------------- #
#   3.1. Content-based Filtering    #
# --------------------------------- #

# --------------------------------- #
#   3.2. Collaborative Filtering    #
# --------------------------------- #
def recommend_items_from_cf(user_test, threshold=0):
    """
    Recommends items based on collaborative filtering predictions. The functions sorts predictions by estimated rating, and
    filters them based on a given threshold.

    Args:
    predictions (list of tuples): A list of predictions tuples, where each tuple contains a user ID, item ID, actual rating,
        estimated rating, and an additional information.
    threshold (float): Minimum estimated rating to consider an item for recommendation. Default is 0.

    Returns:
    recommended_items (list of tuples): A list of tuples where each tuple contains an item ID and its estimated rating. The 
        list is sorted in descending order of the estimated ratings.

    """
    cf_params = {'n_epochs': 20, 'n_factors': 100, 'reg_pu': 0.001, 'reg_qi': 0.01}

    reader = Reader(rating_scale=(0,1)) # like_bool
    new_dataset = Dataset.load_from_df(new_dataset_cf[['id_survey', 'id_photo', 'like_bool']], reader).build_full_trainset()

    model_cf = NMF(**cf_params) # to fix the model
    model_cf.fit(new_dataset)

    to_predict = get_to_predict(user_test, new_dataset)
    predictions_cf = [model_cf.predict(user_test, iid) for iid in to_predict]

    preds = list()
    for _, iid, _, est, _ in predictions_cf:
        preds.append((iid, est))

    preds.sort(key=lambda x: x[1], reverse=True)
    recommended_items = [(iid, est) for (iid, est) in preds if est >= threshold]

    return recommended_items[:5]

def get_to_predict(user, surprise_dataset):
    " Generates a list of items for prediction with randomly items user has not seen"
    all_items = {surprise_dataset.to_raw_iid(iid) for (_, iid, _) in surprise_dataset.all_ratings()}
    items_seen_by_user = {surprise_dataset.to_raw_iid(iid) for (uid, iid, _) in surprise_dataset.all_ratings() if surprise_dataset.to_raw_uid(uid) == user}

    # find items user has not seen, and randomly sample 50 unseen items
    items_unseen_by_user = list(all_items - items_seen_by_user)
    unseen_sample = random.sample(items_unseen_by_user, 50)

    return unseen_sample

############################################################################
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
# -------------------------------------------------------------------------#
############################################################################


## 4. MORE FUNCTIONS

def display_image(show_list, type=None):
    """
    Displays a list of images in a grid format. It reads images from the specified file paths and displays them using widgets.

    Args:
    show_list (list of str): A list of item IDs representing the images to be displayed.
    n_show (int): The number of images to display per row. Default is 3.
    """
    for item in show_list:
        image = dataset_photos.loc[item]['file_name']
        ext = dataset_photos.loc[item]['ext']
        image_path = f'../data/photos/{image}.{ext}'

        image_show = Image.open(image_path)

        image_show.show()

        if type:
            time.sleep(90)

def user_studies_recommendation(user_test, test_type):
    # liked_items = {row['id_photo'] for _, row in new_dataset_cf[new_dataset_cf['id_survey'] == user_test].iterrows() if row['like_bool'] == 1}
    
    # print('PREVIOUSLY LIKED IMAGES:')
    # display_image(liked_items)
    # --- --- --- ---

    match test_type:
        case 1: # content-based
            pass
        case 2: # collaborative
            recommendations = {iid for (iid, _) in recommend_items_from_cf(user_test)}

        case 3: # random
            all_items = list(new_dataset_cf['id_photo'].unique())
            recommendations = random.sample(all_items, 50)

        case _:
            print("There's something wrong.") 
    
    # --- --- --- ---
    print('\nHERE IS YOUR RECOMMENDATIONS:')
    display_image(recommendations, 1)

    pass

user_studies_recommendation(148, 3)