# ----- IMPORTS ----- #
from cbf import recommend_items_from_cbf
from adding_new_data import get_newdata
from cf import recommend_items_from_cf
from surprise import Reader, Dataset
import pandas as pd
import subprocess
import random
import shutil
import sys
import os

dataset_photos = pd.read_csv("../data/csvs/photos.csv")
dataset_photos.set_index('id', inplace=True)

def recommendations_path(recommendations, user_test, test_type):
    dest_path = os.path.join(os.getcwd(), f'recommendations_{user_test}/system_{test_type}/')
    
    for n, item in enumerate(recommendations):
        image = dataset_photos.loc[item]['file_name']
        ext = dataset_photos.loc[item]['ext']
        image_path = f'../data/photos/{image}.{ext}'

        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(image_path, os.path.join(dest_path, f'top{n+1}.{ext}'))
    return dest_path

def user_studies_recommendation(dataset, user_test, test_type):
    print('Please wait for a few seconds...')
    # liked_items = {row['id_photo'] for _, row in new_dataset_cf[new_dataset_cf['id_survey'] == user_test].iterrows() if row['like_bool'] == 1}
    
    # print('PREVIOUSLY LIKED IMAGES:')
    # display_image(liked_items)
    # --- --- --- ---

    match test_type:
        case 1: # content-based
            similarity_df = pd.read_csv("../data/csvs/sim_df_resnet101v2.csv")
            recommendations = recommend_items_from_cbf(user_test, dataset, similarity_df)
            # recommend_items_from_cbf
            pass
        case 2: # collaborative
            recommendations = {iid for (iid, _) in recommend_items_from_cf(user_test, dataset)}

        case 3: # random
            all_items = list({dataset.to_raw_iid(iid) for (_, iid, _) in dataset.all_ratings()})
            recommendations = random.sample(all_items, 5)

        case _:
            print("There's something wrong.") 
    
    # --- --- --- ---
    print('\nYour recommendations are in!')
    dest_path = recommendations_path(recommendations, user_test, test_type)
    subprocess.Popen(['explorer', os.path.realpath(dest_path)])


if __name__ == "__main__":
    user_test = int(sys.argv[1])
    rec_type = int(sys.argv[2])

    dataset = get_newdata()
    reader = Reader(rating_scale=(0,1)) # like_bool
    new_dataset = Dataset.load_from_df(dataset[['id_survey', 'id_photo', 'like_bool']], reader).build_full_trainset()
    print(f'Hi user {user_test}!')
    user_studies_recommendation(new_dataset, user_test, rec_type)