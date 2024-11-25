# ----- IMPORTS ----- #
from surprise import Reader, Dataset, NMF
import pandas as pd
import random

# ----------------------------- #
#    Collaborative Filtering    #
# ----------------------------- #

def get_to_predict(user, dataset):
    """
    Identifies a sample of uneseen items for a specific user from the Surprise dataset.

    Args:
        user (int): The ID of the user for whom unseen items are to be recommended.
        surprise_dataset (Surprise dataset): A dataset containing the user-item interactions.

    Returns:
        unseen_sample (list): A list of randomly sampled unseen items IDs for the specified user. The sample size is 50.
    """
    all_items = {dataset.to_raw_iid(iid) for (_, iid, _) in dataset.all_ratings()}
    items_seen_by_user = {dataset.to_raw_iid(iid) for (uid, iid, _) in dataset.all_ratings() if dataset.to_raw_uid(uid) == user}

    # find items user has not seen, and randomly sample 50 unseen items
    items_unseen_by_user = list(all_items - items_seen_by_user)
    unseen_sample = random.sample(items_unseen_by_user, 50)

    return unseen_sample

def recommend_items_from_cf(user_test, dataset, threshold=0):
    """
    Recommends items to a specified user based on collaborative filtering predictions. 
    The function uses a X model to predict ratings for unseen items.
    Then it filters the recommendations based on a specified threshold and sorts them by estimated rating.

    Args:
        user_test (int): The ID of the user for whom to generate recommendations.
        dataset (DataFrame): A DataFrame containing the user-item interactions.
        threshold (float, optional): The minimum estimated rating to consider an item for recommendation. Default is 0.

    Returns:
        recommended_items (list): A sorted list of tuples where each tuple contains an item ID and its estimated rating. 
            The list is limited to the top 5 recommendations.
    """
    cf_params = {'n_epochs': 20, 'n_factors': 100, 'reg_pu': 0.001, 'reg_qi': 0.01}

    model_cf = NMF(**cf_params) # to fix the model
    model_cf.fit(dataset)

    to_predict = get_to_predict(user_test, dataset)
    predictions_cf = [model_cf.predict(user_test, iid) for iid in to_predict]

    preds = list()
    for _, iid, _, est, _ in predictions_cf:
        preds.append((iid, est))

    preds.sort(key=lambda x: x[1], reverse=True)
    recommended_items = [(iid, est) for (iid, est) in preds if est >= threshold]

    return recommended_items[:5]