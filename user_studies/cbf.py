import pandas as pd
import numpy as np
import random

# ----------------------------- #
#    Content-based Filtering    #
# ----------------------------- #

def get_to_predict(user, surprise_dataset):
    """
    Identifies a sample of uneseen items for a specific user from the Surprise dataset.

    Args:
        user (int): The ID of the user for whom unseen items are to be recommended.
        surprise_dataset (Surprise dataset): A dataset containing the user-item interactions.

    Returns:
        unseen_sample (list): A list of randomly sampled unseen items IDs for the specified user. The sample size is 50.
    """
    all_items = {surprise_dataset.to_raw_iid(iid) for (_, iid, _) in surprise_dataset.all_ratings()}
    items_seen_by_user = {surprise_dataset.to_raw_iid(iid) for (uid, iid, _) in surprise_dataset.all_ratings() if surprise_dataset.to_raw_uid(uid) == user}

    # find items user has not seen, and randomly sample 50 unseen items
    items_unseen_by_user = list(all_items - items_seen_by_user)
    # unseen_sample = random.sample(items_unseen_by_user, 50)

    return items_seen_by_user, items_unseen_by_user

def item_similarity(item_to_recommend, user_profile_items, similarity_df, n):
    """Calculates the similarity between an item to be recommended and the user profile items,
    returns the most similar items.

    Args:
        item_to_recommend (int): an item from the test set that we want to recommend.
        user_profile_items (list): list of item ids present in the user's training set
        n (int): number of items that we want to see how similar are to the user_profile_items.

    Returns:
        list: a list of item ids sorted by similarity score from higher to a lower value select the top n.
    """
    similarity_list = []

    for item_train in user_profile_items:
        # Verify if there is a row where the item to bem recommended is the id_target and the profile item is id_comparison
        row_1 = similarity_df[
            (similarity_df['id_target'] == item_to_recommend) & 
            (similarity_df['id_comparison'] == item_train)
        ]
        
        # Verify if there is a row where the item to bem recommended is the id_comaparison and the profile item is id_target
        row_2 = similarity_df[
            (similarity_df['id_target'] == item_train) & 
            (similarity_df['id_comparison'] == item_to_recommend)
        ]
        
        # Similarity in the first verification
        if not row_1.empty:
            similarity_score = row_1.iloc[0]['similarity_score']
            similarity_list.append((item_train, similarity_score))
        
        # Similarity in the second verification
        elif not row_2.empty:
            similarity_score = row_2.iloc[0]['similarity_score']
            similarity_list.append((item_train, similarity_score))

    similarity_list.sort(key=lambda x: x[1], reverse=True)
    item_ids = [item for item, _ in similarity_list]
    # print(similarity_list)
    # print(item_ids)

    return item_ids[:n]

def recommend_items_from_cbf(user, surprise_dataset, similarity_df, n=5):
    """
    This function compares the items that the user already saw and classified with the items that the user saw to generate personalized recommendations.

    Args:
        trainset (list): The train contains a tuple where we have the user id, item id interaction and rating.
        testset (list): The test contains a tuple where we have the user id, item id interaction and rating.
        n (int, optional): The number of similar items to consider for each user in the testset. Defaults to 5.

    Returns:
        dict: a dictionary where the keys are user ids and the values are lists of the top 10 recommended item IDs.
    """

    items_seen_by_user, unseen_sample = get_to_predict(user, surprise_dataset)

    recommendations = {}
        
    interactions_list = [(surprise_dataset.to_raw_uid(uid), surprise_dataset.to_raw_iid(iid), rating)
                            for (uid, iid, rating) in surprise_dataset.all_ratings()]
        
    interactions_df = pd.DataFrame(interactions_list, columns=['uid', 'iid', 'rating'])
    user_interactions_df = interactions_df[interactions_df['uid'] == user]
        
    # print(user_interactions_df)
    
    items_to_recommend = []
    for item_test in unseen_sample:
        item_train_sim = item_similarity(item_test, items_seen_by_user, similarity_df, n)
        item_train_sim_df = user_interactions_df[user_interactions_df['iid'].isin(item_train_sim)]
        ratings = item_train_sim_df['rating'].tolist()
        mean_value = np.mean(ratings)
        items_to_recommend.append((item_test, mean_value))

    items_to_recommend_ids = [item for item, _ in items_to_recommend]
    if len(items_to_recommend_ids) < 10:
        available_items = list(unseen_sample - set(items_seen_by_user) - set(items_to_recommend_ids))
        num_to_add = 10 - len(items_to_recommend_ids)
        random_items = random.sample(available_items, num_to_add)

        for item_random in random_items:
            item_random_sim = item_similarity(item_random, items_seen_by_user, similarity_df, n)
            item_random_sim_df = user_interactions_df[user_interactions_df['iid'].isin(item_random_sim)]
            ratings_random = item_random_sim_df['rating'].tolist()
            mean_value_random = np.mean(ratings_random) 
            items_to_recommend.append((item_random, mean_value_random))

    # Sort the items by their evaluation and return the ids
    items_to_recommend.sort(key=lambda x: x[1], reverse=True)
    # print(items_to_recommend)
    items_to_recommend_ids = [item for item, _ in items_to_recommend]

    recommendations = items_to_recommend_ids[:5] 

    return recommendations
