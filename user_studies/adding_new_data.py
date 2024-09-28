import pandas as pd

dataset_cf = pd.read_csv("../data/csvs/ratings.csv")

def get_newdata():
    return dataset_cf

# def processing_userstudies_newdata(participant_number):
#     """
#     Processes data related to a participant's survey, ratings, hobbies, diseases, and visual acuities. 
#     It transforms the data to facilitate further analysis or modeling.

#     Args:
#     participant_number (int): The participant number used to locate their specific data files.

#     Returns:
#     new_survey (pandas.DataFrame): The transformed survey data including hobby, disease, and visual acuity features.
#     new_ratings (pandas.DataFrame): The new ratings data for the participant.

#     """
#     main_path = f'./participants_data/participant_{str(participant_number)}'
#     new_ratings = pd.read_csv(f"{main_path}/new_ratings{str(participant_number)}.csv")

#     new_survey = pd.read_csv(f"{main_path}/new_surveys{str(participant_number)}.csv", encoding="utf-8")
#     new_survey.columns = ['id_survey', 'age', 'populational_aff', 'gender', 'education', 'city', 'country_residence', 'date_survey', 'consented', 'hobby_other']

#     # -- hobbies
#     new_hobbies = pd.read_csv(f"{main_path}/new_survey_hobbies{str(participant_number)}.csv")
#     hobbies_grouped = new_hobbies.groupby('id_survey')['id_hobby'].apply(list).reset_index()
#     new_survey = pd.merge(new_survey, hobbies_grouped, on='id_survey', how='left')

#     for hobby in list(set(new_hobbies['id_hobby'].tolist())):
#         new_survey["hobby_"+str(hobby)] = new_survey['id_hobby'].apply(lambda x: 1 if isinstance(x, list) and hobby in x else 0)

#     new_survey.drop(columns=['id_hobby'], inplace=True)

#     # -- diseases
#     new_diseases = pd.read_csv(f"{main_path}/new_survey_diseases{str(participant_number)}.csv")
#     diseases_grouped = new_diseases.groupby('id_survey')['id_disease'].apply(list).reset_index()
#     new_survey = pd.merge(new_survey, diseases_grouped, on='id_survey', how='left')

#     for disease in list(set(new_diseases['id_disease'].tolist())):
#         new_survey["disease_"+str(disease)] = new_survey['id_disease'].apply(lambda x: 1 if isinstance(x, list) and disease in x else 0)
    
#     new_survey.drop(columns=['id_disease'], inplace=True)

#     # -- visual acuities
#     new_visual_acuities = pd.read_csv(f"{main_path}/new_survey_visual_acuities{str(participant_number)}.csv")
#     visual_acuities_grouped = new_visual_acuities.groupby('id_survey')['id_visual_acuities'].apply(list).reset_index()
#     new_survey = pd.merge(new_survey, visual_acuities_grouped, on='id_survey', how='left')
    
#     for visual_acuity in list(set(new_visual_acuities['id_visual_acuities'].tolist())):
#         new_survey["visual_acuity"+str(visual_acuity)] = new_survey['id_visual_acuities'].apply(lambda x: 1 if isinstance(x, list) and visual_acuity in x else 0)
    
#     new_survey.drop(columns=['id_visual_acuities'], inplace=True)

#     return new_survey, new_ratings