

# def data():
#     with sshtunnel.SSHTunnelForwarder(
#         ('rachel.waik.eu', 22),
#         ssh_username = 'emorecsys',
#         ssh_password = 'TMgx64gP8iD37wmx3wdn',
#         remote_bind_address = ('127.0.0.1', 3306)
#     ) as tunnel:
#         connection = mysql.connector.connect(
#             user = 'emorecsys_reader',
#             password = 'dDoZzsLz3r7pkJ9YooXn',
#             host = '127.0.0.1',
#             port = tunnel.local_bind_port,
#             database = 'emorecsys', 
#             use_pure=True
#         )
    
#     mycursor = connection.cursor()

#     query2 = "SELECT * FROM surveys"
#     df_surveys = pd.read_sql(query2, con = connection)

#     connection.close()
#     return df_surveys['id_survey'].iloc[-1]


# if __name__ == "__main__":
#     user_id = data()
    # print(f'Your user ID is {user_id}')

import mysql.connector
import sshtunnel
import pandas as pd

def data():
    with sshtunnel.SSHTunnelForwarder(
        ('rachel.waik.eu', 22),
        ssh_username = 'emorecsys',
        ssh_password = 'TMgx64gP8iD37wmx3wdn',
        remote_bind_address = ('127.0.0.1', 3306)
    ) as tunnel:
        connection = mysql.connector.connect(
            user = 'emorecsys_reader',
            password = 'dDoZzsLz3r7pkJ9YooXn',
            host = '127.0.0.1',
            port = tunnel.local_bind_port,
            database = 'emorecsys', 
            use_pure=True
        )
        mycursor = connection.cursor()
        query1 = "SELECT * FROM ratings" 
        df_ratings = pd.read_sql(query1, con = connection)

        query2 = "SELECT * FROM surveys"
        df_surveys = pd.read_sql(query2, con = connection)
        connection.close()   

        return df_ratings, df_surveys

if __name__ == "__main__":
    ratings, surveys = data()
    print(surveys)

    user_id = surveys['id'].iloc[-1]
    print(f'Your user ID is {user_id}')