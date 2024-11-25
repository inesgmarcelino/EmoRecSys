import pandas as pd
import mysql.connector
import sshtunnel


def connecting():
    with sshtunnel.SSHTunnelForwarder(
        ('', 22),
        ssh_username = '',
        ssh_password = '',
        remote_bind_address = ('127.0.0.1', 3306)
    ) as tunnel:
        connection = mysql.connector.connect(
            user = '',
            password = '',
            host = '127.0.0.1',
            port = tunnel.local_bind_port,
            database = '', 
            use_pure=True
        )
    
    return connection

def get_ratings():
    connection = connecting()

    query1 = "SELECT * FROM ratings" 
    df_ratings = pd.read_sql(query1, con = connection)

    connection.close()
    return df_ratings

def get_userid():
    connection = connecting()

    query2 = "SELECT * FROM surveys"
    df_surveys = pd.read_sql(query2, con = connection)

    connection.close()
    return df_surveys['id_survey'].iloc[-1]


if __name__ == "__main__":
    user_id = get_userid()
    print(f'Your user ID is {user_id}')
