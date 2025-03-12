import os
from dotenv import load_dotenv
from database.database_client import DatabaseClient
from google.cloud.sql.connector import Connector

def main():
    load_dotenv()
    db_connection_name = os.getenv("DB_CONNECTION_NAME")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    

    connector = Connector()
    db_client = DatabaseClient(connector, db_connection_name, db_name, db_user, db_password)


if __name__ == "__main__":
    main()
