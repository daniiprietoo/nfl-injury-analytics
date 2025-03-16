import os
# import sys
# module_path = os.path.abspath(os.path.join('..', 'src'))
# if module_path not in sys.path:
#   sys.path.append(module_path)

from src.database.database_client import DatabaseClient
from google.cloud.sql.connector import Connector
from dotenv import load_dotenv
load_dotenv()

connection_name = os.getenv("DB_CONNECTION_NAME")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

connector = Connector()

try:
    # Test the connection before creating client
    conn = connector.connect(
        connection_name,
        "pg8000",
        user=db_user,
        password=db_password,
        db=db_name
    )
    print("Test connection successful!")
    conn.close()
    
    # Now try with the client
    client = DatabaseClient(
      connector=connector,
      connection_name=connection_name,
      user=db_user,
      password=db_password,
      name=db_name
    )
    print("Database client created successfully")
    
    # Try a simple operation
    client.insert_dataset({
      'name': 'PlayList',
      'source': 'NFL 1st and Future - Analytics',
      'status': 'active',
    })
    print("Dataset inserted successfully")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()