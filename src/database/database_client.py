from sqlalchemy import create_engine
from db_models import Base, Dataset, Preprocessor, PreprocessorStrategy, Algorithm, AlgorithmParameters, Model, ModelAccuracy, Visualization
from sqlalchemy.orm import sessionmaker

class DatabaseClient:
    def __init__(self, connector, connection_name:str, name: str, user: str, password: str):
        def get_conn():
            try:
                conn = connector.connect(
                    connection_name,
                    "pg8000",
                    user=user,
                    password=password,
                    db=name
                )
                return conn
            except Exception as e:
                print(f'Error connecting to PostgreSQL: {e}')
        
        self.engine = create_engine(
            f'postgresql+pg8000://',
            creator=get_conn,
            pool_pre_ping=True
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)