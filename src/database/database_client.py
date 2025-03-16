from sqlalchemy import create_engine
from .db_models import Base, Dataset, Preprocessor, PreprocessorStrategy, Algorithm, AlgorithmParameters, Model, ModelAccuracy, Visualization
from sqlalchemy.orm import sessionmaker
from typing import Dict

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


    def insert_dataset(self, data: Dict):
        session = self.Session()
        try:
            dataset = Dataset(
                name=data['name'],
                source=data.get('source'),
                status=data.get('status')
            )
            session.add(dataset)
            session.commit()
            print(f"Dataset '{data['name']}' inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting dataset: {e}")
        finally:
            session.close()


    def insert_preprocessor(self, data: Dict):
        session = self.Session()
        try:
            preprocessor = Preprocessor(
                dataset_id=data['dataset_id']
            )
            session.add(preprocessor)
            session.commit()
            print(f"Preprocessor for dataset_id {data['dataset_id']} inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting preprocessor: {e}")
        finally:
            session.close()
    
    def insert_preprocessor_strategy(self, data: Dict):
        session = self.Session()
        try:
            strategy = PreprocessorStrategy(
                preprocessing_id=data['preprocessing_id'],
                strategy_type=data['strategy_type'],
                strategy_value=data.get('strategy_value')
            )
            session.add(strategy)
            session.commit()
            print(f"Preprocessor strategy for preprocessing_id {data['preprocessing_id']} inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting preprocessor strategy: {e}")
        finally:
            session.close()
    
    def insert_algorithm(self, data: Dict):
        session = self.Session()
        try:
            algorithm = Algorithm(
                name=data['name']
            )
            session.add(algorithm)
            session.commit()
            print(f"Algorithm '{data['name']}' inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting algorithm: {e}")
        finally:
            session.close()
    
    def insert_algorithm_parameters(self, data: Dict):
        session = self.Session()
        try:
            params = AlgorithmParameters(
                algorithm_id=data['algorithm_id'],
                parameter_name=data['parameter_name'],
                parameter_value=data['parameter_value']
            )
            session.add(params)
            session.commit()
            print(f"Parameters for algorithm_id {data['algorithm_id']} inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting algorithm parameters: {e}")
        finally:
            session.close()
    
    def insert_model(self, data: Dict):
        session = self.Session()
        try:
            model = Model(
                algorithm_id=data['algorithm_id'],
                dataset_id=data['dataset_id']
            )
            session.add(model)
            session.commit()
            print(f"Model inserted successfully with algorithm_id {data['algorithm_id']} and dataset_id {data['dataset_id']}.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting model: {e}")
        finally:
            session.close()
    
    def insert_model_accuracy(self, data: Dict):
        session = self.Session()
        try:
            model_acc = ModelAccuracy(
                model_id=data['model_id'],
                accuracy_score=data['accuracy_score']
            )
            session.add(model_acc)
            session.commit()
            print(f"Model accuracy for model_id {data['model_id']} inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting model accuracy: {e}")
        finally:
            session.close()
    
    def insert_visualization(self, data: Dict):
        session = self.Session()
        try:
            viz = Visualization(
                model_id=data['model_id'],
                chart_type=data['chart_type']
            )
            session.add(viz)
            session.commit()
            print(f"Visualization for model_id {data['model_id']} inserted successfully.")
        except Exception as e:
            session.rollback()
            print(f"Error inserting visualization: {e}")
        finally:
            session.close()