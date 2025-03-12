# db_models.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, ForeignKey, Enum, func
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Dataset(Base):
    __tablename__ = 'dataset'
    
    dataset_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    upload_date = Column(DateTime, default=func.now())
    source = Column(String(255))
    status = Column(String(50))
    
class Preprocessor(Base):
    __tablename__ = 'preprocessor'
    
    preprocessing_id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'), nullable=False)
    
class PreprocessorStrategy(Base):
    __tablename__ = 'preprocessor_strategy'
    
    strategy_id = Column(Integer, primary_key=True, autoincrement=True)
    preprocessing_id = Column(Integer, ForeignKey('preprocessor.preprocessing_id'), nullable=False)
    strategy_type = Column(Enum("scaling", "missing_value", name="strategy_enum"), nullable=False)
    strategy_value = Column(String(255))
    
class Algorithm(Base):
    __tablename__ = 'algorithm'
    
    algorithm_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    
class AlgorithmParameters(Base):
    __tablename__ = 'algorithm_parameters'
    
    parameter_id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_id = Column(Integer, ForeignKey('algorithm.algorithm_id'), nullable=False)
    parameter_name = Column(String(255), nullable=False)
    parameter_value = Column(String(255), nullable=False)
    
class Model(Base):
    __tablename__ = 'model'
    
    model_id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_id = Column(Integer, ForeignKey('algorithm.algorithm_id'), nullable=False)
    dataset_id = Column(Integer, ForeignKey('dataset.dataset_id'), nullable=False)
    
class ModelAccuracy(Base):
    __tablename__ = 'model_accuracy'
    
    accuracy_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('model.model_id'), nullable=False)
    accuracy_score = Column(Float)
    
class Visualization(Base):
    __tablename__ = 'visualization'
    
    visualization_id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('model.model_id'), nullable=False)
    chart_type = Column(String(255))
