import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

# Add the root directory of your project to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, data_path=None):
        logging.info('Entered the data ingestion method or component')
        try:
            # Use the provided data path or default to a specific file
            if data_path is None:
                data_path = r'notebook\data\stud.csv'  # Default path
            
            df = pd.read_csv(data_path)  # Ensure the file path is correct
            logging.info(f'Read the dataset from {data_path} as DataFrame with shape: {df.shape}')
            
            # Create directories for artifacts if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train-test split initiated')
            # Perform the train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info(f'Data ingestion completed: Train shape: {train_set.shape}, Test shape: {test_set.shape}')
        
            # Return the paths to the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Initialize and apply the data transformation component
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Initialize the model trainer component and start training
    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f'Model R2 Score: {score}')
