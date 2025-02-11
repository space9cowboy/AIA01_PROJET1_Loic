from src.data.data_loader import DataLoader

from config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
import pandas as pd
import os

class DataProcessing:
    def __init__(self, data):
        data = [data] if isinstance(data, str) else data
        if 'items_prop_1' in data or 'items_prop_2 in' in data:
            data = list(set(data + ['items_prop_1', 'items_prop_2']))
            self.data = data
            self.data_loader = DataLoader(dataset_path=RAW_DATA_PATH, datasets=self.data)
            self.processed_path = os.path.join(PROCESSED_DATA_PATH)
            os.makedirs(self.processed_path, exist_ok=True)
    
    # lire le csv
    def load_processed_data(self, dataset_name):
        file_path = os.path.join(self.processed_path, f'{dataset_name}_processed.csv')
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        # data = pd.read_csv(self.processed_path)
        return None
    
    def save_processed_data(self, data, dataset_name):
        file_path = os.path.join(self.processed_path, f'{dataset_name}_processed.csv')
        data.to_csv(file_path, index=False)
        
    def load_specific_data(self):
        return self.data_loader.load_data()
       
    
    def preprocess_data(self):
        processed_data = {}
        processing_map = {
            'items_prop': lambda: self.preprocess_items() if all (x in self.data for x in ['items_prop_1', 'items_prop_2']) else None,
        }
        
        for dataset in set(self.data):
            key = 'items_prop' if 'items_prop' in dataset else dataset
            
            if key not in processed_data:
                try:
                    existing_data = self.load_processed_data(key)
                    if existing_data is not None:
                        # Lire la data dans le csv
                        processed_data[key] = existing_data
                    elif key in processing_map:
                        processed = processing_map[key]()
                        if processed_data is not None:
                            self.save_processed_data(processed, key)
                            processed_data[key] = processed
                except Exception as e:
                    print(f"Error processing {key}: {e}")
                    # if key in processing_map and key not in processed_data:
                    #     processed_data[key] = processing_map[key]()
        return processed_data
        
    
    def preprocess_items(self):
        '''Préretraite et fusionne les propriétés des items'''
        data = self.load_specific_data()
        if 'items_prop_1' in data and 'items_prop_2' in data:
            
            # Fusion des deux parties
            items_combined = pd.concat([data['items_prop_1'], data['items_prop_2']])
            
            # Nettoyage des doublons
            items_cleaned = items_combined.drop_duplicates()
            
            # Conversion des timestamps
            if 'timestamp' in items_cleaned.columns:
                items_cleaned['timestamp'] = pd.to_datetime(items_cleaned['timestamp'], unit='ms')
            
            return items_cleaned
        return None