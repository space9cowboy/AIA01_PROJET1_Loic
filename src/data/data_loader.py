import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
# from dotenv import load_dotenv
# load_dotenv()

class DataLoader:
    """
    Classe pour télécharger et charger les données du dataset ecommerce-dataset
    """
    def __init__(self, dataset_path="data"):
        """
        Initialisation le chemin du dataset
        """
        self.dataset_path = dataset_path
        self.api = KaggleApi()
        self.api.authenticate()
    
    def download_data(self):
        """
        Télécharge les données du dataset ecommerce-dataset
        """
        self.api.dataset_download_files(
            "retailrocket/ecommerce-dataset",
            path=self.dataset_path,
            unzip=True
        )

    def load_data(self):
        """
        Charge les données du dataset emcommerce-dataset
        """
        try:
            category_tree = pd.read_csv(os.path.join(self.dataset_path, "category_tree.csv"))
            items_prop_1 = pd.read_csv(os.path.join(self.dataset_path, "item_properties_part1.csv"))
            items_prop_2 = pd.read_csv(os.path.join(self.dataset_path, "item_properties_part2.csv"))
            events = pd.read_csv(os.path.join(self.dataset_path, "events.csv"))
            
            return category_tree, items_prop_1, items_prop_2, events
        
        except Exception as e:

            print(f"Error loading data: {e}")
            self.download_data()

            return self.load_data()
           
        