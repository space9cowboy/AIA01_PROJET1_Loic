import pandas as pd
import os
from kaggle.api.kaggle_api_extended import KaggleApi
# from dotenv import load_dotenv
# load_dotenv()

class DataLoader:
    """
    Classe pour télécharger et charger les données du dataset ecommerce-dataset
    """
    def __init__(self, dataset_path="data", datasets=None):
        """
        Initialisation le chemin du dataset
        """
        self.dataset_path = dataset_path
        
        # Dictionnaire des datasets disponibles (nom logique → fichier CSV correspondant)
        self.available_datasets = datasets = {
            "category_tree": "category_tree.csv",
            "items_prop_1": "item_properties_part1.csv",
            "items_prop_2": "item_properties_part2.csv",
            "events": "events.csv",
        }
        
        # Si `datasets` est spécifié, charge uniquement ces fichiers, sinon charge tout
        self.datasets = datasets if datasets else list(self.available_datasets.keys())
        
        # Initialisation de l'API Kaggle et authentification
        self.api = KaggleApi()
        self.api.authenticate()
    
    # ================================================================
    # MÉTHODE POUR TÉLÉCHARGER LES DONNÉES
    # ================================================================
    
    def download_data(self):
        """
        Télécharge les données du dataset ecommerce-dataset
        """
        self.api.dataset_download_files(
            "retailrocket/ecommerce-dataset",
            path=self.dataset_path,
            unzip=True
        )

    # ================================================================
    # MÉTHODE POUR CHARGER LES DONNÉES EN DATAFRAMES
    # ================================================================
    
    def load_data(self):
        """
        Charge les données du dataset emcommerce-dataset
        """
        try:
            loaded_data = {} # Dictionnaire pour stocker les DataFrames
            files_missing = False
            for dataset in self.datasets: # Parcours tous les datasets demandés
                file_path = os.path.join(self.dataset_path, self.available_datasets[dataset])
                if os.path.exists(file_path): # Si le fichier csv existe on le charge en DF
                    loaded_data[dataset] = pd.read_csv(file_path)
                else:
                    files_missing = True # Si fichier manquant on recharge tout
                    print(f"File {file_path} does not exist")
            if files_missing:
                self.download_data()
                return self.load_data()
            return loaded_data
            
        except Exception as e:

            print(f"Error loading data: {e}")
            self.download_data()

            return self.load_data()
           
        