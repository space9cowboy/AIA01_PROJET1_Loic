from src.data.data_loader import DataLoader
from config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
import pandas as pd
import os

class DataProcessing:
    """
    Classe pour charger, traiter et sauvegarder les données du projet.
    """
    def __init__(self, data):
        
        # Vérifie si 'data' est une liste, sinon on la transforme en liste
        data = [data] if isinstance(data, str) else data
        
        # Ajouter automatiquement 'items_prop_1' et 'items_prop_2' si l'un des 2 est demandé
        if 'items_prop_1' in data or 'items_prop_2 in' in data:
            data = list(set(data + ['items_prop_1', 'items_prop_2']))
            
            # Stocker les dataset a charger
            self.data = data
            
            # Initialisation du chargeur de données (DataLoader) : data dans data/raw
            self.data_loader = DataLoader(dataset_path=RAW_DATA_PATH, datasets=self.data)
            
            # Path du dossier : data/processed pour stocker les données traitées
            self.processed_path = os.path.join(PROCESSED_DATA_PATH)
            
            # Création du dossier 'processed_data' si il n'existe pas encore 
            os.makedirs(self.processed_path, exist_ok=True)
    
    # ================================================================
    # MÉTHODES POUR CHARGER ET SAUVEGARDER LES DONNÉES TRAITÉES
    # ================================================================
    
    def load_processed_data(self, dataset_name):
        """
        Charge un fichier CSV déjà traité, si disponible === ici items_prop_processed.csv
        """
        file_path = os.path.join(self.processed_path, f'{dataset_name}_processed.csv')
        
        # Vérifie si le fichier traité existe, sinon retourne None
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None
    
    def save_processed_data(self, data, dataset_name):
        """
        Sauvegarde les données traitées sous forme de CSV.
        """
        file_path = os.path.join(self.processed_path, f'{dataset_name}_processed.csv')
        
        # Sauvegarde du DataFrame sans l'index
        data.to_csv(file_path, index=False)
        
    def load_specific_data(self):
        """
        Charge les données brutes via le DataLoader.
        """
        return self.data_loader.load_data()
    
    # ================================================================
    # MÉTHODE PRINCIPALE POUR LE PRÉTRAITEMENT
    # ================================================================
    
    def preprocess_data(self):
        
        # Dictionnaire pour stocker les datasets traités
        processed_data = {}
        
        # Dictionnaire qui mappe les datasets aux fonctions de prétraitement correspondantes
        processing_map = {
            'items_prop': lambda: self.preprocess_items() if all (x in self.data for x in ['items_prop_1', 'items_prop_2']) else None,
        }
        
        # Boucle sur chaque dataset à traiter
        for dataset in set(self.data):
            key = 'items_prop' if 'items_prop' in dataset else dataset
            
            if key not in processed_data: # Éviter de traiter deux fois le même dataset
                try:
                    existing_data = self.load_processed_data(key)
                    if existing_data is not None:
                        # Lire la data dans le csv
                        processed_data[key] = existing_data
                    elif key in processing_map: # Appliquer la transformation si nécessaire
                        processed = processing_map[key]()
                        if processed_data is not None:
                            self.save_processed_data(processed, key) # Sauvegarder les données traitées
                            processed_data[key] = processed
                except Exception as e:
                    print(f"Error processing {key}: {e}")
                    
        return processed_data
    
    # ================================================================
    # MÉTHODE POUR TRAITER LES PROPRIÉTÉS DES ARTICLES
    # ================================================================
    
    def preprocess_items(self):
        '''Préretraite et fusionne les propriétés des items'''
        data = self.load_specific_data()
        
        # Vérifier que les deux fichiers existent dans les données brutes
        if 'items_prop_1' in data and 'items_prop_2' in data:
            
            # Fusion des deux parties en un seul DF
            items_combined = pd.concat([data['items_prop_1'], data['items_prop_2']])
            
            # Nettoyage des doublons
            items_cleaned = items_combined.drop_duplicates()
            
            # Conversion des timestamps au format DateTime
            if 'timestamp' in items_cleaned.columns:
                items_cleaned['timestamp'] = pd.to_datetime(items_cleaned['timestamp'], unit='ms')
            
            return items_cleaned
        return None