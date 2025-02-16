from src.data.data_loader import DataLoader
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

class DataProcessings:
    def __init__(self, data):
        
        data = [data] if isinstance(data, str) else data
        if 'items_prop_1' in data or 'items_prop_2' in data:
            data = list(set(data + ['items_prop_1', 'items_prop_2']))
        self.data = data 
        self.data_loader = DataLoader(dataset_path=RAW_DATA_PATH, datasets=self.data)
        self.processed_path = os.path.join(PROCESSED_DATA_PATH)
    
    def load_processed_data(self, dataset_name):
        file_path = os.path.join(self.processed_path, f"{dataset_name}_processed.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None
    
    def save_processed_data(self, dataset_name, data):
        file_path = os.path.join(self.processed_path, f"{dataset_name}_processed_csv")
        data.to_csv(file_path, index=False)
        
    def load_specific_data(self,):
        data = self.data_loader.load_data()
        return data
    
    def preprocess_data(self):
        
        processed_data = {}
        
        processing_map = {
            'events': (self.analyze_events, self.preprocess_events),
            'category_tree': (self.analyze_category_tree, self.preprocess_category_tree),
            'items_prop': (self.analyze_items, self.preprocess_items),
            # 'events': self.preprocess_events,
            # 'category_tree': self.preprocess_category_tree,
            # 'items_prop': lambda: self.preprocess_items() if all(x in self.data for x in ['items_prop_1', 'items_prop_2']) else None
            # 'items_prop_1' : lambda: self.preprocess_items() if 'items_prop_2' in self.data else None,
            # 'items_prop_2' : lambda: self.preprocess_items() if 'items_prop_1' in self.data else None
            
        }
        for dataset in set(self.data):
            key = 'items_prop' if 'items_prop' in dataset else dataset
            logger.info(f"\nTraitement du dataset : {key}")
            if key not in processed_data:
                try:
                    existing_data = self.load_processed_data(key)
                    if existing_data is not None:
                        analyze_func, _ = processing_map[key]
                        analyze_func(existing_data)
                        processed_data[key] = existing_data
                        # processed_data[key] = existing_data
                    elif key in processing_map:
                        analyze_func, process_func = processing_map[key]
                        raw_data = self.load_specific_data()
                        if isinstance(raw_data, dict) and key in raw_data:
                            raw_data = raw_data[key]
                        analyze_func(raw_data)
                        processed = process_func()
                        if processed is not None:
                            self.save_processed_data(processed, key)
                            processed_data[key] = processed
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {key}: {e}")
                    raise
                        # processed = processing_map[key]()
                        # if processed is not None:
                        #     self.save_processed_data(key, processed)
                        #     processed_data[key] = processed
                # except Exception as e:
                #     print(f"Error processing {key}: {e}")
     
        return processed_data
    
    def preprocess_items(self, data):
        """Prétraite les données des propriétés des items"""
        logger.info("Début du prétraitement des propriétés des items")
        
        # Conversion du timestamp
        logger.info("Conversion du timestamp en datetime")
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Analyse des propriétés les plus fréquentes
        logger.info("Analyse des propriétés")
        prop_counts = data['property'].value_counts()
        logger.info(f"Nombre total de propriétés uniques: {len(prop_counts)}")
        
        # Sélection des propriétés les plus fréquentes
        top_n = 20
        top_properties = prop_counts.head(top_n).index
        logger.info(f"Sélection des {top_n} propriétés les plus fréquentes")
        
        # Création du pivot
        logger.info("Création du format pivot pour les propriétés principales")
        pivot_df = data[data['property'].isin(top_properties)].pivot_table(
            index='itemid',
            columns='property',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Ajout d'informations temporelles
        logger.info("Ajout des informations temporelles")
        time_stats = data.groupby('itemid')['datetime'].agg(['min', 'max']).reset_index()
        pivot_df = pivot_df.merge(time_stats, on='itemid', how='left')
        pivot_df.rename(columns={'min': 'first_seen', 'max': 'last_seen'}, inplace=True)
        
        logger.info(f"Dimensions finales: {pivot_df.shape}")
        
        # Analyse des données traitées
        self.analyze_items(pivot_df, is_processed=True)
        
        return pivot_df
    
    def preprocess_events(self, data):
        """Prétraite les données des événements"""
        logger.info("Début du prétraitement des événements")
        
        # Conversion du timestamp
        logger.info("Conversion du timestamp en datetime")
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
        
        # Création de features temporelles
        logger.info("Création des features temporelles")
        data['hour'] = data['datetime'].dt.hour
        data['day'] = data['datetime'].dt.day
        data['month'] = data['datetime'].dt.month
        data['day_of_week'] = data['datetime'].dt.day_name()
        
        # Indicateur de transaction
        logger.info("Création de la colonne has_transaction")
        data['has_transaction'] = data['transactionid'].notna().astype(int)
        logger.info(f"Nombre d'événements avec transaction: {data['has_transaction'].sum()}")
        
        # Calcul des durées entre événements par visiteur
        logger.info("Calcul des durées entre événements")
        data = data.sort_values(['visitorid', 'timestamp'])
        data['time_since_last_event'] = data.groupby('visitorid')['datetime'].diff()
        
        # Identification des sessions (30 minutes d'inactivité = nouvelle session)
        logger.info("Identification des sessions")
        session_threshold = pd.Timedelta(minutes=30)
        data['new_session'] = (data['time_since_last_event'] > session_threshold).astype(int)
        data['session_id'] = data.groupby('visitorid')['new_session'].cumsum()
        
        # Enrichissement avec des statistiques par session
        logger.info("Calcul des statistiques par session")
        session_stats = data.groupby(['visitorid', 'session_id']).agg({
            'itemid': 'nunique',
            'event': 'count',
            'has_transaction': 'sum'
        }).reset_index()
        
        session_stats.columns = ['visitorid', 'session_id', 'unique_items', 'event_count', 'transactions']
        
        # Fusion des statistiques de session avec les données principales
        data = data.merge(session_stats, on=['visitorid', 'session_id'])
        
        logger.info("Prétraitement des événements terminé")
        logger.info(f"Dimensions finales: {data.shape}")
        
        return data
    
    def preprocess_category_tree(self, data):
        """Prétraite les données de l'arbre des catégories"""
        logger.info("Début du prétraitement de l'arbre des catégories")
        
        # Création de la colonne level
        parent_dict = data.set_index('categoryid')['parentid'].to_dict()
        
        def get_category_level(cat_id, parent_dict, visited=None):
            if visited is None:
                visited = set()
            if cat_id in visited:
                return -1
            visited.add(cat_id)
            parent = parent_dict.get(cat_id)
            if pd.isna(parent):
                return 0
            level = get_category_level(parent, parent_dict, visited)
            return -1 if level == -1 else level + 1
        
        logger.info("Calcul des niveaux de catégories")
        data['level'] = data['categoryid'].apply(lambda x: get_category_level(x, parent_dict))
        
        # Création du chemin complet
        def get_category_path(cat_id, parent_dict, visited=None):
            if visited is None:
                visited = set()
            if cat_id in visited:
                return []
            visited.add(cat_id)
            parent = parent_dict.get(cat_id)
            if pd.isna(parent):
                return [cat_id]
            parent_path = get_category_path(parent, parent_dict, visited)
            return parent_path + [cat_id] if parent_path else [cat_id]
        
        logger.info("Création des chemins de catégories")
        data['category_path'] = data['categoryid'].apply(lambda x: ' > '.join(map(str, get_category_path(x, parent_dict))))
        
        logger.info("Prétraitement de l'arbre des catégories terminé")
        return data
    
    def analyze_events(self, data, is_processed=False):
        """Analyse le dataset des événements"""
        logger.info("\n\n" + "="*50)
        logger.info(f"=== Analyse du dataset Events {'(traité)' if is_processed else '(brut)'} ===")
        logger.info("="*50 + "\n")
        
        # Informations de base
        logger.info("\n" + "="*30)
        logger.info("Dimensions:")
        logger.info("="*30 + "\n")
        logger.info(f"{data.shape}\n")
        
        logger.info("\n" + "="*30)
        logger.info("Types des colonnes:")
        logger.info("="*30 + "\n")
        for col, dtype in data.dtypes.items():
            logger.info(f"{col:<20} {dtype}")
        logger.info("")
        
        # Valeurs manquantes
        logger.info("\n" + "="*30)
        logger.info("Valeurs manquantes par colonne:")
        logger.info("="*30 + "\n")
        for col, count in data.isnull().sum().items():
            logger.info(f"{col:<20} {count}")
        logger.info("")
        
        # Distribution des événements
        logger.info("\n" + "="*30)
        logger.info("Distribution des types d'événements:")
        logger.info("="*30 + "\n")
        for event, count in data['event'].value_counts().items():
            logger.info(f"{event:<20} {count}")
        logger.info("")
        
        # Analyse des transactions
        logger.info("\n" + "="*30)
        logger.info("Statistiques générales:")
        logger.info("="*30 + "\n")
        n_transactions = data['transactionid'].notna().sum()
        n_visitors = data['visitorid'].nunique()
        n_items = data['itemid'].nunique()
        logger.info(f"Nombre total d'événements:   {len(data):,}")
        logger.info(f"Nombre de transactions:      {n_transactions:,}")
        logger.info(f"Nombre de visiteurs uniques: {n_visitors:,}")
        logger.info(f"Nombre d'items uniques:      {n_items:,}")
        logger.info("")
        
        # Analyse temporelle
        if 'timestamp' in data.columns:
            logger.info("\n" + "="*30)
            logger.info("Plage temporelle des données:")
            logger.info("="*30 + "\n")
            timestamp_start = pd.to_datetime(data['timestamp'].min(), unit='ms')
            timestamp_end = pd.to_datetime(data['timestamp'].max(), unit='ms')
            logger.info(f"Début:         {timestamp_start}")
            logger.info(f"Fin:           {timestamp_end}")
            logger.info(f"Durée totale:  {timestamp_end - timestamp_start}")
            logger.info("")
            
            # Distribution par jour de la semaine
            data['temp_date'] = pd.to_datetime(data['timestamp'], unit='ms')
            logger.info("\n" + "="*30)
            logger.info("Distribution par jour de la semaine:")
            logger.info("="*30 + "\n")
            for day, count in data['temp_date'].dt.day_name().value_counts().items():
                logger.info(f"{day:<15} {count:,}")
            data.drop('temp_date', axis=1, inplace=True)
            logger.info("")
        
        # Analyse des séquences d'événements par visiteur
        logger.info("\n" + "="*30)
        logger.info("Statistiques des événements par visiteur:")
        logger.info("="*30 + "\n")
        events_per_visitor = data.groupby('visitorid').size()
        logger.info(f"Moyenne:  {events_per_visitor.mean():.2f}")
        logger.info(f"Médiane:  {events_per_visitor.median():.2f}")
        logger.info(f"Maximum:  {events_per_visitor.max()}")
        logger.info("")
        
        # Taux de conversion
        if 'transactionid' in data.columns:
            logger.info("\n" + "="*30)
            logger.info("Taux de conversion:")
            logger.info("="*30 + "\n")
            visitors_with_transaction = data[data['transactionid'].notna()]['visitorid'].nunique()
            conversion_rate = (visitors_with_transaction / n_visitors) * 100
            logger.info(f"Taux de conversion global: {conversion_rate:.2f}%")
            logger.info("")
        
        # Suggestions de prétraitement
        logger.info("\n" + "="*30)
        logger.info("Suggestions de prétraitement:")
        logger.info("="*30 + "\n")
        logger.info("1. Conversion du timestamp en datetime")
        logger.info("2. Création de features temporelles (heure, jour, mois, jour de semaine)")
        logger.info("3. Calcul des durées entre événements par visiteur")
        logger.info("4. Création d'indicateurs de session")
        logger.info("5. Agrégation des comportements par visiteur")
        logger.info("6. Identification des séquences d'événements courantes")
        logger.info("7. Enrichissement avec les données catégorielles des items")
        logger.info("\n")
        
        return data
    
    def analyze_items(self, data, is_processed=False):
        """Analyse le dataset des propriétés des items"""
        stage = "(traité)" if is_processed else "(brut)"
        logger.info(f"\n\n{'='*50}")
        logger.info(f"=== Analyse du dataset Items Properties {stage} ===")
        logger.info(f"{'='*50}\n")
        
        # Dimensions
        logger.info(f"\n{'='*30}")
        logger.info("Dimensions:")
        logger.info(f"{'='*30}\n")
        logger.info(f"{data.shape}\n")
        
        # Types des colonnes
        logger.info(f"\n{'='*30}")
        logger.info("Types des colonnes:")
        logger.info(f"{'='*30}\n")
        for col, dtype in data.dtypes.items():
            logger.info(f"{col:<20} {dtype}")
        logger.info("")
        
        if not is_processed:
            # Statistiques sur les propriétés
            logger.info(f"\n{'='*30}")
            logger.info("Distribution des propriétés:")
            logger.info(f"{'='*30}\n")
            if 'property' in data.columns:
                prop_dist = data['property'].value_counts().head(10)
                for prop, count in prop_dist.items():
                    logger.info(f"{prop:<20} {count:,}")
            logger.info("")
            
            # Nombre d'items uniques
            n_items = data['itemid'].nunique()
            logger.info(f"\n{'='*30}")
            logger.info("Statistiques générales:")
            logger.info(f"{'='*30}\n")
            logger.info(f"Nombre d'items uniques: {n_items:,}")
            logger.info(f"Nombre total d'enregistrements: {len(data):,}")
            logger.info(f"Moyenne de propriétés par item: {len(data)/n_items:.2f}")
        else:
            # Analyse des colonnes après pivot
            logger.info(f"\n{'='*30}")
            logger.info("Statistiques des colonnes pivotées:")
            logger.info(f"{'='*30}\n")
            
            for col in data.columns:
                if col not in ['itemid', 'first_seen', 'last_seen']:
                    n_unique = data[col].nunique()
                    n_missing = data[col].isna().sum()
                    logger.info(f"{col}:")
                    logger.info(f"  - Valeurs uniques: {n_unique}")
                    logger.info(f"  - Valeurs manquantes: {n_missing} ({n_missing/len(data)*100:.2f}%)\n")
        
        # Plage temporelle
        if 'first_seen' in data.columns and 'last_seen' in data.columns:
            logger.info(f"\n{'='*30}")
            logger.info("Plage temporelle des données:")
            logger.info(f"{'='*30}\n")
            logger.info(f"Première observation: {data['first_seen'].min()}")
            logger.info(f"Dernière observation:  {data['last_seen'].max()}")
            logger.info(f"Durée totale: {data['last_seen'].max() - data['first_seen'].min()}")
        
        # Suggestions d'optimisation
        logger.info(f"\n{'='*30}")
        logger.info("Suggestions d'optimisation:")
        logger.info(f"{'='*30}\n")
        logger.info("1. Sélection des propriétés les plus pertinentes")
        logger.info("2. Normalisation des valeurs numériques")
        logger.info("3. Encodage des valeurs catégorielles")
        logger.info("4. Gestion des valeurs manquantes")
        logger.info("5. Réduction de la dimensionnalité")
        logger.info("6. Création de features agrégées")
        logger.info("7. Optimisation des types de données")
        logger.info("8. Création d'index pour les recherches fréquentes")
        logger.info("9. Compression des données textuelles")
        logger.info("10. Mise en cache des calculs fréquents")
        logger.info("\n")
        
        return data
    
    def analyze_category_tree(self, data, is_processed=False):
        """Analyse le dataset de l'arbre des catégories"""
        logger.info("\n\n" + "="*50)
        logger.info(f"=== Analyse du dataset Category Tree {'(traité)' if is_processed else '(brut)'} ===")
        logger.info("="*50 + "\n")
        
        # Informations de base
        logger.info("\n" + "="*30)
        logger.info("Dimensions:")
        logger.info("="*30 + "\n")
        logger.info(f"{data.shape}\n")
        
        logger.info("\n" + "="*30)
        logger.info("Types des colonnes:")
        logger.info("="*30 + "\n")
        for col, dtype in data.dtypes.items():
            logger.info(f"{col:<20} {dtype}")
        logger.info("")
        
        # Analyse de la structure de l'arbre
        logger.info("\n" + "="*30)
        logger.info("Analyse de la structure de l'arbre:")
        logger.info("="*30 + "\n")
        
        n_categories = len(data)
        n_parents = data['parentid'].nunique()
        n_roots = data['parentid'].isna().sum()
        
        logger.info(f"Nombre total de catégories:        {n_categories:,}")
        logger.info(f"Nombre de parents uniques:         {n_parents}")
        logger.info(f"Catégories sans parent (racines):  {n_roots}")
        logger.info("")
        
        # Distribution des niveaux
        if 'level' in data.columns:
            logger.info("\n" + "="*30)
            logger.info("Distribution des niveaux de profondeur:")
            logger.info("="*30 + "\n")
            
            level_dist = data['level'].value_counts().sort_index()
            for level, count in level_dist.items():
                logger.info(f"Niveau {level:<3} : {count} catégories")
            logger.info("")
        
        logger.info("\n")
        
        # Suggestions de prétraitement
        logger.info("\n" + "="*30)
        logger.info("Suggestions de prétraitement:")
        logger.info("="*30 + "\n")
        logger.info("1. Ajout d'une colonne 'level' pour chaque catégorie")
        logger.info("2. Création d'un chemin complet pour chaque catégorie")
        logger.info("3. Identification et correction des cycles")
        logger.info("4. Gestion des catégories orphelines")
        logger.info("5. Création d'une structure hiérarchique complète")
        logger.info("\n")
        
        return data