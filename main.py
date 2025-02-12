from src.data.data_loader import DataLoader
from src.data.data_processing import DataProcessing
from config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
import logging

# ================================================================
# CONFIGURATION DU LOGGING
# ================================================================

def setup_logging():
    """
    Configure le logging pour le projet
    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


# ================================================================
# FONCTION PRINCIPALE
# ================================================================

def main():
    """
    Fonction principale pour exécuter le projet
    """
    # Configuration du logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting dataloading import")
    data_processing = DataProcessing(data=['items_prop_1'])
    
    # Exécute le pipeline de prétraitement
    finaldf = data_processing.preprocess_data()
    
    # ================================================================
    # ANALYSE DES DONNÉES
    # ================================================================
    
    for dataset_name, dataset in finaldf.items():
        # print(f"Dataset: {dataset_name}")
        # print(dataset.head())
        # print(dataset.info())
        # print(dataset.describe())
        # print(dataset['itemid'].value_counts())
        # print(dataset['timestamp'].value_counts())
        # temp = dataset.groupby('itemid').agg({'itemid': 'size', 'timestamp': 'count'})
        # temp = temp.rename(columns={'itemid': 'count_itemid', 'timestamp': 'count_timestamp'})
        # temp = temp.reset_index()
        # temp.sort_values(by='count_itemid', ascending=False, inplace=True)
        dataunique = dataset[dataset['itemid'] == 158903]
        print(dataunique.head(10))
        # print(temp.head(10))
        print("\n")
    logger.info("Data loaded successfully")

    logger.info("Starting preprocessing")
    
if __name__ == "__main__":
    main()