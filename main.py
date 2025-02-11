from src.data.data_loader import DataLoader
from src.data.data_processing import DataProcessing
from config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
import logging

def setup_logging():
    """
    Configure le logging pour le projet
    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

def main():
    """
    Fonction principale pour exécuter le projet
    """
    # Configuration du logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting dataloading import")
   
    # loader = DataLoader(dataset_path= RAW_DATA_PATH, datasets=['category_tree'])
    data_processing = DataProcessing(data=['items_prop_1'])
    finaldf = data_processing.preprocess_data()
    
    for dataset_name, dataset in finaldf.items():
        print(f"Dataset: {dataset_name}")
        print(dataset.head())
        print("\n")
    logger.info("Data loaded successfully")
    # print(finaldf.head())
    logger.info("Starting preprocessing")
    
if __name__ == "__main__":
    main()