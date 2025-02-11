from src.data.data_loader import DataLoader
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
   
    loader = DataLoader(dataset_path= RAW_DATA_PATH, datasets=['category_tree'])
   
    logger.info("Data loaded successfully")
    data = loader.load_data()
    print(data['category_tree'].head())
    logger.info("Starting preprocessing")
    
if __name__ == "__main__":
    main()