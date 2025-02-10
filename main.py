from src.data.data_loader import DataLoader
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
    data_loader = DataLoader()
    category_tree, items_prop_1, items_prop_2, events = data_loader.load_data()
    logger.info("Data loaded successfully")
    logger.info("Starting preprocessing")

    print(items_prop_1.head())
    print(items_prop_2.head())
    print(category_tree.head())
    print(events.head())

if __name__ == "__main__":
    main()