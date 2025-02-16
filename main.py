from src.data.data_loader import DataLoader
from src.data.data_processings import DataProcessings


from config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
import logging
import argparse

# ================================================================
# CONFIGURATION DU LOGGING
# ================================================================

def setup_logging():
    """
    Configure le logging pour le projet
    """
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def parse_arguments():
    """
    Parse les arguments de la ligne de commande
    """
    parser = argparse.ArgumentParser(description='Traitement des données')
    parser.add_argument('--datasets', nargs='+', default=['events'],
                       help='Liste des datasets à traiter(events, category_tree, items_prop)')
    # parser.add_argument('--force-download', action='store_true',
    #                    help='Force le re-téléchargement des données')
    # parser.add_argument('--force-processing', action='store_true',
    #                    help='Force le retraitement des données')
    # parser.add_argument('--enrich-events', action='store_true',
    #                    help='Enrichir le dataset des événements')
    # parser.add_argument('--force-enrichment', action='store_true',
    #                    help='Force le retraitement de l\'enrichissement')
    # parser.add_argument('--viz', action='store_true',
    #                    help='Afficher les visualisations')
    # parser.add_argument('--export', type=str,
    #                    help='Exporter les données enrichies en CSV (ex: export.csv)')
    
    return parser.parse_args()

# ================================================================
# FONCTION PRINCIPALE
# ================================================================

def main():
    """
    Fonction principale pour exécuter le projet
    """
    # Configuration du logging
    args = parse_arguments()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting dataloading import")
    
    logger.info(f"Datasets sélectionnés pour le traitement : {args.datasets}")
    
    try: 
        logger.info('Init du Dataprocessing')
        data_processing = DataProcessings(data=args.datasets)
        
        logger.info('Début du prétraitement')
        finaldf = data_processing.preprocess_data()
        
        logger.info('Prétraitement términé')
        
        for dataset_name, dataset in finaldf.items():
            print(f"\nApercu du dataset : {dataset_name}:")
            print(dataset.head())
        
        if dataset_name == 'events':
            logger.info("stats supp pour les events")
            print("\nDIstrib des transac")
            print(dataset['has_transaction'].value_counts())
            print("\nDistrib des types d'events")
            print(dataset['event'].value_counts())
    
    except Exception as e:
        logger.error(f"Erreur dans le traitement : {e}")
        
    
if __name__ == "__main__":
    main()