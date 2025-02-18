from src.data.data_exporter import DataExporter
from src.data.data_loader import DataLoader
from src.data.data_processings import DataProcessings

from config import DATA_PATH, RAW_DATA_PATH, PROCESSED_DATA_PATH
import logging
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

plt.show(block=True)

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
    parser.add_argument('--datasets', nargs='+', default=['events', 'items_prop', 'category_tree'],
                        help='Liste des datasets à traiter (events, category_tree, items_prop)')
    parser.add_argument('--export', type=str, help='Exporter les données enrichies en CSV (ex: export.csv)')
    
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

    logger.info(" Démarrage du processus de traitement des données")

    try:
        logger.info("Initialisation du DataProcessing")
        data_processing = DataProcessings(data=args.datasets)

        logger.info("⏳ Début du prétraitement des données")
        finaldf = data_processing.preprocess_data()
        logger.info(f"Prétraitement terminé. Nombre de datasets traités : {len(finaldf)}")

        # Vérification des datasets
        for dataset_name, dataset in finaldf.items():
            logger.info(f"📊 Analyse du dataset: {dataset_name}")
            print(f"\nAperçu du dataset {dataset_name}:")
            print(dataset.head())

        if "events" in finaldf:
            event_counts = finaldf["events"]["event"].value_counts()
            day_counts = finaldf["events"]["day_of_week"].value_counts()
            conversion_rate = (finaldf["events"]["has_transaction"].sum() / len(finaldf["events"])) * 100
            unique_visitors = finaldf["events"]["visitorid"].nunique()
            unique_items = finaldf["items_prop"]["itemid"].nunique()
            visits_per_day = finaldf["events"]["day_of_week"].value_counts()
            
            transactions = finaldf["events"][finaldf["events"]["event"] == "transaction"]
            views = finaldf["events"][finaldf["events"]["event"] == "view"]
            merged = transactions.merge(views, on="visitorid", suffixes=("_trans", "_view"))

            # Génération des visualisations
            data_processing.visualize_event_distribution(event_counts)
            data_processing.visualize_activity_per_day(day_counts)
            data_processing.visualize_conversion_rate(conversion_rate)
            data_processing.visits_by_day_analysis(visits_per_day)
            data_processing.unique_visitors_analysis(unique_visitors)
            data_processing.unique_items_analysis(unique_items)

            data_processing.session_duration_analysis(merged)
            data_processing.visualize_events_by_hour(finaldf["events"])
            data_processing.visualize_transactions_by_day(finaldf["events"])
            data_processing.visualize_sessions_per_user(finaldf["events"])
            data_processing.visualize_time_between_events(finaldf["events"])
            data_processing.visualize_top_items(finaldf["items_prop"])
            data_processing.visualize_active_visitors(finaldf["events"])
            data_processing.visualize_top_cart_items(finaldf["events"])
            data_processing.visualize_cart_abandonment(finaldf["events"])
            data_processing.analyze_time_to_purchase(finaldf["events"])
            data_processing.visualize_retention_rate(finaldf["events"])
            data_processing.visualize_funnel_conversion(finaldf["events"])

            # Exécution des A/B Tests
            data_processing.ab_test_add_to_cart(finaldf["events"])
            data_processing.ab_test_peak_hours(finaldf["events"])
            data_processing.ab_test_add_to_cart_button(finaldf["events"])

        if "items_prop" in finaldf:
            data_processing.ab_test_discount_effect(finaldf["events"], finaldf["items_prop"])
            unique_items = finaldf["items_prop"]["itemid"].nunique()
            data_processing.unique_items_analysis(unique_items)

        if "category_tree" in finaldf:
            data_processing.visualize_category_structure(finaldf["category_tree"])

        # Compression et export des graphiques
        data_processing.compress_graphs()
        logger.info("📁 Tous les graphiques ont été générés et archivés dans `exports/graphs.zip`")

        #  **EXPORTATION DES DONNÉES AVEC LIEN DE TÉLÉCHARGEMENT**
        if args.export:
            logger.info(f"📂 Export des données enrichies vers {args.export}")
            
            exporter = DataExporter()
            exporter.export_to_csv(finaldf)
            logger.info("Exportation terminée. Liens de téléchargement générés :")
        logger.info("Traitement terminé avec succès!")

    except Exception as e:
        logger.error(f"Erreur dans le programme : {e}")
        raise
        
if __name__ == "__main__":
    main()
