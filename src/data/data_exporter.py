import os
import platform
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataExporter:
    """
    Classe permettant d'exporter les datasets traités au format CSV.
    """

    def __init__(self, export_path="data/exports"):
        """
        Initialise le chemin d'export et crée le dossier s'il n'existe pas.
        """
        self.export_path = os.path.abspath(export_path)  # Chemin absolu
        os.makedirs(self.export_path, exist_ok=True)
        logger.info(f"Dossier d'export configuré : {self.export_path}")

    def export_to_csv(self, datasets):
        """
        Exporte les datasets traités en CSV et ouvre automatiquement le dossier.

        Args:
            datasets (dict): Dictionnaire contenant les datasets à exporter.
        """
        try:
            if not datasets:
                logger.warning("⚠️ Aucun dataset à exporter.")
                return
            
            for dataset_name, dataset in datasets.items():
                file_path = os.path.join(self.export_path, f"{dataset_name}.csv")
                dataset.to_csv(file_path, index=False)
                logger.info(f"Exportation réussie : {file_path}")

            # Ouvrir le dossier après l'export
            self.open_export_folder()

        except Exception as e:
            logger.error(f" Erreur lors de l'export des données : {e}")
            raise


