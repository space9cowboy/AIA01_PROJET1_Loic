from src.data.data_loader import DataLoader
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import numpy as np


logger = logging.getLogger(__name__)

class DataProcessings:
    def __init__(self, data):
        
        data = [data] if isinstance(data, str) else data
        if 'items_prop_1' in data or 'items_prop_2' in data:
            data = list(set(data + ['items_prop_1', 'items_prop_2']))
        self.data = data 
        self.data_loader = DataLoader(dataset_path=RAW_DATA_PATH, datasets=self.data)
        self.processed_path = os.path.join(PROCESSED_DATA_PATH)
        self.graphs_path = "exports/graphs"
        os.makedirs(self.graphs_path, exist_ok=True)
       
    
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
                        processed = process_func(raw_data)
                        if processed is not None:
                            self.save_processed_data(key, processed)
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
    
    def save_and_show_plot(self, filename):
        """Sauvegarde le graphique et l'affiche"""
        plt.savefig(os.path.join(self.graphs_path, filename))
        plt.close()

    def visualize_event_distribution(self, event_counts):
        """ Répartition des types d'événements"""
        plt.figure(figsize=(8, 5))
        sns.barplot(x=event_counts.index, y=event_counts.values, palette="pastel")
        plt.title("Distribution des Types d'Événements")
        plt.xlabel("Type d'événement")
        plt.ylabel("Nombre d'occurrences")
        self.save_and_show_plot("distribution_events.png")
        logger.info(" Distribution des événements analysée.")
    
    def unique_visitors_analysis(self, unique_visitors):
        """Nombre de visiteurs uniques"""
        logger.info(f"Nombre de visiteurs uniques: {unique_visitors}")
    
    def unique_items_analysis(self, unique_items):
        """Nombre d'items uniques"""
        logger.info(f"Nombre d'items uniques: {unique_items}")
    
    def visits_by_day_analysis(self, visits_per_day):
        """Nombre de visites par jour de la semaine"""
        plt.figure(figsize=(10, 5))
        sns.barplot(x=visits_per_day.index, y=visits_per_day.values, palette='coolwarm')
        plt.title("Répartition des Visites par Jour de la Semaine")
        plt.xlabel("Jour de la semaine")
        plt.ylabel("Nombre de visites")
        self.save_and_show_plot("visites_par_jour.png")
        logger.info("Répartition des visites par jour analysée.")
    
    def session_duration_analysis(self, merged):
        """⏳ Durée moyenne entre 'view' et 'transaction'"""
        merged['duration'] = merged['timestamp_trans'] - merged['timestamp_view']
        avg_duration = merged['duration'].mean()
        logger.info(f"Durée moyenne entre la première vue et la transaction: {avg_duration / 1000:.2f} secondes")
    
    def visualize_events_by_hour(self, df):
        """Répartition des événements par heure"""
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df["hour"].value_counts().index, y=df["hour"].value_counts().values, palette="coolwarm")
        plt.title("Distribution des Événements par Heure")
        plt.xlabel("Heure de la journée")
        plt.ylabel("Nombre d'événements")
        self.save_and_show_plot("events_by_hour.png")
        logger.info("Répartition des événements par heure analysée.")

    def visualize_transactions_by_day(self, df):
        """Transactions par jour"""
        plt.figure(figsize=(10, 5))
        sns.barplot(x=df["day_of_week"].value_counts().index, y=df[df["event"] == "transaction"]["day_of_week"].value_counts(), palette="viridis")
        plt.title("Transactions par Jour de la Semaine")
        plt.xlabel("Jour de la semaine")
        plt.ylabel("Nombre de transactions")
        self.save_and_show_plot("transactions_by_day.png")
        logger.info("Transactions par jour analysées.")

    def visualize_sessions_per_user(self, df):
        """Nombre de sessions par utilisateur"""
        plt.figure(figsize=(10, 5))
        sns.histplot(df["session_id"].value_counts(), bins=50, kde=True, color="skyblue")
        plt.title("Distribution des Sessions par Visiteur")
        plt.xlabel("Nombre de sessions par visiteur")
        plt.ylabel("Nombre de visiteurs")
        self.save_and_show_plot("sessions_per_user.png")
        logger.info("Distribution des sessions analysée.")

    def visualize_time_between_events(self, df):
        """Temps entre événements"""
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x="event", y="time_since_last_event", palette="muted")
        plt.yscale("log")
        plt.title("Temps Moyen entre Événements")
        plt.xlabel("Type d'événement")
        plt.ylabel("Temps entre événements (log scale)")
        self.save_and_show_plot("time_between_events.png")
        logger.info("Temps entre événements analysé.")

    def visualize_top_categories(self, df):
        """Catégories les plus populaires"""
        plt.figure(figsize=(12, 5))
        sns.barplot(x=df["categoryid"].value_counts().head(10).index, y=df["categoryid"].value_counts().head(10).values, palette="magma")
        plt.title("Top 10 des Catégories les Plus Populaires")
        plt.xlabel("Catégorie")
        plt.ylabel("Nombre d'événements")
        self.save_and_show_plot("top_categories.png")
        logger.info(" Catégories les plus populaires analysées.")

    def visualize_top_items(self, df):
        """ Produits les plus vus"""
        plt.figure(figsize=(12, 5))
        sns.barplot(x=df["itemid"].value_counts().head(10).index, y=df["itemid"].value_counts().head(10).values, palette="Blues_r")
        plt.title("Top 10 des Produits les Plus Vus")
        plt.xlabel("ID de l'item")
        plt.ylabel("Nombre de vues")
        self.save_and_show_plot("top_items.png")
        logger.info("Produits les plus vus analysés.")
    
    def visualize_active_visitors(self, df):
        """📊 Répartition des visiteurs selon le nombre de visites"""
        plt.figure(figsize=(10, 5))
        sns.histplot(df["visitorid"].value_counts(), bins=50, kde=True, color="teal")
        plt.title("Répartition des Visiteurs selon le Nombre de Visites")
        plt.xlabel("Nombre de Visites")
        plt.ylabel("Nombre de Visiteurs")
        self.save_and_show_plot("visiteurs_actifs.png")
        logger.info("Répartition des visiteurs actifs analysée.")

    def visualize_top_cart_items(self, df):
        """Articles les plus ajoutés au panier"""
        top_cart_items = df[df["event"] == "addtocart"]["itemid"].value_counts().head(10)
        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_cart_items.index, y=top_cart_items.values, palette="coolwarm")
        plt.title("Top 10 des Articles les Plus Ajoutés au Panier")
        plt.xlabel("ID de l'article")
        plt.ylabel("Nombre d'ajouts au panier")
        self.save_and_show_plot("top_cart_items.png")
        logger.info("Articles les plus ajoutés au panier analysés.")
            
    def visualize_cart_abandonment(self, df):
        """Taux d'abandon du panier"""
        total_adds = df[df["event"] == "addtocart"]["visitorid"].nunique()
        total_transactions = df[df["event"] == "transaction"]["visitorid"].nunique()
        abandonment_rate = ((total_adds - total_transactions) / total_adds) * 100

        plt.figure(figsize=(6, 6))
        plt.pie([abandonment_rate, 100 - abandonment_rate], labels=["Abandon", "Transaction"], autopct='%1.1f%%', colors=["red", "green"])
        plt.title("Taux d'Abandon du Panier")
        self.save_and_show_plot("cart_abandonment.png")
        logger.info(f"Taux d'abandon du panier: {abandonment_rate:.2f}%")
    
    def analyze_time_to_purchase(self, df):
        """Temps moyen avant une transaction"""
        transactions = df[df["event"] == "transaction"]
        views = df[df["event"] == "view"]
        merged = transactions.merge(views, on='visitorid', suffixes=('_trans', '_view'))
        merged['duration'] = merged['timestamp_trans'] - merged['timestamp_view']
        avg_duration = merged['duration'].mean() / 1000  # Conversion en secondes
        logger.info(f"Temps moyen avant une transaction: {avg_duration:.2f} secondes")
    
    def visualize_retention_rate(self, df):
        """Taux de rétention des visiteurs"""
        first_visit = df.groupby("visitorid")["timestamp"].min()
        retention_days = (df["timestamp"].max() - first_visit) / (1000 * 60 * 60 * 24)
        
        plt.figure(figsize=(10, 5))
        sns.histplot(retention_days, bins=30, kde=True, color="purple")
        plt.title("Taux de Rétention des Visiteurs (en jours)")
        plt.xlabel("Jours depuis la première visite")
        plt.ylabel("Nombre de visiteurs")
        self.save_and_show_plot("retention_rate.png")
        logger.info("Taux de rétention analysé.")
    
    def visualize_funnel_conversion(self, df):
        """Courbe de conversion par étape"""
        steps = {
            "Views": df[df["event"] == "view"]["visitorid"].nunique(),
            "Add to Cart": df[df["event"] == "addtocart"]["visitorid"].nunique(),
            "Transaction": df[df["event"] == "transaction"]["visitorid"].nunique(),
        }

        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(steps.keys()), y=list(steps.values()), palette="Blues_r")
        plt.title("Conversion des Visiteurs par Étape")
        plt.xlabel("Étape du Tunnel")
        plt.ylabel("Nombre de Visiteurs")
        self.save_and_show_plot("funnel_conversion.png")
        logger.info("🔄 Courbe de conversion analysée.")
    

    
    def ab_test_add_to_cart_button(self, events_df):
        """ Test A/B : Position du bouton 'Ajouter au panier' """
        logger.info("🛒 Test A/B - Position du bouton 'Ajouter au panier'")

        # Simulation : Création aléatoire de deux groupes (A: Haut, B: Bas)
        np.random.seed(30000)
        events_df["button_position"] = np.random.choice(["haut", "bas"], size=len(events_df))

        # Calcul du taux d'ajout au panier par groupe
        conversion_rates = events_df[events_df["event"] == "addtocart"].groupby("button_position")["visitorid"].count()

        # Visualisation
        plt.figure(figsize=(8, 5))
        sns.barplot(x=conversion_rates.index, y=conversion_rates.values, palette="coolwarm")
        plt.title("📊 Ajouts au Panier par Position du Bouton")
        plt.ylabel("Nombre d'ajouts au panier")
        plt.xlabel("Position du bouton")
        self.save_and_show_plot("ab_test_add_to_cart_button.png")

  
    def ab_test_add_to_cart(self, df):
        """ A/B Test sur l'effet du 'add_to_cart' sur les transactions """

        users_with_addtocart = df[df["event"] == "addtocart"]["visitorid"].unique()
        users_with_transaction = df[df["event"] == "transaction"]["visitorid"].unique()

        group_A = len(set(users_with_transaction) - set(users_with_addtocart))  # Achat sans ajout au panier
        group_B = len(set(users_with_transaction) & set(users_with_addtocart))  # Achat après ajout au panier

        total_A = len(set(df["visitorid"]) - set(users_with_addtocart))  # Visiteurs n'ayant jamais ajouté au panier
        total_B = len(set(users_with_addtocart))  # Visiteurs ayant ajouté au panier

        conversion_A = (group_A / total_A) * 100
        conversion_B = (group_B / total_B) * 100

        # Visualisation
        plt.figure(figsize=(8, 5))
        sns.barplot(x=["Sans add_to_cart", "Avec add_to_cart"], y=[conversion_A, conversion_B], palette="coolwarm")
        plt.ylabel("Taux de conversion (%)")
        plt.title("Impact de l'ajout au panier sur le taux de conversion")
        self.save_and_show_plot("ab_test_add_to_cart.png")
        logger.info("📊 A/B Test Add to Cart réalisé.")

    # ================== A/B TEST 2 : Impact de l'heure sur la conversion ==================
    def ab_test_peak_hours(self, df):
        """ A/B Test sur les heures de la journée """

        transactions = df[df["event"] == "transaction"]
        visitors = df.groupby("visitorid")["hour"].min().reset_index()

        morning_visits = visitors[visitors["hour"] < 18]["visitorid"].count()
        evening_visits = visitors[visitors["hour"] >= 18]["visitorid"].count()

        morning_transactions = transactions[transactions["hour"] < 18]["visitorid"].nunique()
        evening_transactions = transactions[transactions["hour"] >= 18]["visitorid"].nunique()

        conversion_morning = (morning_transactions / morning_visits) * 100
        conversion_evening = (evening_transactions / evening_visits) * 100

        # Visualisation
        plt.figure(figsize=(8, 5))
        sns.barplot(x=["Matin (0h-17h)", "Soir (18h-23h)"], y=[conversion_morning, conversion_evening], palette="viridis")
        plt.ylabel("Taux de conversion (%)")
        plt.title("Comparaison du taux de conversion matin vs soir")
        self.save_and_show_plot("ab_test_peak_hours.png")
        logger.info("📊 A/B Test Peak Hours réalisé.")

    # ================== A/B TEST : Impact des réductions ==================
    def ab_test_discount_effect(self, df, items_df):
        """ A/B Test sur l'effet des promotions sur la conversion """

        # Vérifier que la colonne `value` existe bien et créer une colonne 'discount'
        if "value" in items_df.columns:
            items_df["discount"] = items_df["value"].astype(str).apply(lambda x: 1 if "%" in x else 0)
        else:
            logger.warning("⚠️ Colonne 'value' absente des données items_prop, impossible d'effectuer le test")
            return

        # Fusion des données `events` avec `items`
        df = df.merge(items_df[["itemid", "discount"]], on="itemid", how="left")

        # Calcul du taux de conversion
        transactions = df[df["event"] == "transaction"].groupby("discount")["visitorid"].nunique()
        total_visitors = df.groupby("discount")["visitorid"].nunique()

        # Vérification des valeurs
        if len(transactions) != len(total_visitors):
            logger.error("⚠️ Problème de dimension entre transactions et visiteurs")
            return

        conversion_rates = (transactions / total_visitors) * 100

        # Vérifier la correspondance des longueurs avant de tracer le graphique
        if len(conversion_rates) != 2:
            logger.error(f"⚠️ Nombre de catégories inattendu pour discount: {len(conversion_rates)}")
            return

        # Visualisation
        plt.figure(figsize=(8, 5))
        sns.barplot(x=["Sans réduction", "Avec réduction"], y=conversion_rates.values, palette="coolwarm")
        plt.ylabel("Taux de conversion (%)")
        plt.title("Impact des réductions sur le taux de conversion")
        self.save_and_show_plot("ab_test_discount_effect.png")
        logger.info("A/B Test Discount Effect réalisé.")


    def visualize_activity_per_day(self, day_counts):
        """Activité par jour de la semaine"""
        plt.figure(figsize=(10, 5))
        sns.barplot(x=day_counts.index, y=day_counts.values, palette="muted")
        plt.title("Activité des utilisateurs par jour de la semaine")
        plt.xlabel("Jour de la semaine")
        plt.ylabel("Nombre d'événements")
        self.save_and_show_plot("activity_per_day.png")

    def visualize_conversion_rate(self, conversion_rate):
        """Taux de conversion"""
        plt.figure(figsize=(5, 5))
        plt.pie([conversion_rate, 100 - conversion_rate], labels=["Transactions", "Non-transactions"], autopct="%1.1f%%", colors=["green", "red"])
        plt.title("Taux de conversion global")
        self.save_and_show_plot("conversion_rate.png")

    def visualize_top_properties(self, prop_counts):
        """Propriétés les plus fréquentes"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x=prop_counts.head(10).index, y=prop_counts.head(10).values, palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Top 10 des Propriétés les Plus Fréquentes")
        plt.xlabel("Propriété")
        plt.ylabel("Nombre d'occurrences")
        self.save_and_show_plot("top_properties.png")

    def visualize_category_structure(self, category_data):
        """Distribution des niveaux des catégories"""
        plt.figure(figsize=(10, 5))
        sns.histplot(category_data["level"], bins=10, kde=True, color="purple")
        plt.title("Distribution des niveaux des catégories")
        plt.xlabel("Niveau de la catégorie")
        plt.ylabel("Nombre de catégories")
        self.save_and_show_plot("category_levels.png")

    def compress_graphs(self):
        """Archive tous les graphiques en un fichier ZIP"""
        zip_path = "exports/graphs.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(self.graphs_path):
                zipf.write(os.path.join(self.graphs_path, file), file)
        print(f"Graphiques compressés dans {zip_path}")
    
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

            # Visualisation des propriétés
            # self.visualize_property_distribution(data['property'].value_counts())

        # Nombre d'items uniques
        n_items = data['itemid'].nunique()
        logger.info(f"\n{'='*30}")
        logger.info("Statistiques générales:")
        logger.info(f"{'='*30}\n")
        logger.info(f"Nombre d'items uniques: {n_items:,}")
        logger.info(f"Nombre total d'enregistrements: {len(data):,}")
        logger.info(f"Moyenne de propriétés par item: {len(data)/n_items:.2f}")

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
    
