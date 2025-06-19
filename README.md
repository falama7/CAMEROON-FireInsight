🔥CM-FI_CAMEROON FireInsight

Description
Cette application interactive, développée avec Streamlit, offre une solution complète pour l'analyse et la prévision de l'activité des feux au Cameroun. Elle permet aux utilisateurs de charger des données de détection de feux VIIRS, de les filtrer et de visualiser des tendances spatiales, temporelles et prédictives pour une meilleure gestion des risques d'incendie.

Table des matières
Description
Fonctionnalités
Technologies Utilisées
Installation
Utilisation
Source des Données
Capture d'Écran (Optionnel)
Contribuer
License
Contact
Fonctionnalités
L'application offre les fonctionnalités clés suivantes :

Chargement de Données Multiples : Importez facilement plusieurs fichiers CSV de détections de feux VIIRS (un par année) pour une analyse sur de longues périodes.
Filtrage Avancé : Filtrez les données par niveau de confiance des détections (faible, nominal, élevé), par période (jour, nuit, jour et nuit), et par plage de dates.
Analyse Spatiale des Hotspots :
Visualisation des zones à forte pression de feu via des cartes de chaleur interactives (Folium).
Identification et classement des 10 principales zones à risque, avec le nombre de feux et la puissance radiative moyenne (FRP).
Analyse Temporelle :
Graphiques de séries temporelles montrant l'évolution du nombre de feux par heure, jour, semaine ou mois.
Identification des pics d'activité.
Comparaison Inter-Annuelle : Comparez l'activité des feux (nombre total, tendances mensuelles, intensité FRP moyenne) entre deux années sélectionnées.
Prévisions d'Activité des Feux :
Utilise des modèles de régression linéaire et ARIMA pour prévoir l'activité future des feux sur une période configurable (journalière, hebdomadaire, mensuelle).
Affichage des insights clés dérivés des prévisions.
Génération d'une carte animée interactive des prévisions.
Option de télécharger une animation GIF des prévisions pour un partage facile.
Rapport Détaillé : Générez un rapport HTML complet, téléchargeable, incluant :
Statistiques générales et un résumé exécutif.
Analyse détaillée des hotspots et de la distribution géographique par région.
Analyse saisonnière et horaire.
Récapitulatif des prévisions et des tendances.
Recommandations spécifiques basées sur l'analyse des données réelles pour la prévention, l'intervention, la conservation et la sensibilisation.
Visualisations clés intégrées (graphiques temporels, comparaisons, prévisions).
Technologies Utilisées
Python
Streamlit : Pour la construction de l'interface utilisateur web interactive.
Pandas : Pour la manipulation et l'analyse des données.
NumPy : Pour les opérations numériques.
Plotly Express & Plotly Graph Objects : Pour la création de visualisations interactives.
Folium & Streamlit-Folium : Pour la création et l'affichage des cartes géospatiales.
statsmodels : Pour la modélisation de séries temporelles (ARIMA, SARIMAX).
scikit-learn : Pour la régression linéaire.
imageio & PIL (Pillow) : Pour la génération d'animations GIF.
kaleido : Backend nécessaire pour l'exportation des figures Plotly en images statiques (utilisé dans le rapport).
Installation
Suivez ces étapes pour configurer et exécuter l'application sur votre machine locale :

Cloner le dépôt :

Bash

git clone https://github.com/[VotreNomUtilisateur]/[NomDeVotreDépôt].git
cd [NomDeVotreDépôt]
Créer un environnement virtuel (recommandé) :

Bash

python -m venv venv
Activer l'environnement virtuel :

Sur macOS / Linux :
Bash

source venv/bin/activate
Sur Windows :
Bash

.\venv\Scripts\activate
Installer les dépendances :
Créez un fichier requirements.txt à la racine de votre projet avec le contenu suivant :

streamlit
pandas
numpy
plotly
folium
streamlit-folium
statsmodels
scikit-learn
imageio
Pillow
kaleido
Puis installez-les :

Bash

pip install -r requirements.txt
Exécuter l'application Streamlit :

Bash

streamlit run app4.py
L'application s'ouvrira automatiquement dans votre navigateur web à l'adresse http://localhost:8501.

Utilisation
Chargement des données : Dans la barre latérale gauche, utilisez le bouton "Téléchargez vos fichiers CSV VIIRS" pour importer vos données. L'application est conçue pour gérer plusieurs fichiers CSV, idéalement un par année.
Application des filtres : Utilisez les options de la barre latérale pour ajuster le niveau de confiance minimum et la période (jour/nuit) des détections à analyser.
Navigation par onglets :
Zones à forte pression de feu : Visualisez les hotspots sur une carte interactive et consultez le tableau des zones les plus à risque.
Série temporelle : Explorez l'évolution des feux sur différentes fréquences (horaire, journalière, hebdomadaire, mensuelle).
Comparaison entre années : Comparez l'activité des feux entre deux années de votre choix.
Prévisions : Obtenez des prévisions sur l'activité future des feux et visualisez-les sur des graphiques et une carte animée.
Rapport détaillé : Générez et téléchargez un rapport HTML complet avec toutes les analyses et recommandations.
Source des Données
Les données de détection de feux actives (Active Fire Data) utilisées pour cette application proviennent du système VIIRS (Visible Infrared Imaging Radiometer Suite) de la NASA FIRMS (Fire Information for Resource Management System).

Vous pouvez télécharger les données brutes à partir de leur portail : NASA FIRMS Download

Le format de fichier CSV attendu doit contenir a minima les colonnes suivantes : latitude, longitude, acq_date, acq_time, confidence, daynight. La colonne frp (Fire Radiative Power) est optionnelle mais recommandée pour des analyses d'intensité.

Capture d'Écran (Optionnel)
Incluez ici une capture d'écran attrayante de votre application en action. C'est souvent la première chose que les visiteurs regardent !

Contribuer
Les contributions sont les bienvenues ! Si vous souhaitez améliorer cette application, n'hésitez pas à :

Forker le dépôt.
Créer une nouvelle branche (git checkout -b feature/AmazingFeature).
Faire vos modifications et les commiter (git commit -m 'Add some AmazingFeature').
Pusher vers la branche (git push origin feature/AmazingFeature).
Ouvrir une Pull Request.
License
Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

Contact
[Votre Nom/Prénom] - [Votre Email ou Lien LinkedIn/Twitter]

Lien du projet : [lien suspect supprimé][VotreNomUtilisateur]/[NomDeVotreDépôt]
