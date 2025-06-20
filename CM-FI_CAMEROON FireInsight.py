import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import folium
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
from streamlit_folium import folium_static
import os
import calendar
import io
import base64
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import imageio
from PIL import Image
import tempfile

# Configuration de la page
st.set_page_config(
    page_title="CM-FI",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonctions utilitaires
@st.cache_data
def load_data(file_path):
    """Charger et prétraiter les données VIIRS"""
    try:
        df = pd.read_csv(file_path)
        
        # Convertir les dates et heures
        df['date_heure'] = pd.to_datetime(
            df['acq_date'] + ' ' + df['acq_time'].astype(str).str.zfill(4),
            format='%Y-%m-%d %H%M'
        )
        
        # Extraire des colonnes utiles pour l'analyse
        df['annee'] = df['date_heure'].dt.year
        df['mois'] = df['date_heure'].dt.month
        df['jour'] = df['date_heure'].dt.day
        df['heure'] = df['date_heure'].dt.hour
        df['nom_mois'] = df['date_heure'].dt.month.apply(lambda x: calendar.month_name[x])
        
        # Calculer une colonne de confiance numérique
        confidence_map = {'l': 1, 'n': 2, 'h': 3}
        df['confidence_num'] = df['confidence'].map(confidence_map)
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

def filter_data(df, confidence_min=None, date_range=None, day_night=None):
    """Filtrer les données selon plusieurs critères"""
    filtered_df = df.copy()
    
    # Filtre par niveau de confiance
    if confidence_min:
        if confidence_min == 'l':
            pass  # Pas de filtre
        elif confidence_min == 'n':
            filtered_df = filtered_df[filtered_df['confidence'].isin(['n', 'h'])]
        elif confidence_min == 'h':
            filtered_df = filtered_df[filtered_df['confidence'] == 'h']
    
    # Filtre par plage de dates
    if date_range:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date_heure'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date_heure'] <= pd.to_datetime(end_date))
        ]
    
    # Filtre jour/nuit
    if day_night and day_night != 'both':
        filtered_df = filtered_df[filtered_df['daynight'] == day_night]
    
    return filtered_df

def create_heatmap(df, zoom_start=6):
    """Créer une carte de chaleur des feux"""
    # Centrer la carte sur le Cameroun
    center_lat = df['latitude'].mean() if not df.empty else 7.3697
    center_lon = df['longitude'].mean() if not df.empty else 12.3547
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, 
                   tiles="CartoDB positron")
    
    # Ajouter des tuiles supplémentaires
    folium.TileLayer('CartoDB dark_matter', name='Dark Map', attr='CartoDB').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Terrain', attr='Stamen Design | OpenStreetMap').add_to(m)
    
    # Préparer les données pour la carte de chaleur
    if not df.empty:
        # Utiliser FRP comme intensité du feu (normalisé)
        if 'frp' in df.columns:
            heat_data = [[row['latitude'], row['longitude'], min(1, row['frp']/100)] 
                         for _, row in df.iterrows() if pd.notna(row['frp'])]
        else:
            heat_data = [[row['latitude'], row['longitude']] 
                         for _, row in df.iterrows()]
        
        # Ajouter la carte de chaleur
        HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(m)
    
    # Ajouter le contrôle de couches
    folium.LayerControl().add_to(m)
    
    return m

def create_hotspot_map(df, grid_size=0.1):
    """Identifier et visualiser les zones à forte pression de feu"""
    if df.empty:
        return None, pd.DataFrame()
    
    # Créer une grille pour agréger les feux
    df['lat_bin'] = np.floor(df['latitude'] / grid_size) * grid_size
    df['lon_bin'] = np.floor(df['longitude'] / grid_size) * grid_size
    
    # Compter le nombre de feux par cellule de la grille
    hotspots = df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
    
    # Calculer la moyenne de FRP par cellule si disponible
    if 'frp' in df.columns:
        frp_by_cell = df.groupby(['lat_bin', 'lon_bin'])['frp'].mean().reset_index(name='frp_mean')
        hotspots = hotspots.merge(frp_by_cell, on=['lat_bin', 'lon_bin'])
    
    # Trier par nombre de feux décroissant
    hotspots = hotspots.sort_values('count', ascending=False)
    
    # Créer la carte
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    
    # Ajouter des tuiles
    folium.TileLayer('CartoDB positron', name='Light Map', attr='CartoDB').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map', attr='CartoDB').add_to(m)
    
    # Créer un cluster de marqueurs
    marker_cluster = MarkerCluster().add_to(m)
    
    # Ajouter les hotspots à la carte
    for idx, row in hotspots.head(100).iterrows():  # Limiter aux 100 premiers hotspots
        radius = np.sqrt(row['count']) * 500  # Taille proportionnelle à la racine du nombre de feux
        
        # Couleur basée sur le nombre de feux
        if row['count'] > hotspots['count'].quantile(0.9):
            color = 'red'
            fill_color = '#ff4444'
        elif row['count'] > hotspots['count'].quantile(0.7):
            color = 'orange'
            fill_color = '#ffaa44'
        else:
            color = 'blue'
            fill_color = '#4285F4'
        
        # Créer le popup avec les informations
        if 'frp_mean' in row:
            popup_text = f"Nombre de feux: {row['count']}<br>FRP moyenne: {row['frp_mean']:.2f}"
        else:
            popup_text = f"Nombre de feux: {row['count']}"
        
        folium.Circle(
            location=[row['lat_bin'] + grid_size/2, row['lon_bin'] + grid_size/2],
            radius=radius,
            popup=popup_text,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.6
        ).add_to(m)
    
    # Ajouter le contrôle de couches
    folium.LayerControl().add_to(m)
    
    return m, hotspots

def create_time_series(df, frequency='M'):
    """Créer une série temporelle des feux"""
    if df.empty:
        return None
    
    # Agréger par période
    if frequency == 'D':
        df_time = df.groupby(pd.Grouper(key='date_heure', freq='D')).size().reset_index(name='count')
        title = "Nombre de feux par jour"
    elif frequency == 'W':
        df_time = df.groupby(pd.Grouper(key='date_heure', freq='W')).size().reset_index(name='count')
        title = "Nombre de feux par semaine"
    elif frequency == 'M':
        df_time = df.groupby(pd.Grouper(key='date_heure', freq='M')).size().reset_index(name='count')
        title = "Nombre de feux par mois"
    elif frequency == 'H':
        # Groupe par heure de la journée
        df_time = df.groupby('heure').size().reset_index(name='count')
        
        # Créer un graphique en barres pour les heures
        fig = px.bar(df_time, x='heure', y='count', 
                     title="Distribution des feux par heure de la journée",
                     labels={'heure': 'Heure de la journée', 'count': 'Nombre de feux'})
        return fig
    
    # Créer un graphique linéaire pour les séries temporelles
    fig = px.line(df_time, x='date_heure', y='count', 
                  title=title,
                  labels={'date_heure': 'Date', 'count': 'Nombre de feux'})
    
    # Ajouter des repères pour les débuts de feux fréquents
    if frequency != 'H':
        # Identifier les pics (débuts potentiels de feux importants)
        threshold = df_time['count'].mean() + df_time['count'].std()
        peaks = df_time[df_time['count'] > threshold]
        
        for idx, row in peaks.iterrows():
            fig.add_annotation(
                x=row['date_heure'],
                y=row['count'],
                text="Pic",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
    
    return fig

def compare_years(df, year1, year2):
    """Comparer les feux entre deux années différentes"""
    if df.empty or year1 not in df['annee'].unique() or year2 not in df['annee'].unique():
        return None, None, None
    
    # Filtrer les données pour chaque année
    df_year1 = df[df['annee'] == year1]
    df_year2 = df[df['annee'] == year2]
    
    # 1. Comparaison du nombre total de feux
    total_fires = pd.DataFrame({
        'Année': [year1, year2],
        'Nombre de feux': [len(df_year1), len(df_year2)]
    })
    
    fig_total = px.bar(total_fires, x='Année', y='Nombre de feux',
                       title=f"Comparaison du nombre total de feux: {year1} vs {year2}",
                       color='Année')
    
    # 2. Comparaison mensuelle
    monthly_year1 = df_year1.groupby('nom_mois').size().reset_index(name='count')
    monthly_year1['Année'] = str(year1)
    
    monthly_year2 = df_year2.groupby('nom_mois').size().reset_index(name='count')
    monthly_year2['Année'] = str(year2)
    
    monthly_combined = pd.concat([monthly_year1, monthly_year2])
    
    # Ordonner les mois correctement
    month_order = {month: i for i, month in enumerate(calendar.month_name[1:])}
    monthly_combined['month_num'] = monthly_combined['nom_mois'].map(month_order)
    monthly_combined = monthly_combined.sort_values('month_num')
    
    fig_monthly = px.line(monthly_combined, x='nom_mois', y='count', color='Année',
                          title=f"Comparaison mensuelle des feux: {year1} vs {year2}",
                          labels={'nom_mois': 'Mois', 'count': 'Nombre de feux'},
                          markers=True)
    
    # 3. Comparaison de l'intensité (FRP) si disponible
    if 'frp' in df.columns:
        frp_year1 = df_year1['frp'].mean()
        frp_year2 = df_year2['frp'].mean()
        
        frp_comparison = pd.DataFrame({
            'Année': [year1, year2],
            'FRP moyenne': [frp_year1, frp_year2]
        })
        
        fig_frp = px.bar(frp_comparison, x='Année', y='FRP moyenne',
                         title=f"Comparaison de l'intensité moyenne des feux (FRP): {year1} vs {year2}",
                         color='Année')
    else:
        fig_frp = None
    
    return fig_total, fig_monthly, fig_frp

def create_animated_forecast_map(df, forecast_data, period=6, grid_size=0.1, freq='M'):
    """Créer une carte animée des prévisions de feux
    
    Args:
        df: DataFrame avec les données historiques
        forecast_data: DataFrame avec les prévisions
        period: Nombre de périodes de prévision
        grid_size: Taille de la grille pour l'agrégation
        freq: Fréquence des prévisions ('D', 'W', 'M')
        
    Returns:
        m: Carte folium avec animation temporelle
    """
    if df.empty or forecast_data is None:
        return None
    
    # Convertir les prévisions globales en prévisions spatiales
    # Nous allons distribuer les prévisions selon la distribution spatiale historique
    
    # 1. Créer une grille spatiale des feux historiques
    df['lat_bin'] = np.floor(df['latitude'] / grid_size) * grid_size
    df['lon_bin'] = np.floor(df['longitude'] / grid_size) * grid_size
    
    # Compter les feux par cellule et calculer la distribution en pourcentage
    grid_counts = df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
    grid_counts['pct'] = grid_counts['count'] / grid_counts['count'].sum()
    
    # 2. Créer des prévisions spatialisées pour chaque période
    last_date = df['date_heure'].max()
    
    # Préparer les données pour l'animation
    all_frames = []
    
    # Ajouter les données historiques (nous utilisons les 3 dernières périodes)
    if freq == 'D':
        hist_period = pd.Timedelta(days=3)
        freq_delta = pd.Timedelta(days=1)
    elif freq == 'W':
        hist_period = pd.Timedelta(weeks=3)
        freq_delta = pd.Timedelta(weeks=1)
    else:  # 'M'
        hist_period = pd.Timedelta(days=90)  # ~3 mois
        freq_delta = pd.Timedelta(days=30)  # ~1 mois
    
    # Filtrer les données historiques récentes
    recent_df = df[df['date_heure'] > (last_date - hist_period)]
    
    # Agréger par période et par cellule
    if freq == 'D':
        recent_grouped = recent_df.groupby(['lat_bin', 'lon_bin', pd.Grouper(key='date_heure', freq='D')])
    elif freq == 'W':
        recent_grouped = recent_df.groupby(['lat_bin', 'lon_bin', pd.Grouper(key='date_heure', freq='W')])
    else:  # 'M'
        recent_grouped = recent_df.groupby(['lat_bin', 'lon_bin', pd.Grouper(key='date_heure', freq='M')])
    
    recent_counts = recent_grouped.size().reset_index(name='count')
    
    # Ajouter les données historiques à all_frames
    for date, group in recent_counts.groupby('date_heure'):
        frame_data = []
        for _, row in group.iterrows():
            frame_data.append([row['lat_bin'] + grid_size/2, row['lon_bin'] + grid_size/2, min(1.0, row['count']/10)])
        
        all_frames.append({
            'data': frame_data,
            'date': date.strftime('%Y-%m-%d')
        })
    
    # Ajouter les prévisions
    for i, row in forecast_data.iterrows():
        date = row['date']
        predicted_fires = row['prévision']
        
        # Distribuer les feux prévus selon la distribution spatiale historique
        frame_data = []
        for _, grid_cell in grid_counts.iterrows():
            # Calculer le nombre de feux prévu pour cette cellule
            cell_fires = predicted_fires * grid_cell['pct']
            if cell_fires > 0:
                frame_data.append([
                    grid_cell['lat_bin'] + grid_size/2, 
                    grid_cell['lon_bin'] + grid_size/2, 
                    min(1.0, cell_fires/10)  # Normaliser l'intensité
                ])
        
        all_frames.append({
            'data': frame_data,
            'date': date.strftime('%Y-%m-%d') + ' (prévision)'
        })
    
    # Créer la carte de base
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")
    
    # Ajouter des tuiles supplémentaires
    folium.TileLayer('CartoDB dark_matter', name='Dark Map', attr='CartoDB').add_to(m)
    folium.TileLayer('Stamen Terrain', name='Terrain', attr='Stamen Design | OpenStreetMap').add_to(m)
    
    # Créer un plugin HeatMapWithTime
    from folium.plugins import HeatMapWithTime
    
    # Préparer les données pour HeatMapWithTime
    heat_data = [frame['data'] for frame in all_frames]
    heat_timestamps = [frame['date'] for frame in all_frames]
    
    # Ajouter le plugin HeatMapWithTime à la carte
    HeatMapWithTime(
        heat_data,
        index=heat_timestamps,
        auto_play=True,
        max_opacity=0.8,
        radius=10,
        use_local_extrema=True
    ).add_to(m)
    
    # Ajouter une légende
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; padding: 10px;
                border-radius: 5px;">
        <span style="font-weight: bold;">Légende</span><br>
        <i class="fa fa-circle" style="color:green;"></i> Intensité faible<br>
        <i class="fa fa-circle" style="color:orange;"></i> Intensité moyenne<br>
        <i class="fa fa-circle" style="color:red;"></i> Intensité élevée<br>
        <hr style="margin: 5px 0;">
        <span style="font-style: italic;">Les dates avec "(prévision)" sont des projections basées sur les tendances historiques</span>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Ajouter le contrôle de couches
    folium.LayerControl().add_to(m)
    
    return m

def create_forecast_animation_gif(df, forecast_data, grid_size=0.1, freq='M', period=6, dpi=100):
    """Créer une animation GIF des prévisions de feux
    
    Args:
        df: DataFrame avec les données historiques
        forecast_data: DataFrame avec les prévisions
        grid_size: Taille de la grille pour l'agrégation
        freq: Fréquence des prévisions ('D', 'W', 'M')
        period: Nombre de périodes de prévision
        dpi: Résolution des images
        
    Returns:
        gif_path: Chemin vers le fichier GIF généré
    """
    if df.empty or forecast_data is None:
        return None
    
    # Créer un dossier temporaire pour stocker les images
    temp_dir = tempfile.mkdtemp()
    
    # 1. Créer une grille spatiale des feux historiques
    df['lat_bin'] = np.floor(df['latitude'] / grid_size) * grid_size
    df['lon_bin'] = np.floor(df['longitude'] / grid_size) * grid_size
    
    # Compter les feux par cellule et calculer la distribution en pourcentage
    grid_counts = df.groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
    grid_counts['pct'] = grid_counts['count'] / grid_counts['count'].sum()
    
    # 2. Préparer les données historiques récentes et les prévisions
    last_date = df['date_heure'].max()
    
    # Définir la période historique et le delta de fréquence
    if freq == 'D':
        hist_period = pd.Timedelta(days=3)
        freq_delta = pd.Timedelta(days=1)
        time_unit = "jour"
    elif freq == 'W':
        hist_period = pd.Timedelta(weeks=3)
        freq_delta = pd.Timedelta(weeks=1)
        time_unit = "semaine"
    else:  # 'M'
        hist_period = pd.Timedelta(days=90)  # ~3 mois
        freq_delta = pd.Timedelta(days=30)  # ~1 mois
        time_unit = "mois"
    
    # Filtrer les données historiques récentes
    recent_df = df[df['date_heure'] > (last_date - hist_period)]
    
    # Agréger par période et par cellule
    if freq == 'D':
        recent_grouped = recent_df.groupby(['lat_bin', 'lon_bin', pd.Grouper(key='date_heure', freq='D')])
    elif freq == 'W':
        recent_grouped = recent_df.groupby(['lat_bin', 'lon_bin', pd.Grouper(key='date_heure', freq='W')])
    else:  # 'M'
        recent_grouped = recent_df.groupby(['lat_bin', 'lon_bin', pd.Grouper(key='date_heure', freq='M')])
    
    recent_counts = recent_grouped.size().reset_index(name='count')
    
    # 3. Générer une image pour chaque période
    all_frames = []
    frame_paths = []
    
    # Déterminer les limites de la carte
    min_lat = df['latitude'].min() - 1
    max_lat = df['latitude'].max() + 1
    min_lon = df['longitude'].min() - 1
    max_lon = df['longitude'].max() + 1
    
    # Créer un colormap pour les intensités de feu
    cmap = plt.cm.get_cmap('YlOrRd')
    
    # Ajouter les données historiques
    for date, group in recent_counts.groupby('date_heure'):
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        
        # Tracer les frontières du Cameroun (simplifiées)
        # Note: Dans une version réelle, on utiliserait un shapefile des frontières
        cameroon_lat = [2, 4, 6, 8, 10, 12, 13, 12, 10, 8, 6, 4, 2]
        cameroon_lon = [9, 8, 9, 10, 13, 15, 14, 15, 15, 14, 13, 11, 9]
        ax.plot(cameroon_lon, cameroon_lat, 'k-', linewidth=1, alpha=0.5)
        
        # Définir les limites de la carte
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        
        # Tracer les points de feu
        intensities = []
        for _, row in group.iterrows():
            intensity = min(1.0, row['count']/10)
            intensities.append(intensity)
            ax.scatter(row['lon_bin'] + grid_size/2, row['lat_bin'] + grid_size/2, 
                      s=intensity*100, c=[cmap(intensity)], alpha=0.7)
        
        # Ajouter un titre et des informations
        ax.set_title(f"Activité des feux au Cameroun\n{date.strftime('%d %B %Y')}", fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Ajouter une légende
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.2), markersize=8, label='Faible intensité'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.5), markersize=12, label='Intensité moyenne'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.9), markersize=16, label='Forte intensité')
        ]
        ax.legend(handles=handles, loc='lower right')
        
        # Ajouter une indication que c'est une donnée historique
        ax.text(0.5, 0.02, "Données historiques", transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, color='blue',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))
        
        # Enregistrer l'image
        frame_path = os.path.join(temp_dir, f"frame_{len(frame_paths):03d}.png")
        plt.savefig(frame_path, bbox_inches='tight')
        plt.close(fig)
        
        frame_paths.append(frame_path)
    
    # Ajouter les prévisions
    for i, row in forecast_data.iterrows():
        date = row['date']
        predicted_fires = max(0, row['prévision'])  # Assurer que c'est positif
        
        fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
        
        # Tracer les frontières du Cameroun
        ax.plot(cameroon_lon, cameroon_lat, 'k-', linewidth=1, alpha=0.5)
        
        # Définir les limites de la carte
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        
        # Distribuer les feux prévus selon la distribution spatiale historique
        for _, grid_cell in grid_counts.iterrows():
            # Calculer le nombre de feux prévu pour cette cellule
            cell_fires = predicted_fires * grid_cell['pct']
            if cell_fires > 0:
                intensity = min(1.0, cell_fires/10)
                ax.scatter(grid_cell['lon_bin'] + grid_size/2, grid_cell['lat_bin'] + grid_size/2, 
                          s=intensity*100, c=[cmap(intensity)], alpha=0.7)
        
        # Ajouter un titre et des informations
        ax.set_title(f"Prévision d'activité des feux au Cameroun\n{date.strftime('%d %B %Y')}", fontsize=14)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Ajouter une légende
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.2), markersize=8, label='Faible intensité'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.5), markersize=12, label='Intensité moyenne'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.9), markersize=16, label='Forte intensité')
        ]
        ax.legend(handles=handles, loc='lower right')
        
        # Ajouter une indication que c'est une prévision
        ax.text(0.5, 0.02, f"PRÉVISION (basée sur les tendances historiques)", transform=ax.transAxes,
                ha='center', va='bottom', fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        # Ajouter une valeur prévue
        ax.text(0.5, 0.97, f"Nombre de feux prévus: {int(predicted_fires)}", transform=ax.transAxes,
                ha='center', va='top', fontsize=12, color='black',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
        
        # Enregistrer l'image
        frame_path = os.path.join(temp_dir, f"frame_{len(frame_paths):03d}.png")
        plt.savefig(frame_path, bbox_inches='tight')
        plt.close(fig)
        
        frame_paths.append(frame_path)
    
    # 4. Créer le GIF avec imageio
    frames = []
    for frame_path in frame_paths:
        frames.append(imageio.imread(frame_path))
    
    # Créer le GIF dans le dossier temporaire
    gif_path = os.path.join(temp_dir, "forecast_animation.gif")
    imageio.mimsave(gif_path, frames, duration=1.0)  # 1 seconde par image
    
    return gif_path

def forecast_fire_activity(df, period=6, freq='M'):
    """Réaliser des prévisions sur l'activité des feux
    
    Args:
        df: DataFrame contenant les données des feux
        period: Nombre de périodes à prévoir
        freq: Fréquence ('D' pour jour, 'W' pour semaine, 'M' pour mois)
        
    Returns:
        figures: Liste de figures pour les prévisions
        forecast_data: DataFrame avec les prévisions
        insights: Liste d'insights sur les prévisions
    """
    if df.empty:
        return None, None, []
    
    figures = []
    insights = []
    
    # Agréger les données selon la fréquence
    if freq == 'D':
        time_unit = 'jours'
        df_time = df.groupby(pd.Grouper(key='date_heure', freq='D')).size().reset_index(name='count')
    elif freq == 'W':
        time_unit = 'semaines'
        df_time = df.groupby(pd.Grouper(key='date_heure', freq='W')).size().reset_index(name='count')
    else:  # 'M' par défaut
        time_unit = 'mois'
        df_time = df.groupby(pd.Grouper(key='date_heure', freq='M')).size().reset_index(name='count')
    
    # Vérifier si nous avons assez de données pour les prévisions
    if len(df_time) < 4:
        insights.append("Pas assez de données pour réaliser des prévisions fiables. Au moins 4 points temporels sont nécessaires.")
        return None, None, insights
    
    # Préparer les données pour le forecasting
    # Définir explicitement la fréquence pour éviter les avertissements
    df_time = df_time.set_index('date_heure')
    if freq == 'D':
        df_time.index = pd.DatetimeIndex(df_time.index.values, freq='D')
    elif freq == 'W':
        df_time.index = pd.DatetimeIndex(df_time.index.values, freq='W')
    else:  # 'M' par défaut
        df_time.index = pd.DatetimeIndex(df_time.index.values, freq='M')
    
    try:
        # Méthode 1: Régression simple
        # Utilisons d'abord une approche simple qui est plus robuste
        try:
            # Préparation des données pour la régression
            X = np.array(range(len(df_time))).reshape(-1, 1)
            y = df_time['count'].values
            
            # Ajuster un modèle de régression linéaire
            model = LinearRegression()
            model.fit(X, y)
            
            # Prédictions pour les données historiques et futures
            hist_pred = model.predict(X)
            future_X = np.array(range(len(df_time), len(df_time) + period)).reshape(-1, 1)
            future_pred = model.predict(future_X)
            
            # S'assurer que les prévisions sont positives
            future_pred = np.maximum(0, future_pred)
            
            # Visualisation de la tendance à long terme
            fig_trend = go.Figure()
            
            # Données historiques
            fig_trend.add_trace(go.Scatter(
                x=df_time.index,
                y=df_time['count'],
                mode='lines+markers',
                name='Historique',
                line=dict(color='blue')
            ))
            
            # Tendance historique
            fig_trend.add_trace(go.Scatter(
                x=df_time.index,
                y=hist_pred,
                mode='lines',
                name='Tendance historique',
                line=dict(color='green')
            ))
            
            # Prévision de la tendance
            future_dates = pd.date_range(start=df_time.index[-1] + pd.Timedelta(days=1), periods=period, freq=freq)
            fig_trend.add_trace(go.Scatter(
                x=future_dates,
                y=future_pred,
                mode='lines',
                name='Tendance future',
                line=dict(color='red', dash='dash')
            ))
            
            fig_trend.update_layout(
                title="Analyse de la tendance à long terme des feux",
                xaxis_title="Date",
                yaxis_title="Nombre de feux",
                legend_title="Légende",
                hovermode="x unified"
            )
            
            figures.append(fig_trend)
            
            # Créer un DataFrame pour les prévisions
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'prévision': future_pred,
                'limite_inférieure': future_pred * 0.8,  # Estimation simple des limites
                'limite_supérieure': future_pred * 1.2   # pour l'intervalle de confiance
            })
            
            # Extraire des insights sur la tendance
            slope = model.coef_[0]
            annual_change = slope * (12 if freq == 'M' else 52 if freq == 'W' else 365)
            
            if slope > 0:
                trend_intensity = "forte" if slope > 0.1 else "légère"
                insights.append(f"Tendance à la hausse {trend_intensity} détectée : environ {annual_change:.1f} feux supplémentaires par an.")
            elif slope < 0:
                trend_intensity = "forte" if slope < -0.1 else "légère"
                insights.append(f"Tendance à la baisse {trend_intensity} détectée : environ {abs(annual_change):.1f} feux de moins par an.")
            else:
                insights.append("Aucune tendance significative détectée sur le long terme.")
        
        except Exception as e:
            insights.append(f"Impossible de réaliser des prévisions par régression: {str(e)}")
        
        # Méthode 2: ARIMA (p,d,q) - modèle plus complexe mais parfois instable
        try:
            # Supprimons les avertissements pour une meilleure expérience utilisateur
            import warnings
            warnings.filterwarnings("ignore")
            
            # Différencier les données pour assurer la stationnarité
            from statsmodels.tsa.stattools import adfuller
            
            # Tester la stationnarité
            adf_result = adfuller(df_time['count'])
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05 indique stationnarité
            
            # Déterminer l'ordre de différenciation
            d = 0 if is_stationary else 1
            
            # Utiliser une spécification simple et robuste du modèle ARIMA
            p, q = 1, 1  # Ordres simples qui convergent généralement bien
            
            # Créer et ajuster le modèle avec plus d'itérations et une méthode d'optimisation robuste
            model = ARIMA(df_time['count'], order=(p, d, q))
            model_fit = model.fit(maxiter=1000, method='powell', disp=0)
            
            # Réaliser les prévisions
            last_date = df_time.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period, freq=freq)
            forecast = model_fit.forecast(steps=period)
            
            # S'assurer que les prévisions sont positives
            forecast = np.maximum(0, forecast)
            
            # Calculer l'erreur standard pour l'intervalle de confiance
            std_err = np.std(model_fit.resid) if hasattr(model_fit, 'resid') else forecast.std() * 0.1
            
            # Mettre à jour ou créer le DataFrame de prévisions
            if 'forecast_df' in locals():
                # Moyenne pondérée des deux modèles pour des prévisions plus robustes
                forecast_df['prévision'] = (forecast_df['prévision'] + forecast) / 2
                forecast_df['limite_inférieure'] = np.maximum(0, forecast - 2*std_err)
                forecast_df['limite_supérieure'] = forecast + 2*std_err
            else:
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'prévision': forecast,
                    'limite_inférieure': np.maximum(0, forecast - 2*std_err),
                    'limite_supérieure': forecast + 2*std_err
                })
            
            # Visualisation des prévisions ARIMA
            fig_arima = go.Figure()
            
            # Données historiques
            fig_arima.add_trace(go.Scatter(
                x=df_time.index,
                y=df_time['count'],
                mode='lines+markers',
                name='Historique',
                line=dict(color='blue')
            ))
            
            # Prévisions
            fig_arima.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['prévision'],
                mode='lines+markers',
                name='Prévision ARIMA',
                line=dict(color='red', dash='dash')
            ))
            
            # Intervalle de confiance
            fig_arima.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=forecast_df['limite_supérieure'].tolist() + forecast_df['limite_inférieure'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Intervalle de confiance 95%'
            ))
            
            fig_arima.update_layout(
                title=f"Prévision du nombre de feux pour les {period} prochains {time_unit}",
                xaxis_title="Date",
                yaxis_title="Nombre de feux",
                legend_title="Légende",
                hovermode="x unified"
            )
            
            # Insérer ce graphique en première position car c'est le plus important
            figures.insert(0, fig_arima)
            
            # Extraire des insights du modèle
            max_forecast_date = forecast_df.loc[forecast_df['prévision'].idxmax(), 'date']
            max_forecast_value = forecast_df['prévision'].max()
            
            insights.insert(0, f"La plus forte activité de feux est prévue autour du {max_forecast_date.strftime('%d/%m/%Y')} avec environ {int(max_forecast_value)} feux attendus.")
            
            if forecast_df['prévision'].iloc[-1] > df_time['count'].mean():
                insights.insert(1, f"Tendance à la hausse : l'activité des feux devrait être supérieure à la moyenne historique ({df_time['count'].mean():.1f}) dans les prochains {time_unit}.")
            else:
                insights.insert(1, f"Tendance à la baisse : l'activité des feux devrait être inférieure à la moyenne historique ({df_time['count'].mean():.1f}) dans les prochains {time_unit}.")
        
        except Exception as e:
            # Si ARIMA échoue, nous avons toujours les prévisions par régression
            insights.append(f"Note: Le modèle ARIMA avancé n'a pas convergé, mais les prévisions de tendance sont toujours valides.")
        
        # Méthode 3: Analyse de saisonnalité (si applicable)
        if freq == 'M' and len(df_time) >= 12:
            # Analyser la saisonnalité mensuelle
            months = []
            counts = []
            
            for month in range(1, 13):
                month_data = df_time[df_time.index.month == month]
                if not month_data.empty:
                    months.append(calendar.month_name[month])
                    counts.append(month_data['count'].mean())
            
            if months and counts:
                seasonal_df = pd.DataFrame({
                    'Mois': months,
                    'Moyenne de feux': counts
                })
                
                # Ordonner les mois correctement
                month_order = {month: i for i, month in enumerate(calendar.month_name[1:])}
                seasonal_df['month_num'] = seasonal_df['Mois'].map(month_order)
                seasonal_df = seasonal_df.sort_values('month_num')
                
                # Visualisation de la saisonnalité
                fig_seasonal = px.bar(seasonal_df, x='Mois', y='Moyenne de feux',
                                      title="Saisonnalité des feux par mois",
                                      labels={'Moyenne de feux': 'Nombre moyen de feux'})
                
                figures.append(fig_seasonal)
                
                # Identifier les mois à haut risque
                high_risk_months = seasonal_df.nlargest(3, 'Moyenne de feux')['Mois'].tolist()
                insights.append(f"Mois à plus haut risque d'incendie : {', '.join(high_risk_months)}")
                
                low_risk_months = seasonal_df.nsmallest(3, 'Moyenne de feux')['Mois'].tolist()
                insights.append(f"Mois à plus faible risque d'incendie : {', '.join(low_risk_months)}")
        
        # Réactiver les avertissements
        warnings.resetwarnings()
        
        return figures, forecast_df if 'forecast_df' in locals() else None, insights
    
    except Exception as e:
        insights.append(f"Erreur lors des prévisions: {str(e)}")
        return None, None, insights

def create_detailed_report(df, filtered_df, hotspots_df=None, forecast_insights=None):
    """Générer un rapport détaillé sur les données de feux avec des recommandations basées sur les données réelles"""
    buffer = io.BytesIO()
    
    # Créer le HTML pour le rapport
    report_html = f"""
    <html>
    <head>
        <title>Rapport détaillé sur les feux au Cameroun</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #d9534f; }}
            h2 {{ color: #f0ad4e; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
            h3 {{ color: #5bc0de; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f5f5f5; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #fff3cd; padding: 15px; border-left: 5px solid #ffc107; margin: 20px 0; }}
            .critical {{ background-color: #f8d7da; padding: 15px; border-left: 5px solid #dc3545; margin: 20px 0; }}
            .success {{ background-color: #d4edda; padding: 15px; border-left: 5px solid #28a745; margin: 20px 0; }}
            .footer {{ margin-top: 50px; font-size: 0.8em; color: #777; text-align: center; }}
            .chart {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Rapport détaillé sur les feux au Cameroun</h1>
        <p>Date de génération: {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        
        <div class="highlight">
            <h3>Résumé exécutif</h3>
            <p>Ce rapport présente une analyse détaillée des données VIIRS sur les feux au Cameroun pour la période du {filtered_df['date_heure'].min().strftime('%d/%m/%Y')} au {filtered_df['date_heure'].max().strftime('%d/%m/%Y')}. 
            Il identifie les zones à haut risque, les tendances temporelles et fournit des recommandations basées sur l'analyse des {len(filtered_df):,} points de feux détectés.</p>
    """
    
    # Ajouter des points clés au résumé exécutif
    if not filtered_df.empty:
        # Identifier les régions les plus touchées
        regions = {
            'Nord': (filtered_df['latitude'] > 8.0) & (filtered_df['longitude'] > 12.0) & (filtered_df['longitude'] < 15.0),
            'Extrême-Nord': (filtered_df['latitude'] > 10.0) & (filtered_df['longitude'] > 13.0),
            'Adamaoua': (filtered_df['latitude'] > 6.0) & (filtered_df['latitude'] < 8.0) & (filtered_df['longitude'] > 11.0) & (filtered_df['longitude'] < 15.0),
            'Est': (filtered_df['latitude'] > 2.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 13.0),
            'Centre': (filtered_df['latitude'] > 3.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 10.0) & (filtered_df['longitude'] < 13.0),
            'Sud': (filtered_df['latitude'] < 3.0) & (filtered_df['longitude'] > 10.0) & (filtered_df['longitude'] < 14.0),
            'Littoral': (filtered_df['latitude'] > 3.0) & (filtered_df['latitude'] < 5.0) & (filtered_df['longitude'] > 9.0) & (filtered_df['longitude'] < 10.5),
            'Sud-Ouest': (filtered_df['latitude'] > 4.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 8.0) & (filtered_df['longitude'] < 10.0),
            'Nord-Ouest': (filtered_df['latitude'] > 5.5) & (filtered_df['latitude'] < 7.0) & (filtered_df['longitude'] > 9.5) & (filtered_df['longitude'] < 11.0),
            'Ouest': (filtered_df['latitude'] > 5.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 10.0) & (filtered_df['longitude'] < 11.0)
        }
        
        region_counts = {}
        for region, mask in regions.items():
            region_counts[region] = len(filtered_df[mask])
        
        # Trier les régions par nombre de feux
        region_counts = {k: v for k, v in sorted(region_counts.items(), key=lambda item: item[1], reverse=True) if v > 0}
        
        # Identifier le mois avec le plus de feux
        monthly_counts = filtered_df.groupby(filtered_df['date_heure'].dt.month).size()
        max_month_num = monthly_counts.idxmax()
        max_month_name = calendar.month_name[max_month_num]
        
        # Ajouter les points clés au résumé
        if region_counts:
            top_region = list(region_counts.keys())[0]
            report_html += f"""
            <p><strong>Points clés :</strong></p>
            <ul>
                <li>La région du <strong>{top_region}</strong> est la plus touchée avec {region_counts[top_region]:,} feux détectés ({region_counts[top_region]/len(filtered_df)*100:.1f}% du total).</li>
                <li>Le mois de <strong>{max_month_name}</strong> présente la plus forte activité de feux ({monthly_counts.max():,} détections).</li>
            """
            
            # Ajouter des informations sur la confiance des détections
            if 'confidence' in filtered_df.columns:
                high_conf_pct = len(filtered_df[filtered_df['confidence'] == 'h']) / len(filtered_df) * 100
                report_html += f"""
                <li>{high_conf_pct:.1f}% des détections sont de confiance élevée, suggérant des données fiables pour la prise de décision.</li>
                """
            
            # Ajouter des informations sur la distribution jour/nuit
            day_pct = len(filtered_df[filtered_df['daynight'] == 'D']) / len(filtered_df) * 100
            night_pct = 100 - day_pct
            report_html += f"""
                <li>Répartition jour/nuit : {day_pct:.1f}% des feux sont détectés pendant la journée et {night_pct:.1f}% pendant la nuit.</li>
            </ul>
            """
        
        report_html += """
        </div>
        
        <h2>1. Statistiques générales</h2>
        <table>
            <tr>
                <th>Métrique</th>
                <th>Valeur</th>
            </tr>
            <tr>
                <td>Nombre total de feux analysés</td>
                <td>{len(filtered_df):,}</td>
            </tr>
            <tr>
                <td>Période couverte</td>
                <td>{filtered_df['date_heure'].min().strftime('%d/%m/%Y')} - {filtered_df['date_heure'].max().strftime('%d/%m/%Y')}</td>
            </tr>
            <tr>
                <td>Feux détectés de jour</td>
                <td>{len(filtered_df[filtered_df['daynight'] == 'D']):,} ({len(filtered_df[filtered_df['daynight'] == 'D']) / len(filtered_df) * 100:.1f}%)</td>
            </tr>
            <tr>
                <td>Feux détectés de nuit</td>
                <td>{len(filtered_df[filtered_df['daynight'] == 'N']):,} ({len(filtered_df[filtered_df['daynight'] == 'N']) / len(filtered_df) * 100:.1f}%)</td>
            </tr>
        """
    
    # Ajouter les statistiques sur la confiance des détections
    if 'confidence' in filtered_df.columns:
        confidence_stats = filtered_df['confidence'].value_counts(normalize=True) * 100
        report_html += f"""
            <tr>
                <td>Feux à confiance élevée (h)</td>
                <td>{len(filtered_df[filtered_df['confidence'] == 'h']):,} ({confidence_stats.get('h', 0):.1f}%)</td>
            </tr>
            <tr>
                <td>Feux à confiance nominale (n)</td>
                <td>{len(filtered_df[filtered_df['confidence'] == 'n']):,} ({confidence_stats.get('n', 0):.1f}%)</td>
            </tr>
            <tr>
                <td>Feux à confiance faible (l)</td>
                <td>{len(filtered_df[filtered_df['confidence'] == 'l']):,} ({confidence_stats.get('l', 0):.1f}%)</td>
            </tr>
        """
    
    # Ajouter les statistiques FRP si disponibles
    if 'frp' in filtered_df.columns:
        report_html += f"""
            <tr>
                <td>Puissance radiative moyenne (FRP)</td>
                <td>{filtered_df['frp'].mean():.2f} MW</td>
            </tr>
            <tr>
                <td>Puissance radiative maximale (FRP)</td>
                <td>{filtered_df['frp'].max():.2f} MW</td>
            </tr>
        """
    
    report_html += """
        </table>
        
        <h2>2. Analyse spatiale</h2>
    """
    
    # Ajouter l'analyse des hotspots
    if hotspots_df is not None and not hotspots_df.empty:
        report_html += f"""
        <h3>2.1 Zones à forte pression de feu</h3>
        <p>Les {min(10, len(hotspots_df))} zones avec la plus forte concentration de feux sont présentées ci-dessous :</p>
        <table>
            <tr>
                <th>Rang</th>
                <th>Latitude</th>
                <th>Longitude</th>
                <th>Nombre de feux</th>
                {('<th>FRP moyenne (MW)</th>' if 'frp_mean' in hotspots_df.columns else '')}
            </tr>
        """
        
        for i, row in hotspots_df.head(10).iterrows():
            report_html += f"""
            <tr>
                <td>{i+1}</td>
                <td>{row['lat_bin']:.4f}</td>
                <td>{row['lon_bin']:.4f}</td>
                <td>{row['count']:,}</td>
                {(f'<td>{row["frp_mean"]:.2f}</td>' if 'frp_mean' in hotspots_df.columns else '')}
            </tr>
            """
        
        report_html += """
        </table>
        """
        
        # Ajouter une analyse des hotspots spécifique
        if not hotspots_df.empty:
            # Identifier le hotspot principal
            top_hotspot = hotspots_df.iloc[0]
            
            # Calculer le pourcentage du total des feux dans les top 10 hotspots
            top10_count = hotspots_df.head(10)['count'].sum()
            top10_percentage = (top10_count / len(filtered_df)) * 100
            
            # Déterminer si les hotspots sont concentrés ou dispersés
            concentration_level = "très concentrés" if top10_percentage > 50 else \
                                 "modérément concentrés" if top10_percentage > 25 else "dispersés"
            
            report_html += f"""
            <div class="critical">
                <h3>Analyse des hotspots</h3>
                <p>Le point chaud principal est situé à <strong>{top_hotspot['lat_bin']:.4f}°N, {top_hotspot['lon_bin']:.4f}°E</strong> avec {top_hotspot['count']:,} feux détectés.</p>
                <p>Les 10 principaux hotspots représentent <strong>{top10_percentage:.1f}%</strong> de tous les feux détectés, indiquant que les feux sont <strong>{concentration_level}</strong> au Cameroun.</p>
            """
            
            # Ajouter des informations sur l'intensité si disponibles
            if 'frp_mean' in hotspots_df.columns:
                high_intensity_hotspots = hotspots_df[hotspots_df['frp_mean'] > hotspots_df['frp_mean'].mean()]
                high_intensity_count = len(high_intensity_hotspots)
                
                report_html += f"""
                <p>Parmi les zones identifiées, <strong>{high_intensity_count}</strong> présentent une intensité de feu (FRP) supérieure à la moyenne,
                suggérant des feux particulièrement intenses qui pourraient nécessiter une attention prioritaire.</p>
                """
            
            report_html += """
            </div>
            """
    
    # Analyse de la distribution géographique
    report_html += """
        <h3>2.2 Distribution géographique</h3>
    """
    
    # Calculer la distribution par région (approximation basée sur les coordonnées)
    if not filtered_df.empty:
        # Définir les régions approximatives du Cameroun (simplifié)
        regions = {
            'Nord': (filtered_df['latitude'] > 8.0) & (filtered_df['longitude'] > 12.0) & (filtered_df['longitude'] < 15.0),
            'Extrême-Nord': (filtered_df['latitude'] > 10.0) & (filtered_df['longitude'] > 13.0),
            'Adamaoua': (filtered_df['latitude'] > 6.0) & (filtered_df['latitude'] < 8.0) & (filtered_df['longitude'] > 11.0) & (filtered_df['longitude'] < 15.0),
            'Est': (filtered_df['latitude'] > 2.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 13.0),
            'Centre': (filtered_df['latitude'] > 3.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 10.0) & (filtered_df['longitude'] < 13.0),
            'Sud': (filtered_df['latitude'] < 3.0) & (filtered_df['longitude'] > 10.0) & (filtered_df['longitude'] < 14.0),
            'Littoral': (filtered_df['latitude'] > 3.0) & (filtered_df['latitude'] < 5.0) & (filtered_df['longitude'] > 9.0) & (filtered_df['longitude'] < 10.5),
            'Sud-Ouest': (filtered_df['latitude'] > 4.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 8.0) & (filtered_df['longitude'] < 10.0),
            'Nord-Ouest': (filtered_df['latitude'] > 5.5) & (filtered_df['latitude'] < 7.0) & (filtered_df['longitude'] > 9.5) & (filtered_df['longitude'] < 11.0),
            'Ouest': (filtered_df['latitude'] > 5.0) & (filtered_df['latitude'] < 6.0) & (filtered_df['longitude'] > 10.0) & (filtered_df['longitude'] < 11.0)
        }
        
        region_counts = {}
        for region, mask in regions.items():
            region_counts[region] = len(filtered_df[mask])
        
        # Autres feux non classifiés
        total_classified = sum(region_counts.values())
        region_counts['Non classifié'] = len(filtered_df) - total_classified
        
        # Trier par nombre de feux décroissant
        region_counts = {k: v for k, v in sorted(region_counts.items(), key=lambda item: item[1], reverse=True)}
        
        report_html += """
        <p>Distribution approximative des feux par région :</p>
        <table>
            <tr>
                <th>Région</th>
                <th>Nombre de feux</th>
                <th>Pourcentage</th>
            </tr>
        """
        
        for region, count in region_counts.items():
            report_html += f"""
            <tr>
                <td>{region}</td>
                <td>{count:,}</td>
                <td>{count / len(filtered_df) * 100:.1f}%</td>
            </tr>
            """
        
        report_html += """
        </table>
        """
        
        # Ajouter une analyse régionale spécifique
        top_regions = [k for k, v in region_counts.items() if v > 0 and k != 'Non classifié'][:3]
        if top_regions:
            report_html += """
            <div class="highlight">
                <h3>Zones prioritaires</h3>
            """
            
            for idx, region in enumerate(top_regions):
                count = region_counts[region]
                pct = count / len(filtered_df) * 100
                
                # Générer des recommandations spécifiques à chaque région
                if region == 'Nord' or region == 'Extrême-Nord':
                    report_html += f"""
                    <p><strong>{idx+1}. Région {region} ({pct:.1f}% des feux)</strong> - Zone de savane sèche avec pratiques agricoles extensives.
                    Cette région nécessite une stratégie de gestion adaptée à la vulnérabilité de la savane et aux pratiques
                    de brûlis agricoles. Recommandation : mise en place de pare-feux communautaires et introduction de
                    techniques agricoles alternatives au brûlis.</p>
                    """
                elif region == 'Est':
                    report_html += f"""
                    <p><strong>{idx+1}. Région {region} ({pct:.1f}% des feux)</strong> - Zone forestière avec progression de la déforestation.
                    Cette région nécessite une surveillance accrue des zones frontalières des aires protégées. Recommandation :
                    renforcement des patrouilles dans les zones tampons des parcs nationaux et renforcement de l'application de la loi.</p>
                    """
                elif region == 'Adamaoua':
                    report_html += f"""
                    <p><strong>{idx+1}. Région {region} ({pct:.1f}% des feux)</strong> - Zone de transition avec forte activité pastorale.
                    La pratique du brûlis pour le renouvellement des pâturages contribue significativement aux feux dans cette région.
                    Recommandation : travail avec les communautés pastorales pour développer des systèmes de rotation des pâturages et
                    des périodes contrôlées de brûlis.</p>
                    """
                else:
                    report_html += f"""
                    <p><strong>{idx+1}. Région {region} ({pct:.1f}% des feux)</strong> - Cette région présente une concentration significative de feux
                    et devrait être prioritaire pour les interventions de prévention et de gestion. Recommandation : évaluation détaillée
                    des causes spécifiques à la région et développement d'une stratégie adaptée au contexte local.</p>
                    """
            
            report_html += """
            </div>
            """
    
    # Analyse temporelle
    report_html += """
        <h2>3. Analyse temporelle</h2>
    """
    
    if not filtered_df.empty:
        # Distribution par mois
        monthly_df = filtered_df.groupby(filtered_df['date_heure'].dt.month).size()
        monthly_df = monthly_df.reset_index()
        monthly_df.columns = ['Mois', 'Nombre de feux']
        monthly_df['Nom du mois'] = monthly_df['Mois'].apply(lambda x: calendar.month_name[x])
        
        report_html += """
        <h3>3.1 Distribution mensuelle</h3>
        <table>
            <tr>
                <th>Mois</th>
                <th>Nombre de feux</th>
                <th>Pourcentage</th>
            </tr>
        """
        
        for _, row in monthly_df.iterrows():
            report_html += f"""
            <tr>
                <td>{row['Nom du mois']}</td>
                <td>{row['Nombre de feux']:,}</td>
                <td>{row['Nombre de feux'] / len(filtered_df) * 100:.1f}%</td>
            </tr>
            """
        
        report_html += """
        </table>
        """
        
        # Mois avec le plus/moins de feux
        max_month = monthly_df.loc[monthly_df['Nombre de feux'].idxmax()]
        min_month = monthly_df.loc[monthly_df['Nombre de feux'].idxmin()]
        
        # Analyser les saisons de feux
        max_month_num = max_month['Mois']
        season = ""
        if max_month_num in [11, 12, 1, 2]:  # Saison sèche Nord-Cameroun
            season = "saison sèche"
            season_context = "Cette période correspond à la saison sèche dans le nord du pays, où l'absence de précipitations et la végétation desséchée favorisent les départs de feu."
        elif max_month_num in [3, 4, 5]:  # Transition / début de saison des pluies
            season = "fin de saison sèche/début de saison des pluies"
            season_context = "Cette période de transition entre saison sèche et saison des pluies est souvent marquée par des pratiques de défrichement agricole avant les semis."
        else:  # Saison des pluies / préparation des cultures
            season = "saison des pluies"
            season_context = "Cette période correspond à la saison des pluies dans plusieurs régions, suggérant des feux potentiellement liés à la préparation des terres agricoles malgré l'humidité."
        
        report_html += f"""
        <div class="highlight">
            <h3>Analyse saisonnière</h3>
            <p><strong>Pic saisonnier :</strong> {max_month['Nom du mois']} est le mois avec le plus grand nombre de feux ({max_month['Nombre de feux']:,}), représentant {max_month['Nombre de feux'] / len(filtered_df) * 100:.1f}% du total annuel.</p>
            <p><strong>Contexte saisonnier :</strong> Ce pic se produit durant la {season}. {season_context}</p>
            <p><strong>Creux saisonnier :</strong> {min_month['Nom du mois']} est le mois avec le moins de feux ({min_month['Nombre de feux']:,}).</p>
            <p><strong>Implications pour la gestion :</strong> Les ressources de prévention et d'intervention devraient être maximisées {max_month['Nom du mois']} mois avant le pic (en {calendar.month_name[(max_month_num-1) if max_month_num > 1 else 12]}) pour permettre la mise en place de mesures préventives efficaces.</p>
        </div>
        """
        
        # Distribution par heure (jour/nuit)
        if 'acq_time' in filtered_df.columns:
            filtered_df['heure'] = filtered_df['acq_time'] // 100
            hourly_df = filtered_df.groupby('heure').size().reset_index()
            hourly_df.columns = ['Heure', 'Nombre de feux']
            
            report_html += """
            <h3>3.2 Distribution horaire</h3>
            <p>Répartition des feux par heure de détection :</p>
            <table>
                <tr>
                    <th>Heure</th>
                    <th>Nombre de feux</th>
                    <th>Pourcentage</th>
                </tr>
            """
            
            for _, row in hourly_df.iterrows():
                report_html += f"""
                <tr>
                    <td>{row['Heure']:02d}h00</td>
                    <td>{row['Nombre de feux']:,}</td>
                    <td>{row['Nombre de feux'] / len(filtered_df) * 100:.1f}%</td>
                </tr>
                """
            
            report_html += """
            </table>
            """
            
            # Heures avec le plus/moins de feux
            max_hour = hourly_df.loc[hourly_df['Nombre de feux'].idxmax()]
            early_morning = hourly_df[(hourly_df['Heure'] >= 5) & (hourly_df['Heure'] <= 9)]
            afternoon = hourly_df[(hourly_df['Heure'] >= 13) & (hourly_df['Heure'] <= 17)]
            
            early_morning_total = early_morning['Nombre de feux'].sum() if not early_morning.empty else 0
            afternoon_total = afternoon['Nombre de feux'].sum() if not afternoon.empty else 0
            
            peak_period = "matinée" if early_morning_total > afternoon_total else "après-midi"
            
            report_html += f"""
            <p><strong>Pic horaire :</strong> {max_hour['Heure']:02d}h00 est l'heure avec le plus grand nombre de détections ({max_hour['Nombre de feux']:,}).</p>
            <p><strong>Analyse de la distribution horaire :</strong> Les feux sont plus fréquents en {peak_period}, ce qui suggère un lien avec les activités humaines comme {"le défrichement agricole matinal" if peak_period == "matinée" else "les activités agricoles et pastorales de l'après-midi"}.</p>
            """
    
    # Ajouter les prévisions et insights
    if forecast_insights:
        report_html += """
        <h2>4. Prévisions et tendances</h2>
        <div class="highlight">
        """
        
        for i, insight in enumerate(forecast_insights):
            report_html += f"""
            <p>{i+1}. {insight}</p>
            """
        
        report_html += """
        </div>
        """
    
    # Générer des recommandations basées sur les données réelles
    report_html += """
        <h2>5. Recommandations spécifiques basées sur l'analyse</h2>
    """
    
    # Création de recommandations dynamiques basées sur les données
    if not filtered_df.empty:
        # Analyser les tendances pour les recommandations
        has_seasonal_pattern = len(filtered_df.groupby(filtered_df['date_heure'].dt.month).size()) > 3
        has_spatial_hotspots = hotspots_df is not None and not hotspots_df.empty and hotspots_df['count'].max() > len(filtered_df) * 0.05
        
        # Déterminer les régions principales
        if region_counts:
            top_regions = [k for k, v in region_counts.items() if v > 0 and k != 'Non classifié'][:3]
        else:
            top_regions = []
        
        report_html += """
        <div class="critical">
            <h3>5.1 Actions prioritaires</h3>
            <ol>
        """
        
        # Recommandations spatiales
        if has_spatial_hotspots and hotspots_df is not None:
            top_hotspot = hotspots_df.iloc[0]
            second_hotspot = hotspots_df.iloc[1] if len(hotspots_df) > 1 else None
            
            report_html += f"""
                <li><strong>Intervention ciblée dans le hotspot principal ({top_hotspot['lat_bin']:.4f}°N, {top_hotspot['lon_bin']:.4f}°E) :</strong> 
                Déployer immédiatement des équipes de surveillance dans cette zone qui représente une concentration significative de feux.
                Établir une base opérationnelle temporaire pendant la saison des feux pour réduire les temps d'intervention.</li>
            """
            
            if second_hotspot is not None:
                report_html += f"""
                <li><strong>Établissement d'un périmètre de sécurité :</strong> 
                Créer un réseau de pare-feux entre les deux principaux hotspots ({top_hotspot['lat_bin']:.4f}°N, {top_hotspot['lon_bin']:.4f}°E et 
                {second_hotspot['lat_bin']:.4f}°N, {second_hotspot['lon_bin']:.4f}°E) pour éviter la propagation des incendies d'une zone critique à l'autre.</li>
                """
        
        # Recommandations temporelles
        if has_seasonal_pattern:
            # Déterminer le mois de début de la saison des feux (2 mois avant le pic)
            max_month_num = max_month['Mois'] 
            prep_month_num = (max_month_num - 2) % 12
            if prep_month_num == 0:
                prep_month_num = 12
            prep_month = calendar.month_name[prep_month_num]
            
            report_html += f"""
                <li><strong>Mise en place d'un système d'alerte précoce :</strong> 
                Intensifier la surveillance des conditions météorologiques et de la végétation dès {prep_month}, 
                soit 2 mois avant le pic saisonnier habituel. Déployer des observateurs communautaires formés
                pour signaler les départs de feu.</li>
            """
        
        # Recommandations régionales
        if top_regions:
            region_specific_recs = {
                'Nord': "Mettre en place un programme de brûlis contrôlés préventifs avant la saison sèche et former les communautés rurales à ces techniques.",
                'Extrême-Nord': "Développer un réseau de points d'eau stratégiques accessibles aux équipes d'intervention et promouvoir les cultures résistantes à la sécheresse.",
                'Adamaoua': "Collaborer avec les éleveurs pour établir un calendrier rotatif des zones de pâturage pour éviter la surexploitation et réduire le besoin de brûlis.",
                'Est': "Établir des zones tampons autour des aires protégées et engager les communautés limitrophes dans la surveillance participative.",
                'Centre': "Mettre en œuvre des programmes d'agroforesterie pour réduire la dépendance au défrichement par brûlis.",
                'Sud': "Renforcer la protection des zones forestières primaires et secondaires et développer des alternatives économiques à l'exploitation non durable.",
                'Littoral': "Développer des systèmes d'alerte précoce autour des zones industrielles et des interfaces urbain-forêt.",
                'Sud-Ouest': "Établir des corridors coupe-feu dans les zones de plantation et former les agriculteurs à la gestion des résidus de culture.",
                'Nord-Ouest': "Mettre en place un système communautaire de gestion des feux adapté au contexte montagneux de la région.",
                'Ouest': "Promouvoir des techniques agricoles sans feu et établir des comités locaux de vigilance."
            }
            
            for region in top_regions[:2]:  # Limiter à 2 recommandations régionales
                if region in region_specific_recs:
                    report_html += f"""
                    <li><strong>Région {region} :</strong> {region_specific_recs[region]}</li>
                    """
        
        report_html += """
            </ol>
        </div>
        """
        
        # Recommandations de conservation
        report_html += """
        <h3>5.2 Conservation et gestion des écosystèmes</h3>
        """
        
        # Identifier les écosystèmes sensibles potentiellement affectés
        ecosystems = []
        if len(filtered_df[filtered_df['latitude'] > 8.0]) > len(filtered_df) * 0.3:
            ecosystems.append("savanes sèches du nord")
        if len(filtered_df[filtered_df['latitude'].between(6.0, 8.0) & filtered_df['longitude'].between(11.0, 15.0)]) > len(filtered_df) * 0.2:
            ecosystems.append("savanes humides de l'Adamaoua")
        if len(filtered_df[filtered_df['latitude'] < 6.0]) > len(filtered_df) * 0.2:
            ecosystems.append("forêts tropicales du sud")
        
        if ecosystems:
            report_html += """
            <ul>
            """
            
            for ecosystem in ecosystems:
                if "savanes sèches" in ecosystem:
                    report_html += f"""
                    <li><strong>Protection des {ecosystem} :</strong> Ces écosystèmes montrent une forte vulnérabilité aux feux d'après les données analysées.
                    Mettre en place un réseau de surveillance spécifique et développer un programme de restauration des sols dégradés par les feux répétés.</li>
                    """
                elif "savanes humides" in ecosystem:
                    report_html += f"""
                    <li><strong>Gestion intégrée des {ecosystem} :</strong> Développer un modèle de gestion collaborative impliquant les éleveurs et les agriculteurs
                    pour réduire les conflits d'usage qui conduisent souvent à des feux délibérés. Promouvoir des techniques de pâturage améliorées.</li>
                    """
                elif "forêts tropicales" in ecosystem:
                    report_html += f"""
                    <li><strong>Conservation des {ecosystem} :</strong> Renforcer la surveillance des lisières forestières qui montrent une activité de feu élevée.
                    Établir des zones tampons entre les zones agricoles et forestières avec une transition progressive des usages.</li>
                    """
            
            report_html += """
            </ul>
            """
        else:
            report_html += """
            <p>Les données ne permettent pas d'identifier clairement les écosystèmes affectés. Une analyse plus détaillée croisant ces données avec des cartes d'écosystèmes est recommandée.</p>
            """
        
        # Recommandations pour l'intervention et la prévention
        report_html += """
        <h3>5.3 Stratégies d'intervention et de prévention</h3>
        """
        
        # Analyser les données pour des recommandations spécifiques
        day_night_ratio = len(filtered_df[filtered_df['daynight'] == 'D']) / len(filtered_df[filtered_df['daynight'] == 'N']) if len(filtered_df[filtered_df['daynight'] == 'N']) > 0 else float('inf')
        
        report_html += """
        <ul>
        """
        
        # Recommandations basées sur la distribution jour/nuit
        if day_night_ratio > 3:  # Beaucoup plus de feux de jour
            report_html += """
            <li><strong>Surveillance diurne renforcée :</strong> Les données montrent une forte prédominance des feux diurnes,
            suggérant des causes humaines délibérées (défrichement agricole, brûlis pastoral). Concentrer les ressources de
            surveillance pendant les heures de jour et mettre en place des campagnes de sensibilisation ciblées sur ces pratiques.</li>
            """
        elif day_night_ratio < 0.5:  # Beaucoup plus de feux de nuit
            report_html += """
            <li><strong>Patrouilles nocturnes :</strong> La prépondérance des feux nocturnes suggère des activités
            potentiellement illégales (défrichement clandestin, braconnage). Mettre en place des patrouilles nocturnes
            dans les zones critiques et renforcer la législation contre les feux nocturnes non autorisés.</li>
            """
        else:  # Distribution équilibrée
            report_html += """
            <li><strong>Surveillance continue :</strong> La distribution équilibrée des feux entre jour et nuit nécessite
            une présence constante des équipes de surveillance. Développer un système de rotation des équipes pour assurer
            une couverture 24/7 dans les zones prioritaires pendant la saison des feux.</li>
            """
        
        # Recommandation basée sur les hotspots
        if has_spatial_hotspots and not hotspots_df.empty:
            report_html += f"""
            <li><strong>Cartographie dynamique du risque :</strong> Développer une carte de risque d'incendie régulièrement mise à jour
            et partagée avec les acteurs locaux. La concentration des feux dans des hotspots spécifiques ({hotspots_df.iloc[0]['lat_bin']:.4f}°N, {hotspots_df.iloc[0]['lon_bin']:.4f}°E)
            suggère que des interventions très ciblées peuvent avoir un impact significatif.</li>
            """
        
        # Recommandations basées sur la confiance des détections
        if 'confidence' in filtered_df.columns:
            high_conf_pct = len(filtered_df[filtered_df['confidence'] == 'h']) / len(filtered_df) * 100
            if high_conf_pct < 30:
                report_html += """
                <li><strong>Amélioration du système de détection :</strong> Le faible pourcentage de détections à haute confiance
                suggère la nécessité de compléter les données satellitaires par des observations au sol. Mettre en place un réseau
                d'observateurs communautaires équipés d'une application mobile simple pour signaler et vérifier les feux.</li>
                """
        
        report_html += """
        </ul>
        """
        
        # Recommandations pour la sensibilisation et l'éducation
        report_html += """
        <h3>5.4 Sensibilisation et éducation</h3>
        <ul>
        """
        
        # Analyser la saisonnalité pour des recommandations spécifiques
        if has_seasonal_pattern:
            # Calculer les mois précédant le pic
            max_month_num = max_month['Mois']
            pre_peak_month1 = (max_month_num - 1) if max_month_num > 1 else 12
            pre_peak_month2 = (max_month_num - 2) if max_month_num > 2 else (12 if max_month_num == 1 else 11)
            
            report_html += f"""
            <li><strong>Campagnes de sensibilisation saisonnières :</strong> Intensifier les efforts de sensibilisation
            deux mois avant le pic de feux observé, soit en {calendar.month_name[pre_peak_month2]} et {calendar.month_name[pre_peak_month1]}.
            Cibler spécifiquement les écoles, les associations d'agriculteurs et les groupements d'éleveurs dans les zones à haut risque.</li>
            """
            
            report_html += """
            <li><strong>Formation communautaire :</strong> Développer un programme de formation des leaders communautaires
            aux techniques de prévention et de gestion des feux. Ces personnes peuvent ensuite servir de relais d'information
            et de premiers intervenants dans leurs communautés respectives.</li>
            """
        
        report_html += """
        </ul>
        
        <h3>5.5 Suivi et évaluation</h3>
        <ul>
            <li><strong>Analyse d'impact :</strong> Mettre en place un système de suivi pour évaluer l'efficacité des interventions
            en comparant les données avant et après mise en œuvre des recommandations.</li>
            <li><strong>Rapports périodiques :</strong> Produire des analyses trimestrielles sur l'évolution de la situation
            des feux pour ajuster les stratégies en fonction des résultats observés.</li>
            <li><strong>Collaboration scientifique :</strong> Établir des partenariats avec des institutions de recherche pour
            approfondir l'analyse des causes et des impacts des feux sur les écosystèmes camerounais.</li>
        </ul>
        """
        
        # Conclusion
        report_html += f"""
        <div class="success">
            <h3>Conclusion</h3>
            <p>Cette analyse de {len(filtered_df):,} points de feux détectés au Cameroun entre le {filtered_df['date_heure'].min().strftime('%d/%m/%Y')} et le {filtered_df['date_heure'].max().strftime('%d/%m/%Y')}
            révèle des modèles spatiaux et temporels distincts qui peuvent guider efficacement les stratégies de gestion des feux.</p>
            
            <p>Les interventions ciblées dans les hotspots identifiés, combinées à des approches préventives
            avant les périodes de pic saisonnier, offrent la meilleure opportunité de réduire significativement l'incidence
            des feux et leurs impacts sur les écosystèmes et les communautés.</p>
            
            <p>Un suivi régulier de l'évolution des tendances et l'adaptation des stratégies en conséquence seront
            essentiels pour maintenir l'efficacité des interventions à long terme.</p>
        </div>
        
        <div class="footer">
            <p>Rapport généré par l'application d'analyse des feux du Cameroun</p>
        </div>
    </body>
    </html>
    """
    
    return report_html

# Interface utilisateur
def main():
    st.title("🔥 CM-FI (CAMEROON FireInsight)")
    
    # Barre latérale pour le chargement des fichiers et les options
    with st.sidebar:
        st.header("Chargement des données")
        
        uploaded_files = st.file_uploader(
            "Téléchargez vos fichiers CSV VIIRS (un par année)",
            type=["csv"],
            accept_multiple_files=True
        )
        
        if not uploaded_files:
            st.warning("Veuillez télécharger au moins un fichier CSV pour commencer l'analyse")
            
            # Exemple de données pour démonstration
            if st.button("Utiliser les données de démonstration"):
                # Dans un environnement réel, vous pourriez avoir des fichiers de démonstration
                st.info("Fonctionnalité de démonstration non disponible dans ce prototype")
        
        st.divider()
        st.header("Filtres")
        
        # Options de filtrage
        confidence_options = st.selectbox(
            "Niveau de confiance minimum",
            options=[('l', 'Faible (inclut tous)'), ('n', 'Nominal'), ('h', 'Élevé')],
            format_func=lambda x: x[1],
            index=0
        )
        
        day_night = st.selectbox(
            "Période",
            options=[('both', 'Jour et Nuit'), ('D', 'Jour seulement'), ('N', 'Nuit seulement')],
            format_func=lambda x: x[1],
            index=0
        )
        
        st.divider()
        st.header("Prévisions")
        forecast_period = st.slider(
            "Période de prévision",
            min_value=1,
            max_value=12,
            value=6,
            help="Nombre de périodes à prévoir dans le futur"
        )
        
        forecast_freq = st.selectbox(
            "Fréquence de prévision",
            options=[('M', 'Mensuelle'), ('W', 'Hebdomadaire'), ('D', 'Journalière')],
            format_func=lambda x: x[1],
            index=0
        )
    
    # Chargement des données
    dfs = {}
    years_available = []
    
    if uploaded_files:
        for file in uploaded_files:
            # Extraction de l'année du nom de fichier (supposant format "viirssnpp_YYYY_*.csv")
            file_name = file.name
            try:
                year = int(file_name.split('_')[1])
            except:
                # Si l'année n'est pas dans le nom de fichier, l'extraire des données
                temp_df = pd.read_csv(file)
                if 'acq_date' in temp_df.columns:
                    year = pd.to_datetime(temp_df['acq_date']).dt.year.mode()[0]
                else:
                    year = datetime.now().year
            
            df = load_data(file)
            if df is not None:
                dfs[year] = df
                years_available.append(year)
        
        if dfs:
            # Combiner tous les DataFrames
            all_data = pd.concat(dfs.values())
            
            # Appliquer les filtres
            filtered_data = filter_data(
                all_data, 
                confidence_min=confidence_options[0],
                day_night=day_night[0] if day_night[0] != 'both' else None
            )
            
            # Créer les onglets pour différentes analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Zones à forte pression de feu", 
                "Série temporelle", 
                "Comparaison entre années",
                "Prévisions",
                "Rapport détaillé"
            ])
            
            with tab1:
                st.header("🗺️ Analyse des zones à forte pression de feu")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Carte des hotspots de feux")
                    
                    grid_size = st.slider(
                        "Taille de la grille (degrés)",
                        min_value=0.05,
                        max_value=0.5,
                        value=0.1,
                        step=0.05,
                        help="Définit la résolution de l'analyse des hotspots"
                    )
                    
                    hotspot_map, hotspots_df = create_hotspot_map(filtered_data, grid_size)
                    
                    if hotspot_map:
                        folium_static(hotspot_map, width=700)
                    else:
                        st.warning("Pas assez de données pour créer la carte des hotspots")
                
                with col2:
                    st.subheader("Top zones à risque")
                    
                    if not hotspots_df.empty:
                        top_n = min(10, len(hotspots_df))
                        st.dataframe(
                            hotspots_df.head(top_n).reset_index(drop=True)
                            .rename(columns={
                                'lat_bin': 'Latitude',
                                'lon_bin': 'Longitude',
                                'count': 'Nombre de feux',
                                'frp_mean': 'FRP moyenne'
                            }),
                            use_container_width=True
                        )
                        
                        # Téléchargement des données
                        csv = hotspots_df.to_csv(index=False)
                        st.download_button(
                            label="Télécharger les données de hotspots (CSV)",
                            data=csv,
                            file_name=f"hotspots_feux_cameroun.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("Aucune donnée disponible pour l'analyse des hotspots")
            
            with tab2:
                st.header("📈 Analyse temporelle des feux")
                
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    time_freq = st.radio(
                        "Fréquence d'analyse",
                        options=[
                            ('H', 'Horaire'),
                            ('D', 'Journalière'),
                            ('W', 'Hebdomadaire'),
                            ('M', 'Mensuelle')
                        ],
                        format_func=lambda x: x[1],
                        index=3
                    )
                    
                    # Date range selector (if applicable)
                    if filtered_data is not None and not filtered_data.empty:
                        min_date = filtered_data['date_heure'].min().date()
                        max_date = filtered_data['date_heure'].max().date()
                        
                        date_range = st.date_input(
                            "Plage de dates",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date
                        )
                        
                        # Appliquer le filtre de date
                        if len(date_range) == 2:
                            time_data = filter_data(filtered_data, date_range=date_range)
                        else:
                            time_data = filtered_data
                    else:
                        time_data = filtered_data
                
                with col1:
                    # Créer la série temporelle
                    time_fig = create_time_series(time_data, frequency=time_freq[0])
                    
                    if time_fig:
                        st.plotly_chart(time_fig, use_container_width=True)
                    else:
                        st.warning("Pas assez de données pour créer la série temporelle")
            
            with tab3:
                st.header("🔄 Comparaison entre années")
                
                if len(years_available) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        year1 = st.selectbox(
                            "Première année",
                            options=sorted(years_available),
                            index=0
                        )
                    
                    with col2:
                        year2 = st.selectbox(
                            "Deuxième année",
                            options=sorted(years_available),
                            index=min(1, len(years_available)-1)
                        )
                    
                    if year1 != year2:
                        # Générer les comparaisons
                        fig_total, fig_monthly, fig_frp = compare_years(filtered_data, year1, year2)
                        
                        if fig_total and fig_monthly:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.plotly_chart(fig_total, use_container_width=True)
                            
                            with col2:
                                if fig_frp:
                                    st.plotly_chart(fig_frp, use_container_width=True)
                            
                            st.plotly_chart(fig_monthly, use_container_width=True)
                        else:
                            st.warning("Impossible de générer la comparaison avec les données disponibles")
                    else:
                        st.warning("Veuillez sélectionner deux années différentes pour la comparaison")
                else:
                    st.warning(
                        "Pour comparer des années, vous devez télécharger des données pour au moins deux années différentes"
                    )
            
            with tab4:
                st.header("🔮 Prévisions d'activité des feux")
                
                st.write("""
                Cette section utilise des modèles statistiques pour prévoir l'activité future des feux
                en se basant sur les tendances historiques et les modèles saisonniers.
                """)
                
                # Générer les prévisions
                forecast_figures, forecast_data, forecast_insights = forecast_fire_activity(
                    filtered_data, 
                    period=forecast_period, 
                    freq=forecast_freq[0]
                )
                
                if forecast_figures:
                    for fig in forecast_figures:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Afficher les insights
                    st.subheader("Insights clés des prévisions")
                    for i, insight in enumerate(forecast_insights):
                        st.markdown(f"**{i+1}.** {insight}")
                    
                    # Afficher les données de prévision
                    if forecast_data is not None:
                        st.subheader("Données de prévision")
                        st.dataframe(forecast_data)
                        
                        # Téléchargement des prévisions
                        csv = forecast_data.to_csv(index=False)
                        st.download_button(
                            label="Télécharger les prévisions (CSV)",
                            data=csv,
                            file_name="previsions_feux_cameroun.csv",
                            mime="text/csv"
                        )
                        
                        # Créer une carte animée des prévisions
                        st.subheader("Carte animée des prévisions")
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write("""
                            Cette carte interactive montre l'évolution temporelle des feux avec une animation, 
                            intégrant à la fois les données historiques récentes et les prévisions futures. 
                            Les zones rouges indiquent une forte activité de feux.
                            """)
                            
                            animated_map = create_animated_forecast_map(
                                filtered_data,
                                forecast_data,
                                period=forecast_period,
                                freq=forecast_freq[0]
                            )
                            
                            if animated_map:
                                folium_static(animated_map, width=400, height=400)
                            else:
                                st.warning("Impossible de générer la carte animée des prévisions.")
                        
                        with col2:
                            st.write("""
                            Vous pouvez également télécharger une animation GIF qui montre l'évolution des feux
                            au fil du temps. Cette animation peut être facilement partagée et visionnée sans 
                            l'application.
                            """)
                            
                            if st.button("Générer l'animation GIF des prévisions"):
                                with st.spinner("Génération de l'animation en cours..."):
                                    gif_path = create_forecast_animation_gif(
                                        filtered_data,
                                        forecast_data,
                                        grid_size=0.1,
                                        freq=forecast_freq[0],
                                        period=forecast_period
                                    )
                                    
                                    if gif_path and os.path.exists(gif_path):
                                        # Lire le GIF en tant que bytes
                                        with open(gif_path, "rb") as file:
                                            gif_bytes = file.read()
                                        
                                        # Afficher le GIF
                                        st.image(gif_bytes, caption="Animation des prévisions de feux", use_column_width=True)
                                        
                                        # Bouton de téléchargement
                                        st.download_button(
                                            label="Télécharger l'animation GIF",
                                            data=gif_bytes,
                                            file_name="previsions_feux_cameroun.gif",
                                            mime="image/gif"
                                        )
                                        
                                        st.success("Animation GIF générée avec succès !")
                                    else:
                                        st.error("Impossible de générer l'animation GIF.")
                            
                            st.info("L'animation GIF peut être partagée facilement par email ou intégrée dans des présentations.")
                else:
                    st.warning("Impossible de générer des prévisions avec les données disponibles.")
                    if forecast_insights:
                        st.info("\n".join(forecast_insights))
            
            with tab5:
                st.header("📊 Rapport détaillé")
                
                st.write("""
                Générez un rapport complet basé sur les données et les filtres sélectionnés.
                Ce rapport inclut des statistiques détaillées, des analyses spatiales et temporelles,
                ainsi que des recommandations pour la gestion des risques.
                """)
                
                # Génération du rapport
                if st.button("Générer le rapport détaillé"):
                    with st.spinner("Génération du rapport en cours..."):
                        # Obtenir les hotspots pour le rapport
                        _, hotspots_df = create_hotspot_map(filtered_data, grid_size=0.1)
                        
                        # Générer le rapport HTML
                        report_html = create_detailed_report(
                            all_data, 
                            filtered_data,
                            hotspots_df=hotspots_df,
                            forecast_insights=forecast_insights if 'forecast_insights' in locals() else None
                        )
                        
                        # Afficher un aperçu du rapport
                        st.subheader("Aperçu du rapport")
                        st.components.v1.html(report_html, height=500, scrolling=True)
                        
                        # Option de téléchargement
                        b64 = base64.b64encode(report_html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64}" download="rapport_feux_cameroun.html" target="_blank">Télécharger le rapport complet (HTML)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        st.success("Rapport généré avec succès !")
                
                st.info("Le rapport peut être téléchargé et partagé avec les parties prenantes pour faciliter la prise de décision.")
        else:
            st.warning("Aucune donnée n'a pu être chargée correctement")
    else:
        # Afficher des instructions et informations utiles quand aucun fichier n'est chargé
        st.markdown("""
        ## Bienvenue dans l'outil d'analyse des feux du Cameroun
        
        Cette application vous permet d'analyser les données VIIRS sur les feux au Cameroun pour :
        
        - **Identifier les zones à forte pression de feu** à l'aide de cartes de chaleur et d'analyses spatiales
        - **Analyser les tendances temporelles** pour détecter les périodes à risque élevé
        - **Comparer les données entre différentes années** pour évaluer l'évolution des régimes de feu
        
        ### Comment utiliser cette application
        
        1. Téléchargez vos fichiers CSV VIIRS dans le panneau latéral (un fichier par année)
        2. Utilisez les filtres pour affiner votre analyse
        3. Explorez les différents onglets pour visualiser les résultats
        
        ### Format de données attendu
        
        L'application attend des fichiers CSV au format VIIRS SNPP contenant les colonnes suivantes :
        - `latitude`, `longitude` : Coordonnées du feu
        - `acq_date`, `acq_time` : Date et heure d'acquisition
        - `confidence` : Niveau de confiance de la détection
        - `frp` : Puissance radiative du feu (optionnel)
        - `daynight` : Détection de jour ou de nuit
        
        Vous pouvez obtenir ces données depuis le site [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/).
        """)

if __name__ == "__main__":
    main()
