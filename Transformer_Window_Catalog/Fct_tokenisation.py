import matplotlib
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import astropy.units as u
import random
import pandas as pd
import copy
import json
import numba

from tqdm.auto import tqdm
from textwrap import fill
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.coordinates import search_around_sky
from matplotlib.patches import Rectangle
from astropy.visualization import simple_norm
from astropy.wcs import WCS
from sklearn.model_selection import train_test_split
import matplotlib.colors as colors
from astropy.coordinates import angular_separation
import astropy.units as u
from astropy.table import join
from collections import Counter

from Constantes import name_dir
from Constantes import WINDOW_SIZE_ARCMIN
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import use_log_scale_Xamin, use_log_scale_input_clusters, use_log_scale_input_AGN
from Constantes import SELECTED_COLUMNS_Xamin, SELECTED_COLUMNS_input_clusters, SELECTED_COLUMNS_input_AGN
from Constantes import print_parameters

print_parameters()



#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                       DEFINITIONS FUNCTIONS
#///////////////////////////////////////////////////////////////////////////////////////////////////////


def CreateListID_Xamin(all_xamin_ids, list_id_xamin_amas, proportion=1):
    """
    Creates a combined list of IDs by taking the intersection of two ID lists
    and adding a random subset of IDs from the remaining IDs in `all_xamin_ids`
    that are not in `list_id_xamin_amas`, scaled by a given proportion.

    Args:
        all_xamin_ids (np.ndarray or list): Full list of IDs.
        list_id_xamin_amas (np.ndarray or list): Reference subset of IDs.
        proportion (float, optional): Proportion of additional IDs to add 
                                      (relative to the number of common IDs). Default is 1.0.

    Returns:
        np.ndarray: Combined array of IDs.
    """

    # Trouver les IDs communs
    common_ids = np.intersect1d(list_id_xamin_amas, all_xamin_ids)

    # Trouver les IDs disponibles pour le tirage aléatoire (ceux qui ne sont PAS dans list_id_xamin_amas)
    available_ids = np.setdiff1d(all_xamin_ids, list_id_xamin_amas)

    # Nombre d'IDs à ajouter = nombre d'IDs communs
    n_additional = int(proportion * len(common_ids))  # % du nombre d'IDs communs

    if len(available_ids) >= n_additional: # Tirage aléatoire sans remise
        random_ids = np.random.choice(available_ids, size=n_additional, replace=False)
    else:
        random_ids = available_ids  # Si pas assez d'IDs disponibles

    # Combinaison finale
    combined_ids = np.concatenate([common_ids, random_ids])

    return combined_ids







def Batisseuse2Fenetres(data_Xamin, data_clusters, data_AGN, list_ID_Xamin, 
                        cluster_columns_to_keep = ['window', 'R.A.', 'Dec', 'm200', 'z'],
                        #AGN_columns_to_keep = ['window', 'ra_mag_gal', 'dec_mag_gal', 'observed_redshift_gal', 'log_stellar_mass_h70'],
                        AGN_columns_to_keep = ['window', 'ra_mag_gal', 'dec_mag_gal'],
                        window_size_arcmin = WINDOW_SIZE_ARCMIN):
    """
    Extracts sources, clusters, and AGNs within windows centered on Xamin IDs.
    Recenters RA and Dec coordinates relative to window centers and sorts by separation.
    
    Args:
        data_Xamin (Astropy Table or similar): Xamin sources catalog with columns including 'PNT_RA', 'PNT_DEC', 'ID_Xamin'.
        data_clusters (Astropy Table): Cluster catalog with columns including 'R.A.', 'Dec', etc.
        data_AGN (Astropy Table): AGN catalog with columns including 'ra_mag_gal', 'dec_mag_gal', etc.
        list_ID_Xamin (list or array): List of Xamin IDs defining window centers.
        cluster_columns_to_keep (list): Columns to keep from clusters.
        AGN_columns_to_keep (list): Columns to keep from AGNs.
        window_size_arcmin (float): Size of each window in arcminutes.
        
    Returns:
        tuple: (selected_sources, selected_clusters, selected_AGN) stacked tables with recentered coords and window info.
    """

    coords = SkyCoord(ra=data_Xamin['EXT_RA']*u.deg, dec=data_Xamin['EXT_DEC']*u.deg)
    half_size_deg = (window_size_arcmin / 60) / 2
    
    selected_src = [] # Liste pour stocker toutes les lignes sélectionnées
    selected_clusters = []
    selected_AGN = []
    
    for window_num, id in enumerate(list_ID_Xamin):

        # Coordonnees du centre de la fenetre
        ra_center  = data_Xamin[data_Xamin['ID_Xamin'] == id]['EXT_RA']
        dec_center = data_Xamin[data_Xamin['ID_Xamin'] == id]['EXT_DEC']
        center = SkyCoord(ra_center*u.deg, dec_center*u.deg)

        delta_ra  = np.abs(coords.ra.deg - ra_center)
        delta_dec = np.abs(coords.dec.deg - dec_center)
        delta_ra  = np.minimum(delta_ra, 360 - delta_ra)
        
        mask_Xamin_in_window = (delta_ra < half_size_deg) & (delta_dec < half_size_deg)

        # Pour les sources dans cette fenêtre, calculer leur distance au centre
        sources_in_window = data_Xamin[mask_Xamin_in_window]
        
        if len(sources_in_window) > 0:
            sources_coords = SkyCoord(ra=sources_in_window['EXT_RA']*u.deg, dec=sources_in_window['EXT_DEC']*u.deg)
            separations = sources_coords.separation(center)

            # Ajouter la séparation comme colonne temporaire
            sources_in_window = sources_in_window.copy()
            sources_in_window['separation_deg'] = separations.deg
            sources_in_window['window'] = window_num

            # Recentrage des coordonnées Xamin
            for suffix in ['EXT', 'PNT', 'DBL', 'EPN']:
                ra_col = f"{suffix}_RA"
                dec_col = f"{suffix}_DEC"
                
                if ra_col in sources_in_window.colnames and dec_col in sources_in_window.colnames:
                    orig_coords = SkyCoord(ra=sources_in_window[ra_col]*u.deg, 
                                         dec=sources_in_window[dec_col]*u.deg)
                    dra, ddec = orig_coords.spherical_offsets_to(center)
                    sources_in_window[ra_col] = dra.to(u.deg).value
                    sources_in_window[dec_col] = ddec.to(u.deg).value

            sources_in_window.sort('separation_deg') # Trier par distance au centre

            selected_src.append(sources_in_window) # Ajouter aux lignes sélectionnées

        # CLUSTERS PART

        coords_clusters = SkyCoord(ra=data_clusters['R.A.']*u.deg, dec=data_clusters['Dec']*u.deg)

        delta_ra_clusters  = np.abs(coords_clusters.ra.deg - ra_center)
        delta_dec_clusters = np.abs(coords_clusters.dec.deg - dec_center)
        delta_ra_clusters  = np.minimum(delta_ra_clusters, 360 - delta_ra_clusters)
        
        mask_clusters_in_window = (delta_ra_clusters < half_size_deg) & (delta_dec_clusters < half_size_deg)

        clusters_in_window = data_clusters[mask_clusters_in_window]

        if len(clusters_in_window) > 0:
            sources_coords_clusters = SkyCoord(ra=clusters_in_window['R.A.']*u.deg, dec=clusters_in_window['Dec']*u.deg)
            separations = sources_coords_clusters.separation(center)
            
            # Ajouter la séparation comme colonne temporaire
            clusters_in_window = clusters_in_window.copy()
            clusters_in_window['separation_deg'] = separations.deg
            clusters_in_window['window'] = window_num

            ra_col = "R.A."
            dec_col = "Dec"
            
            if ra_col in clusters_in_window.colnames and dec_col in clusters_in_window.colnames:
                orig_coords = SkyCoord(ra=clusters_in_window[ra_col]*u.deg, 
                                        dec=clusters_in_window[dec_col]*u.deg)
                dra, ddec = orig_coords.spherical_offsets_to(center)
                clusters_in_window[ra_col] = dra.to(u.deg).value
                clusters_in_window[dec_col] = ddec.to(u.deg).value

            clusters_in_window.sort('separation_deg') # Trier par distance au centre
            clusters_in_window = clusters_in_window[cluster_columns_to_keep]

            selected_clusters.append(clusters_in_window) # Ajouter aux lignes sélectionnées

        
        # AGN PART

        coords_AGN = SkyCoord(ra=data_AGN['ra_mag_gal']*u.deg, dec=data_AGN['dec_mag_gal']*u.deg)

        delta_ra_AGN  = np.abs(coords_AGN.ra.deg - ra_center)
        delta_dec_AGN = np.abs(coords_AGN.dec.deg - dec_center)
        delta_ra_AGN  = np.minimum(delta_ra_AGN, 360 - delta_ra_AGN)
        
        mask_AGN_in_window = (delta_ra_AGN < half_size_deg) & (delta_dec_AGN < half_size_deg)

        AGN_in_window = data_AGN[mask_AGN_in_window]

        if len(AGN_in_window) > 0:
            sources_coords_AGN = SkyCoord(ra=AGN_in_window['ra_mag_gal']*u.deg, dec=AGN_in_window['dec_mag_gal']*u.deg)
            separations = sources_coords_AGN.separation(center)
            
            # Ajouter la séparation comme colonne temporaire
            AGN_in_window = AGN_in_window.copy()
            AGN_in_window['separation_deg'] = separations.deg
            AGN_in_window['window'] = window_num

            ra_col = "ra_mag_gal"
            dec_col = "dec_mag_gal"
            
            if ra_col in AGN_in_window.colnames and dec_col in AGN_in_window.colnames:
                orig_coords = SkyCoord(ra=AGN_in_window[ra_col]*u.deg, 
                                        dec=AGN_in_window[dec_col]*u.deg)
                dra, ddec = orig_coords.spherical_offsets_to(center)
                AGN_in_window[ra_col] = dra.to(u.deg).value
                AGN_in_window[dec_col] = ddec.to(u.deg).value

            AGN_in_window.sort('separation_deg') # Trier par distance au centre
            AGN_in_window = AGN_in_window[AGN_columns_to_keep]

            selected_AGN.append(AGN_in_window) # Ajouter aux lignes sélectionnées


    # Ajouter toutes les lignes sélectionnées à la table de sortie
    list_windows = vstack(selected_src)
    info_clusters = vstack(selected_clusters)
    info_AGN = vstack(selected_AGN)

    return list_windows, info_clusters, info_AGN











def GardeFenestronsSousPeuples(list_windows, info_clusters, info_AGN, max_Xamin_par_fenestron):
    """
    Filtre trois tables Astropy connexes pour ne conserver que les fenestrons (windows)
    ayant un nombre de sources Xamin <= max_Xamin_par_fenestron.

    La fonction opère en 3 étapes :
    1. Calcule le nombre de sources par fenestron dans list_windows
    2. Identifie les fenestrons sous-peuplés (<= max_Xamin_par_fenestron)
    3. Filtre les trois tables en conservant uniquement ces fenestrons

    Parameters
    ----------
    list_windows : astropy.table.Table
        Table principale contenant les sources Xamin avec colonne 'window'
    info_clusters : astropy.table.Table
        Table d'amas associée avec colonne 'window'
    info_AGN : astropy.table.Table
        Table d'AGN associée avec colonne 'window'
    max_Xamin_par_fenestron : int
        Seuil maximal de sources Xamin par fenestron

    Returns
    -------
    tuple (astropy.table.Table, astropy.table.Table, astropy.table.Table)
        - list_windows filtrée
        - info_clusters filtrée
        - info_AGN filtrée

    Notes
    -----
    - Les trois tables en entrée doivent avoir une colonne 'window'
    - Le filtrage est basé exclusivement sur le comptage dans list_windows
    - Les relations entre tables sont préservées dans le résultat
    """
    
    windows, counts = np.unique(list_windows['window'], return_counts=True) # Compte le nombre de sources par fenestron
    
    valid_windows = windows[counts <= max_Xamin_par_fenestron]

    mask_Xamin = np.isin(list_windows['window'], valid_windows)
    filtered_list_windows = list_windows[mask_Xamin]
    
    mask_amas = np.isin(info_clusters['window'], valid_windows)
    filtered_info_clusters = info_clusters[mask_amas]

    mask_AGN = np.isin(info_AGN['window'], valid_windows)
    filtered_info_AGN = info_AGN[mask_AGN]

    return filtered_list_windows, filtered_info_clusters, filtered_info_AGN





def CompteSourcesParFenetres(list_windows_type):
    counter_sources = Counter(list_windows_type['window']) # Compte les occurrences de chaque valeur de 'window'
    most_common_window, max_count_sources = counter_sources.most_common(1)[0] # Trouve la valeur avec le max d'occurrences
    return max_count_sources






def vstack_prealloc(tables):
    """
    Efficiently stack a list of Astropy Tables by preallocating arrays.
    
    Args:
        tables (list of Table): List of Astropy Tables to stack.
        
    Returns:
        Table: A single Astropy Table containing all rows from input tables.
    """
    if not tables:
        return Table()
    
    # Calculer la taille totale
    total_len = sum(len(t) for t in tables)
    colnames = tables[0].colnames
    
    # Pré-allouer un dictionnaire de tableaux numpy
    concatenated = {}
    for col in colnames:
        dtype = tables[0][col].dtype
        concatenated[col] = np.empty(total_len, dtype=dtype)
    
    # Remplir les tableaux
    idx = 0
    for table in tables:
        for col in colnames:
            concatenated[col][idx:idx+len(table)] = table[col]
        idx += len(table)
    
    return Table(concatenated)









def random_rotations_and_mirror(list_windows, info_clusters, info_AGN, NumberOfRotations):
    """
    Applique des rotations aléatoires et des symétries miroir aux coordonnées RA/Dec
    avec optimisation CPU utilisant NumPy et Numba.
    
    Paramètres :
        list_windows : Table contenant les sources
        info_clusters : Table contenant les clusters
        NumberOfRotations : nombre de rotations à générer par fenêtre
        
    Retour :
        list_windows_augm : table des sources augmentées
        info_clusters_augm : table des clusters augmentés
    """
    
    coord_suffixes = ['EXT', 'PNT', 'DBL', 'EPN']
    
    # Pré-calculer les colonnes de coordonnées à traiter
    coord_cols = []
    for suffix in coord_suffixes:
        ra_col = f"{suffix}_RA"
        dec_col = f"{suffix}_DEC"
        if ra_col in list_windows.colnames and dec_col in list_windows.colnames:
            coord_cols.append((ra_col, dec_col))
    
    unique_windows = np.unique(list_windows['window'])
    max_window_num = max(list_windows['window']) if len(list_windows) > 0 else 0
    

    # //////////// Préparer les résultats ////////////

    inclure_originaux = False

    if(inclure_originaux): # (inclure les originaux)

        augmented_windows = [list_windows.copy()]

        # ***** AMAS *****
        if len(info_clusters) > 0:
            augmented_cluster = [info_clusters.copy()]
            has_cluster_data = True
        else:
            augmented_cluster = []
            has_cluster_data = False
        
        # ***** AGN *****
        if len(info_AGN) > 0:
            augmented_AGN = [info_AGN.copy()]
            has_AGN_data = True
        else:
            augmented_AGN = []
            has_AGN_data = False
    
    else: # (ne pas inclure les originaux)
        augmented_windows = []
        augmented_cluster = []
        augmented_AGN = []
        
        has_cluster_data = len(info_clusters) > 0
        has_AGN_data = len(info_AGN) > 0

    # Fonctions optimisées avec Numba
    @numba.njit(fastmath=True)
    def apply_rotation(ra, dec, angle_rad):
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)
        cos_t = np.cos(angle_rad)
        sin_t = np.sin(angle_rad)
        x_rot = ra_rad * cos_t - dec_rad * sin_t
        y_rot = ra_rad * sin_t + dec_rad * cos_t
        return np.rad2deg(x_rot), np.rad2deg(y_rot)
    
    @numba.njit(fastmath=True)
    def apply_mirror(ra):
        return -ra
    
    # Générer tous les angles de rotation à l'avance
    rng = np.random.default_rng()
    all_rotation_angles = rng.uniform(0, 2*np.pi, size=(len(unique_windows), NumberOfRotations))
    
    # Traitement par fenêtre avec barre de progression
    for i, win in enumerate(unique_windows):
        win_mask = list_windows['window'] == win
        sub_src = list_windows[win_mask]
        
        # ***** AMAS *****
        if has_cluster_data:
            mask_culster = info_clusters['window'] == win
            sub_cluster = info_clusters[mask_culster]
            has_cluster = len(sub_cluster) > 0
        else:
            has_cluster = False
        
        # ***** AGN *****
        if has_AGN_data:
            mask_AGN = info_AGN['window'] == win
            sub_AGN = info_AGN[mask_AGN]
            has_AGN = len(sub_AGN) > 0
        else:
            has_AGN = False

        # /////// ROTATION ALEATOIRE ///////
        angles = all_rotation_angles[i]
        for angle in angles:
            # Créer une nouvelle table pour la rotation
            rotated_src = sub_src.copy()
            
            # Appliquer la rotation à toutes les paires de coordonnées
            for ra_col, dec_col in coord_cols:
                ra = rotated_src[ra_col]
                dec = rotated_src[dec_col]
                new_ra, new_dec = apply_rotation(ra, dec, angle)
                rotated_src[ra_col] = new_ra
                rotated_src[dec_col] = new_dec
            
            # Mettre à jour le numéro de fenêtre
            rotated_src['window'] = max_window_num + 1
            augmented_windows.append(rotated_src)
            
            # ***** AMAS *****
            if has_cluster:
                rotated_cluster = sub_cluster.copy()
                new_ra, new_dec = apply_rotation(rotated_cluster['R.A.'][0], rotated_cluster['Dec'][0], angle)
                rotated_cluster['R.A.'] = new_ra
                rotated_cluster['Dec'] = new_dec
                rotated_cluster['window'] = max_window_num + 1
                augmented_cluster.append(rotated_cluster)

            # ***** AGN *****
            if has_AGN:
                rotated_AGN = sub_AGN.copy()
                new_ra, new_dec = apply_rotation(rotated_AGN['ra_mag_gal'][0], rotated_AGN['dec_mag_gal'][0], angle)
                rotated_AGN['ra_mag_gal'] = new_ra
                rotated_AGN['dec_mag_gal'] = new_dec
                rotated_AGN['window'] = max_window_num + 1
                augmented_AGN.append(rotated_AGN)
            
            max_window_num += 1
        
        # /////// MIROIR ///////

        # Miroir (RA -> -RA)
        mirrored_src = sub_src.copy()
        for ra_col, _ in coord_cols:
            mirrored_src[ra_col] = apply_mirror(mirrored_src[ra_col])
        
        mirrored_src['window'] = max_window_num + 1
        augmented_windows.append(mirrored_src)
        
        if has_cluster:
            mirrored_cluster = sub_cluster.copy()
            mirrored_cluster['R.A.'] = apply_mirror(mirrored_cluster['R.A.'])
            mirrored_cluster['window'] = max_window_num + 1
            augmented_cluster.append(mirrored_cluster)

        if has_AGN:
            mirrored_AGN = sub_AGN.copy()
            mirrored_AGN['ra_mag_gal'] = apply_mirror(mirrored_AGN['ra_mag_gal'])
            mirrored_AGN['window'] = max_window_num + 1
            augmented_AGN.append(mirrored_AGN)

        max_window_num += 1
    
    # Concaténer tous les résultats
    list_windows_augm = vstack_prealloc(augmented_windows)
    info_clusters_augm = vstack_prealloc(augmented_cluster) if augmented_cluster else info_clusters.copy(copy_data=False)
    info_AGN_augm = vstack_prealloc(augmented_AGN) if augmented_AGN else info_AGN.copy(copy_data=False)

    return list_windows_augm, info_clusters_augm, info_AGN_augm











def plot_augmentations(list_windows_augm, original_window_id, max_original_window_id, NumberOfRotations):
    """
    Version robuste qui garantit l'affichage de tous les points avec :
    - Ajustement automatique des limites
    - Vérification des données
    - Meilleure visibilité des points
    """
    # ===== 1. Vérification des données d'entrée =====
    if original_window_id not in list_windows_augm['window']:
        print(f"Attention : la fenêtre {original_window_id} n'existe pas dans les données")
        return
    
    # ===== 2. Sélection des points =====
    original = list_windows_augm[list_windows_augm['window'] == original_window_id]
    
    # Calcul des IDs des versions augmentées
    n_augmentations = NumberOfRotations + 1  # +1 pour le miroir
    first_aug_id = max_original_window_id + 1 + original_window_id * n_augmentations
    augmented_ids = list(range(first_aug_id, first_aug_id + n_augmentations))
    
    # ===== 3. Collecte de TOUS les points à afficher =====
    all_points = {'ra': [], 'dec': []}
    
    # Points originaux
    all_points['ra'].extend(original['PNT_RA'])
    all_points['dec'].extend(original['PNT_DEC'])
    
    # Points augmentés
    augmented_data = []
    for i, win_id in enumerate(augmented_ids):
        data = list_windows_augm[list_windows_augm['window'] == win_id]
        if len(data) > 0:
            augmented_data.append((i, win_id, data))
            all_points['ra'].extend(data['PNT_RA'])
            all_points['dec'].extend(data['PNT_DEC'])
    
    # ===== 4. Calcul des limites intelligentes =====
    '''def compute_limits(values, margin_factor=0.2):
        min_val = np.nanmin(values)
        max_val = np.nanmax(values)
        span = max_val - min_val
        return (min_val - margin_factor*span, max_val + margin_factor*span)
    
    xlim = compute_limits(all_points['ra'])
    ylim = compute_limits(all_points['dec'])'''
    
    # ===== 5. Création du plot =====
    plt.figure(figsize=(4, 4))
    
    # Affichage original
    plt.scatter(original['PNT_RA'], original['PNT_DEC'], 
               c='blue', s=60, label='Original', alpha=0.8, edgecolors='w', linewidth=0.5)
    
    # Affichage des rotations
    rotation_colors = ['#FF7F0E', '#D62728', '#9467BD', '#8C564B']  # Couleurs distinctes
    for i, win_id, data in augmented_data[:-1]:  # Tous sauf le dernier (miroir)
        plt.scatter(data['PNT_RA'], data['PNT_DEC'],
                   c=rotation_colors[i % len(rotation_colors)], s=40,
                   alpha=0.7, label=f'Rotation {i+1}')
    
    # Affichage du miroir (dernier élément)
    if augmented_data and len(augmented_data[-1][2]) > 0:
        plt.scatter(augmented_data[-1][2]['PNT_RA'], augmented_data[-1][2]['PNT_DEC'],
                   c='#2CA02C', s=40, marker='s', alpha=0.7, label='Miroir')
    
    plt.axhline(-WINDOW_SIZE_ARCMIN/60/2, color='gray', linestyle='--')
    plt.axhline(WINDOW_SIZE_ARCMIN/60/2, color='gray', linestyle='--')
    plt.axvline(-WINDOW_SIZE_ARCMIN/60/2, color='gray', linestyle='--')
    plt.axvline(WINDOW_SIZE_ARCMIN/60/2, color='gray', linestyle='--')

    # ===== 6. Configuration finale =====
    plt.xlim(-WINDOW_SIZE_ARCMIN/60/1.5, WINDOW_SIZE_ARCMIN/60/1.5)
    plt.ylim(-WINDOW_SIZE_ARCMIN/60/1.5, WINDOW_SIZE_ARCMIN/60/1.5)
    plt.xlabel('Right Ascension (deg)', fontsize=12)
    plt.ylabel('Declination (deg)', fontsize=12)
    plt.title(f'Window {original_window_id} with {NumberOfRotations} rotations + mirror', pad=20)
    
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Ajout d'un texte d'information
    info_text = f"Total points: Original={len(original)}"
    for i, (_, _, data) in enumerate(augmented_data):
        info_text += f", Aug{i+1}={len(data)}"
    plt.annotate(info_text, xy=(0.5, 0.02), xycoords='axes fraction',
                ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()





def compute_global_stats(data, selected_columns, global_stats=None):
    """
    Calcule les statistiques globales et met à jour un dictionnaire existant
    
    Args:
        data: Tableau numpy structuré ou Astropy Table
        selected_columns: Liste des colonnes à analyser
        global_stats: Dictionnaire existant à mettre à jour (optionnel)
        
    Returns:
        Dictionnaire mis à jour avec les statistiques
    """
    if global_stats is None:
        global_stats = {}

        for col in selected_columns:
            if col in data.dtype.names:
                values = data[col][~np.isnan(data[col])]
                if len(values) > 0:
                    global_stats[col] = {
                        'min': np.min(values),
                        'max': np.max(values),
                        'log_min': np.log10(np.min(values[values > 0]) if np.any(values > 0) else 1e-10)
                    }

        return global_stats
    
    for col in selected_columns:
        # Extraction des valeurs non-NaN
        values = data[col][~np.isnan(data[col])]
        
        if len(values) > 0:
            # Calcul des nouvelles statistiques
            current_min = np.min(values)
            current_max = np.max(values)
            current_log_min = np.log10(np.min(values[values > 0]) if np.any(values > 0) else 1e-10)
            
            # Mise à jour des valeurs existantes
            global_stats[col]['min'] = min(global_stats[col]['min'], current_min)
            global_stats[col]['max'] = max(global_stats[col]['max'], current_max)
            global_stats[col]['log_min'] = min(global_stats[col]['log_min'], current_log_min)
    
    return global_stats










def save_current_state(windows, clusters, agn, output_dir, chunk_num):
    """Sauvegarde l'état courant dans des fichiers FITS"""
    windows.write(os.path.join(output_dir, f"windows_{chunk_num:04d}.fits"), overwrite=True)
    
    if len(clusters) > 0:
        clusters.write(os.path.join(output_dir, f"clusters_{chunk_num:04d}.fits"), overwrite=True)
    
    if len(agn) > 0:
        agn.write(os.path.join(output_dir, f"agn_{chunk_num:04d}.fits"), overwrite=True)











def process_rotations_in_chunks(list_windows, info_clusters, info_AGN, 
                              total_rotations, chunk_size, output_dir,
                              stats_Xamin, stats_input_clusters, stats_input_AGN):
    """
    Applique les rotations par chunks et sauvegarde les résultats sans retourner de valeur
    
    Paramètres :
        list_windows: Table Astropy des sources
        info_clusters: Table Astropy des clusters
        info_AGN: Table Astropy des AGN
        total_rotations: Nombre total de rotations à appliquer
        chunk_size: Nombre de rotations par chunk
        output_dir: Répertoire de sortie pour les fichiers FITS
    """
    # Création du répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)  # Recrée le répertoire vide
    for filename in os.listdir(output_dir):
        if filename.endswith('.fits'): 
            os.remove(os.path.join(output_dir, filename))

    # Initialisation des données courantes
    current_windows = list_windows.copy()
    current_clusters = info_clusters.copy()
    current_agn = info_AGN.copy()
    file_counter = 1 # Compteur de fichiers

    save_current_state(
            current_windows,
            current_clusters,
            current_agn,
            output_dir,
            file_counter
        )
    file_counter += 1

    # Boucle principale
    for start_rot in range(0, total_rotations, chunk_size):
        actual_chunk_size = min(chunk_size, total_rotations - start_rot)
        print(f"Processing rotations {start_rot+1}-{start_rot+actual_chunk_size}/{total_rotations}")
        
        # Application des rotations
        rotated_windows, rotated_clusters, rotated_agn = \
            random_rotations_and_mirror(
                list_windows,  # On part toujours des originaux
                info_clusters,
                info_AGN,
                NumberOfRotations=actual_chunk_size
            )
                
        # Mise a jour des statistiques
        stats_Xamin          = compute_global_stats(rotated_windows, SELECTED_COLUMNS_Xamin, stats_Xamin)
        stats_input_clusters = compute_global_stats(rotated_clusters, SELECTED_COLUMNS_input_clusters, stats_input_clusters)
        stats_input_AGN      = compute_global_stats(rotated_agn, SELECTED_COLUMNS_input_AGN, stats_input_AGN) 

        # Sauvegarde des résultats
        save_current_state(
            rotated_windows,
            rotated_clusters,
            rotated_agn,
            output_dir,
            file_counter
        )
        
        file_counter += 1
    
    print(f"Traitement terminé. Résultats sauvegardés dans {output_dir}")
    return stats_Xamin, stats_input_clusters, stats_input_AGN













def process_and_save_chunks(directory, output_path,
                          stats_Xamin, stats_input_clusters, stats_input_AGN,
                          max_sources, max_clusters, max_agn):
    """
    Traite les fichiers par chunks et sauvegarde dans un seul X.txt en streaming
    
    Args:
        directory: Répertoire contenant les fichiers windows_*, clusters_*, agn_*
        output_path: Chemin complet du fichier de sortie X.txt
        stats_Xamin: Statistiques pour la discrétisation des windows
        stats_input_clusters: Statistiques pour les clusters
        stats_input_AGN: Statistiques pour les AGN
        max_sources: Nombre max de sources
        max_clusters: Nombre max de clusters
        max_agn: Nombre max d'AGN
    """
    
    # Création du répertoire de sortie si besoin
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ouverture du fichier en mode write (efface si existe déjà)
    with open(output_path, 'w') as f:
        # Trouver tous les chunks disponibles
        chunk_files = sorted([f for f in os.listdir(directory) if f.startswith('windows_')])
        
        for chunk_file in chunk_files:
            chunk_num = chunk_file.split('_')[1].split('.')[0]
            print(f"Traitement du chunk {chunk_num}...")
            
            # Chargement des fichiers
            current_windows = Table.read(os.path.join(directory, f'windows_{chunk_num}.fits'))
            current_clusters = Table.read(os.path.join(directory, f'clusters_{chunk_num}.fits')) if os.path.exists(os.path.join(directory, f'clusters_{chunk_num}.fits')) else Table()
            current_agn = Table.read(os.path.join(directory, f'agn_{chunk_num}.fits')) if os.path.exists(os.path.join(directory, f'agn_{chunk_num}.fits')) else Table()
            
            # Application des transformations
            
            #print("\n=== Première ligne ===")
            #for col in current_windows.colnames:
            #    print(f"{col} : {current_windows[col][0]}")
            
            windows = discretise_et_complete(current_windows, current_windows, 
                                          int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), 
                                          stats_Xamin, SELECTED_COLUMNS_Xamin, 
                                          use_log_scale_Xamin, PAD_TOKEN, max_sources)
            
            ClustersInWindows = discretise_et_complete(current_windows, current_clusters, 
                                                    int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX),
                                                    stats_input_clusters, SELECTED_COLUMNS_input_clusters,
                                                    use_log_scale_input_clusters, PAD_TOKEN, max_clusters)
            
            AGNInWindows = discretise_et_complete(current_windows, current_agn,
                                                int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX),
                                                stats_input_AGN, SELECTED_COLUMNS_input_AGN,
                                                use_log_scale_input_AGN, PAD_TOKEN, max_agn)
            
            # Combinaison et écriture directe dans le fichier
            X = combine_and_flatten_with_special_tokens(windows, ClustersInWindows, AGNInWindows)
            np.savetxt(f, X, fmt='%d')
            
            # Nettoyage mémoire explicite
            del current_windows, current_clusters, current_agn, windows, ClustersInWindows, AGNInWindows, X
    
    print(f"Tous les chunks ont été traités et sauvegardés dans {output_path}")
        















def discretise_et_complete(data_ref, data, n_bins, global_stats, selected_columns, log_scale_flags, PAD_TOKEN, max_sources):
    """
    Discrétise les colonnes numériques sélectionnées pour chaque fenêtre, avec normalisation 
    linéaire ou logarithmique selon les indicateurs fournis.

    Pour chaque fenêtre présente dans `data_ref`, on récupère les sources correspondantes dans `data`,
    on extrait les colonnes `selected_columns`, on les normalise (log ou lin), puis on les discrétise
    en `n_bins` entiers. Si une valeur est invalide (NaN, inf, <= 0 en log), elle est remplacée 
    par `PAD_TOKEN`. Les fenêtres sont complétées jusqu'à `max_sources` par padding si nécessaire.
    
    Paramètres :
    - data_ref : Table de référence contenant toutes les fenêtres attendues.
    - data : Table contenant les données à discrétiser.
    - n_bins : Nombre total de classes pour la discrétisation.
    - global_stats : Dictionnaire contenant min, max (et log_min) pour chaque colonne.
    - selected_columns : Liste des colonnes à traiter.
    - log_scale_flags : Liste booléenne indiquant pour chaque colonne si l'échelle log est à utiliser.
    - PAD_TOKEN : Valeur à utiliser pour le padding ou les données manquantes/invalides.
    - max_sources : Nombre maximum de sources par fenêtre (pour uniformiser la taille des sorties).

    Retour :
    - Un tableau numpy de forme (n_fenêtres, max_sources, n_colonnes), contenant les valeurs discrétisées
      ou `PAD_TOKEN` pour les entrées manquantes.
    """
    windows = []
    data_windows = set(np.unique(data['window']))
    ref_windows = np.unique(data_ref['window'])
    pad_length = len(selected_columns)
    empty_padding = [[PAD_TOKEN] * pad_length for _ in range(max_sources)]
    n_bins_minus_1 = n_bins - 1

    # Pré-calcul des paramètres de normalisation
    norm_params = []
    for col, use_log in zip(selected_columns, log_scale_flags):
        stats = global_stats.get(col)
        if stats is None:
            norm_params.append(('none',))
        elif use_log:
            log_min = stats['log_min']
            log_range = np.log10(stats['max']) - log_min
            norm_params.append(('log', log_min, log_range))
        else:
            min_val = stats['min']
            range_val = stats['max'] - min_val
            norm_params.append(('linear', min_val, range_val))

    # Parcours des fenêtres
    for window_id in ref_windows:
        if data_ref is not data and window_id not in data_windows:
            windows.append(empty_padding.copy())
            continue

        win_data = data[data['window'] == window_id]
        win_features = []

        for src in win_data:
            src_features = []
            for col_idx, col in enumerate(selected_columns):
                if col not in win_data.colnames:
                    src_features.append(PAD_TOKEN)
                    continue

                val = src[col]
                if np.isnan(val) or np.isinf(val):
                    src_features.append(PAD_TOKEN)
                    continue

                if (np.ma.is_masked(val) or isinstance(val, np.ma.core.MaskedConstant) or not np.isfinite(val)):
                    src_features.append(PAD_TOKEN)
                    #print("Pb derriere nous?")
                    continue

                norm_type, *params = norm_params[col_idx]

                if norm_type == 'log':
                    if val <= 0:
                        src_features.append(PAD_TOKEN)
                        continue
                    log_val = np.log10(val)
                    norm_val = (log_val - params[0]) / params[1]
                elif norm_type == 'linear':
                    norm_val = (val - params[0]) / params[1]
                else:
                    norm_val = val  # aucune normalisation

                discretized_val = int(np.clip(norm_val * n_bins_minus_1, 0, n_bins_minus_1))
                src_features.append(discretized_val)

            win_features.append(src_features)

        # Padding
        num_pad = max_sources - len(win_features)
        if num_pad > 0:
            padding = [[PAD_TOKEN] * pad_length for _ in range(num_pad)]
            win_features.extend(padding)

        windows.append(win_features)

    return np.array(windows)







def verifier_discretisation(data, selected_columns, log_scale_flags, column_to_check, n_bins, max_sources, stats_Xamin):
    """
    Discretizes selected numerical columns for each window, using either linear or logarithmic 
    scaling based on the provided flags.

    For each window in `data_ref`, this function looks up corresponding sources in `data`, extracts 
    the specified `selected_columns`, normalizes the values (using log or linear scaling), and 
    discretizes them into `n_bins` integer bins. Invalid values (e.g., NaN, inf, or non-positive 
    values when log-scaling) are replaced with `PAD_TOKEN`. Windows are padded with dummy rows 
    up to `max_sources` to ensure uniform output shape.

    Parameters:
    - data_ref: Reference table containing the full list of windows to process.
    - data: Table containing the actual data to be discretized.
    - n_bins: Total number of discrete bins.
    - global_stats: Dictionary with normalization stats per column (min, max, and log_min).
    - selected_columns: List of columns to process.
    - log_scale_flags: Boolean list indicating whether to use log scale per column.
    - PAD_TOKEN: Value used for padding and invalid/missing data.
    - max_sources: Maximum number of sources per window (for fixed-size output).

    Returns:
    - A NumPy array of shape (n_windows, max_sources, n_columns), containing discretized values 
      or `PAD_TOKEN` where data is missing or invalid.
    """
    if column_to_check not in selected_columns:
        print(f"\nErreur: La colonne '{column_to_check}' n'est pas dans selected_columns\n")
        return
    
    #print(f"\nStatistiques globales pour '{column_to_check}':")
    #print(f"Min: {stats_Xamin[column_to_check]['min']}")
    #print(f"Max: {stats_Xamin[column_to_check]['max']}")
    #print(f"Log min: {stats_Xamin[column_to_check]['log_min']}")
    
    # Étape 2: Discrétiser les données
    discretized = discretise_et_complete(data, data, n_bins, stats_Xamin, 
                                        selected_columns, log_scale_flags, 
                                        PAD_TOKEN, max_sources)
    
    # Trouver l'index de la colonne à vérifier
    col_idx = selected_columns.index(column_to_check)
    
    # Extraire les valeurs originales et discrétisées
    original_values = data[column_to_check]
    original_values = original_values[~np.isnan(original_values)]
    
    discretized_values = discretized[:, :, col_idx].flatten()
    discretized_values = discretized_values[discretized_values != PAD_TOKEN]
    
    # Tracer les histogrammes
    plt.figure(figsize=(18, 6))
    
    # Couleurs et styles personnalisés
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Bleu, Orange, Vert
    edge_colors = ['#0c4a8e', '#cc5500', '#1a5e1a']
    alpha = 0.8
    histtype = 'bar'  # ou 'stepfilled' pour un look différent
    
    # 1. Histogramme des valeurs originales (linéaire)
    plt.subplot(1, 3, 1)
    plt.hist(original_values, bins=50, color=colors[0], edgecolor=edge_colors[0],
             alpha=alpha, histtype=histtype, linewidth=1.5)
    plt.xlabel(column_to_check, fontsize=12)
    plt.title('Valeurs originales (linéaire)', fontsize=14, pad=20)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Histogramme des valeurs originales (log si applicable)
    plt.subplot(1, 3, 2)
    if log_scale_flags[col_idx]:
        positive_vals = original_values[original_values > 0]
        plt.hist(np.log10(positive_vals), bins=50, color=colors[1], edgecolor=edge_colors[1],
                 alpha=alpha, histtype=histtype, linewidth=1.5)
        plt.xlabel(f'log({column_to_check})', fontsize=12)
        plt.title('Valeurs originales (log)', fontsize=14, pad=20)
    else:
        plt.hist(original_values, bins=50, color=colors[1], edgecolor=edge_colors[1],
                 alpha=alpha, histtype=histtype, linewidth=1.5)
        plt.xlabel(column_to_check, fontsize=12)
        plt.title('Valeurs originales (linéaire)', fontsize=14, pad=20)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 3. Histogramme des valeurs discrétisées
    plt.subplot(1, 3, 3)
    plt.hist(discretized_values, bins=min(n_bins, 50), color=colors[2], edgecolor=edge_colors[2],
             alpha=alpha, histtype=histtype, linewidth=1.5)
    plt.xlabel(f'{column_to_check} discrétisée', fontsize=12)
    plt.title(f'Valeurs discrétisées ({n_bins} bins)', fontsize=14, pad=20)
    plt.ylabel('Fréquence', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Catalog/results/{name_dir}/hist_{column_to_check}.png')
    
    # Afficher quelques statistiques
    #print(f"\nValeurs originales (non-NaN): {len(original_values)}")
    #print(f"Valeurs discrétisées (non-PAD): {len(discretized_values)}")
    #print(f"Plage des valeurs discrétisées: {np.min(discretized_values)} à {np.max(discretized_values)}")







def combine_and_flatten_with_special_tokens(windows_Xamin, windows_input_cluster, windows_input_AGN, 
                                            cls_token = CLS_TOKEN, sep_token = SEP_TOKEN, sep_amas_token = SEP_AMAS, sep_agn_token = SEP_AGN):
    """
    Returns 2D array of shape (n_windows, max_sources*n_features_Xamin + max_clusters*n_features_input_cluster + max_agn*n_features_input_agn + 2)
    """
    cls_token      = np.array(cls_token).flatten()
    sep_token      = np.array(sep_token).flatten()
    sep_amas_token = np.array(sep_amas_token).flatten()
    sep_agn_token  = np.array(sep_agn_token).flatten()

    if len(windows_Xamin) != len(windows_input_cluster) or len(windows_input_AGN) != len(windows_input_cluster):
        raise ValueError("Les trois listes de fenêtres doivent avoir la même longueur.")

    result = []
    for win_xamin, win_input_cluster, win_input_AGN in zip(windows_Xamin, windows_input_cluster, windows_input_AGN):
        win_xamin = np.array(win_xamin)
        win_input_cluster = np.array(win_input_cluster)
        win_input_AGN = np.array(win_input_AGN)
        seq = []
        seq.extend(cls_token)
        seq.extend(win_xamin.flatten())
        seq.extend(sep_amas_token)
        seq.extend(win_input_cluster.flatten())
        seq.extend(sep_agn_token)
        seq.extend(win_input_AGN.flatten())
        seq.extend(sep_token)
        result.append(seq)

    return np.array(result)






def convert_numpy_types(obj):
    """Convertit les types NumPy en types natifs Python pour la sérialisation JSON"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj




def verify_table(table, table_name):
    print(f"\n======= Verifying {table_name} =======")
    print(f"- Number of rows: {len(table)}")
    
    if hasattr(table, 'mask'):
        # Pour les tables Astropy
        masked_rows = 0
        total_masked = 0
        col_masks = {}
        
        # Vérifier chaque colonne
        for col in table.colnames:
            if hasattr(table[col], 'mask'):
                col_mask = table[col].mask
                if isinstance(col_mask, np.ndarray):
                    col_masked = np.sum(col_mask)
                    if col_masked > 0:
                        col_masks[col] = col_masked
                        total_masked += col_masked
                        masked_rows = max(masked_rows, np.sum(col_mask))
        
        if col_masks:
            print(f"- Contains masked values")
            print(f"- Number of rows with masked values: {masked_rows}/{len(table)} "
                  f"({masked_rows/len(table):.1%})")
            print(f"- Total masked values: {total_masked}")
            
            print("\nMasked values per column:")
            for col, count in col_masks.items():
                print(f"  - {col}: {count} masked values")
        else:
            print("- No masked values detected (empty mask)")
    else:
        print("- No masked values detected")
    
    print(f"- Columns: {table.colnames}")