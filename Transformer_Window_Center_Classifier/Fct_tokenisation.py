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

from Constantes import LIM_FLUX_CLUSTER, LIM_FLUX_AGN
from Constantes import SEARCH_RADIUS_CLUSTER, SEARCH_RADIUS_AGN
from Constantes import WINDOW_SIZE_ARCMIN, NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import SELECTED_COLUMNS_Xamin, use_log_scale_Xamin, name_dir
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







def Batisseuse2Fenetres(data_Xamin, list_ID_Xamin, list_ID_Xamin_clusters,
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
    
    info_class = Table()
    info_class['window'] = np.array([], dtype=int)
    info_class['isCluster'] = np.array([], dtype=int) 
    
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
        if id in list_ID_Xamin_clusters:
            info_class.add_row([window_num, 1])  # Ajoute une ligne (window, isCluster)
        else:
            info_class.add_row([window_num, 0])

    # Ajouter toutes les lignes sélectionnées à la table de sortie
    list_windows = vstack(selected_src)

    return list_windows, info_class











def GardeFenestronsSousPeuples(list_windows, info_class, max_Xamin_par_fenestron):
    """
    Filtre les tables Astropy connexes list_windows, info_class pour ne conserver que les fenestrons (windows)
    ayant un nombre de sources Xamin <= max_Xamin_par_fenestron.

    La fonction opère en 3 étapes :
    1. Calcule le nombre de sources par fenestron dans list_windows
    2. Identifie les fenestrons sous-peuplés (<= max_Xamin_par_fenestron)
    3. Filtre les deux tables en conservant uniquement ces fenestrons

    Parameters
    ----------
    list_windows : astropy.table.Table
        Table principale contenant les sources Xamin avec colonne 'window'
    info_class : astropy.table.Table
        Table indiquant la classe de la source centrale associée avec colonne 'window'
    max_Xamin_par_fenestron : int
        Seuil maximal de sources Xamin par fenestron

    Returns
    -------
    tuple (astropy.table.Table, astropy.table.Table, astropy.table.Table)
        - list_windows filtrée
        - info_class filtrée

    Notes
    -----
    - Les deux tables en entrée doivent avoir une colonne 'window'
    - Le filtrage est basé exclusivement sur le comptage dans list_windows
    - Les relations entre tables sont préservées dans le résultat
    """
    
    windows, counts = np.unique(list_windows['window'], return_counts=True) # Compte le nombre de sources par fenestron
    
    valid_windows = windows[counts <= max_Xamin_par_fenestron]

    mask_Xamin = np.isin(list_windows['window'], valid_windows)
    filtered_list_windows = list_windows[mask_Xamin]
    
    mask_amas = np.isin(info_class['window'], valid_windows)
    filtered_info_class = info_class[mask_amas]

    return filtered_list_windows, filtered_info_class






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









def random_rotations_and_mirror(list_windows, info_class, NumberOfRotations):
    """
    Applique des rotations aléatoires et des symétries miroir aux coordonnées RA/Dec
    avec optimisation CPU utilisant NumPy et Numba.
    
    Paramètres :
        list_windows : Table contenant les sources
        info_class : indiquant la classe de la source centrale
        NumberOfRotations : nombre de rotations à générer par fenêtre
        
    Retour :
        list_windows_augm : table des sources augmentées
        info_class_augm : table des classes augmentées
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

    inclure_originale = False

    if(inclure_originale): # (inclure la version originale)
        augmented_windows = [list_windows.copy()]
        info_class_augm = [info_class.copy()]
    else: 
        augmented_windows = []
        info_class_augm = []

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
    
    # Traitement par fenêtre 
    for i, win in enumerate(unique_windows):
        win_mask = list_windows['window'] == win
        sub_src = list_windows[win_mask]
        
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

            is_cluster = info_class[info_class['window'] == win]['isCluster'][0]
            info_class_augm.append((max_window_num + 1, is_cluster))
            
            max_window_num += 1
        
        # /////// MIROIR ///////

        # Miroir (RA -> -RA)
        mirrored_src = sub_src.copy()
        for ra_col, _ in coord_cols:
            mirrored_src[ra_col] = apply_mirror(mirrored_src[ra_col])
        
        mirrored_src['window'] = max_window_num + 1
        augmented_windows.append(mirrored_src)
        
        is_cluster = info_class[info_class['window'] == win]['isCluster'][0]
        info_class_augm.append((max_window_num + 1, is_cluster))

        max_window_num += 1
    
    # Concaténer tous les résultats
    list_windows_augm = vstack_prealloc(augmented_windows)

    #info_class_augm = np.array(info_class_augm)
    #info_class_augm = Table(info_class_augm, names=('window', 'isCluster'), dtype=('int', 'int'))
    info_class_augm = Table(rows=info_class_augm, names=('window', 'isCluster'), dtype=('int', 'int'))

    return list_windows_augm, info_class_augm











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











def save_current_state(windows, info_class, output_dir, chunk_num):
    """Sauvegarde l'état courant dans des fichiers FITS"""
    windows.write(os.path.join(output_dir, f"windows_{chunk_num:04d}.fits"), overwrite=True)
    
    info_class.write(os.path.join(output_dir, f"class_{chunk_num:04d}.fits"), overwrite=True)
    
    








def process_rotations_in_chunks(list_windows, info_class,
                              total_rotations, chunk_size, output_dir,
                              stats_Xamin):
    """
    Applique les rotations par chunks et sauvegarde les résultats sans retourner de valeur
    
    Paramètres :
        list_windows: Table Astropy des sources
        info_class: Table Astropy indiquant source centrale amas ou pas
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
    current_clusters = info_class.copy()
    file_counter = 1 # Compteur de fichiers

    save_current_state(
            current_windows,
            info_class,
            output_dir,
            file_counter
        )
    file_counter += 1

    # Boucle principale
    for start_rot in range(0, total_rotations, chunk_size):
        actual_chunk_size = min(chunk_size, total_rotations - start_rot)
        print(f"Processing rotations {start_rot+1}-{start_rot+actual_chunk_size}/{total_rotations}")
        
        # Application des rotations
        rotated_windows, rotated_info_class = \
            random_rotations_and_mirror(
                list_windows,  # On part toujours des originaux
                info_class,
                NumberOfRotations=actual_chunk_size
            )
                
        # Mise a jour des statistiques
        stats_Xamin = compute_global_stats(rotated_windows, SELECTED_COLUMNS_Xamin, stats_Xamin)

        # Sauvegarde des résultats
        save_current_state(
            rotated_windows,
            rotated_info_class,
            output_dir,
            file_counter
        )
        
        file_counter += 1
    
    print(f"Traitement terminé. Résultats sauvegardés dans {output_dir}")
    return stats_Xamin







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
    plt.savefig(f'/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/hist_{column_to_check}.png')
    
    # Afficher quelques statistiques
    #print(f"\nValeurs originales (non-NaN): {len(original_values)}")
    #print(f"Valeurs discrétisées (non-PAD): {len(discretized_values)}")
    #print(f"Plage des valeurs discrétisées: {np.min(discretized_values)} à {np.max(discretized_values)}")







def combine_and_flatten_with_special_tokens(windows_Xamin, info_class, 
                                            cls_token = CLS_TOKEN, sep_token = SEP_TOKEN, sep_amas_token = SEP_AMAS):
    """
    Returns 2D array of shape (n_windows, max_sources*n_features_Xamin + max_clusters*n_features_input_cluster + max_agn*n_features_input_agn + 2)
    """
    cls_token      = np.array(cls_token).flatten()
    sep_token      = np.array(sep_token).flatten()
    sep_amas_token = np.array(sep_amas_token).flatten()

    if len(windows_Xamin) != len(info_class):
        raise ValueError("Les trois listes de fenêtres doivent avoir la même longueur.")

    isCluster = info_class["isCluster"]

    result = []
    for win_xamin, srcClassFlag in zip(windows_Xamin, isCluster):
        win_xamin = np.array(win_xamin)
        srcClassFlag = np.array(srcClassFlag)
        seq = []
        seq.extend(cls_token)
        seq.extend(win_xamin.flatten())
        seq.extend(sep_amas_token)
        seq.extend(srcClassFlag.flatten())
        seq.extend(sep_token)
        result.append(seq)

    return np.array(result)




def process_and_save_chunks(directory, output_path, stats_Xamin, max_sources):
    """
    Traite les fichiers par chunks et sauvegarde dans un seul X.txt en streaming
    
    Args:
        directory: Répertoire contenant les fichiers windows_*, clusters_*, agn_*
        output_path: Chemin complet du fichier de sortie X.txt
        stats_Xamin: Statistiques pour la discrétisation des windows
        max_sources: Nombre max de sources
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
            current_info_class = Table.read(os.path.join(directory, f'class_{chunk_num}.fits'))
            
            # Application des transformations
            
            #print("\n=== Première ligne ===")
            #for col in current_windows.colnames:
            #    print(f"{col} : {current_windows[col][0]}")
            
            windows = discretise_et_complete(current_windows, current_windows, 
                                          int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), 
                                          stats_Xamin, SELECTED_COLUMNS_Xamin, 
                                          use_log_scale_Xamin, PAD_TOKEN, max_sources)
            
            # Combinaison et écriture directe dans le fichier
            X = combine_and_flatten_with_special_tokens(windows, current_info_class)
            np.savetxt(f, X, fmt='%d')
            
            # Nettoyage mémoire explicite
            del current_windows, current_info_class, windows, X
    
    print(f"Tous les chunks ont été traités et sauvegardés dans {output_path}")




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