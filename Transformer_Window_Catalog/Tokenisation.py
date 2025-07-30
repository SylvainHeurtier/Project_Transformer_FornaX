

import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import random
import json
import shutil

from astropy.table import Table, vstack
from astropy.table import join
from collections import Counter
import Fct_tokenisation
from Fct_tokenisation import CreateListID_Xamin, Batisseuse2Fenetres, GardeFenestronsSousPeuples, CompteSourcesParFenetres, random_rotations_and_mirror, compute_global_stats, discretise_et_complete, combine_and_flatten_with_special_tokens, convert_numpy_types

from Constantes import NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON, Nbre2Rotations
from Constantes import path_list_ID_Xamin_AMAS, path_list_ID_Xamin_AGN
from Constantes import SELECTED_COLUMNS_Xamin, SELECTED_COLUMNS_input_clusters, SELECTED_COLUMNS_input_AGN
from Constantes import use_log_scale_Xamin, use_log_scale_input_clusters, use_log_scale_input_AGN
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import WINDOW_SIZE_ARCMIN, NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON, name_dir
from Constantes import catalog_path_aftXamin, new_catalog_path_AGN, new_catalog_path_AMAS


titre = "SÉLECTION, ROTATION ET TOKÉNISATION DES FENETRES"
largeur = 80

print("#" * largeur)
print("#" + " " * (largeur-2) + "#")
print("#" + titre.center(largeur-2) + "#")
print("#" + " " * (largeur-2) + "#")
print("#" * largeur)


# //////////// Chargements des fichiers ////////////
print(" ")
print(f"=== Chargement des fichiers ===")

data_Xamin = Table.read(catalog_path_aftXamin)
data_AMAS  = Table.read(new_catalog_path_AMAS)
data_AGN   = Table.read(new_catalog_path_AGN)

data_Xamin['new_ID'] = np.arange(len(data_Xamin))
data_Xamin['Ntot'] = data_Xamin['INST0_EXP'] * data_Xamin['PNT_RATE_MOS'] + data_Xamin['INST1_EXP'] * data_Xamin['PNT_RATE_PN']
print(f"\n Nombre de sources Xamin:\nAvant la coupe sur le nombre de photons: {len(data_Xamin)}")
data_Xamin = data_Xamin[data_Xamin['Ntot']>=NOMBRE_PHOTONS_MIN]
print(f"Apres la coupe sur le nombre de photons: {len(data_Xamin)}")
print(f"\nRappel: NOMBRE_PHOTONS_MIN = {NOMBRE_PHOTONS_MIN} photons")

list_ID_Xamin_AMAS = np.loadtxt(path_list_ID_Xamin_AMAS, dtype=int)
list_ID_Xamin_AGN  = np.loadtxt(path_list_ID_Xamin_AGN, dtype=int)



# //////////// Separation des données d'entrainement et test ////////////
print(" ")
print(f"=== Séparation des données d'entraînement et test ===")

DEC_LIM_FOR_TRAINING = 2.15 # en degres 
                          # 2deg-> 0.75deg de largeur en dec pour le test sur 1deg en ra

# AMAS

mask_for_training = data_AMAS['Dec'] > DEC_LIM_FOR_TRAINING # en degres

data_AMAS_train = data_AMAS[mask_for_training]
data_AMAS_test  = data_AMAS[~mask_for_training]

pourcentage_train = len(data_AMAS_train)*100/len(data_AMAS)
pourcentage_test  = len(data_AMAS_test) *100/len(data_AMAS)

print(f"\nNombre total d'amas: {len(data_AMAS)}")
print(f"Zone train: {len(data_AMAS_train)} >> {pourcentage_train:.1f}%")
print(f"Zone test:  {len(data_AMAS_test)} >> {pourcentage_test:.1f}%")

# AGN

mask_for_training = data_AGN['dec_mag_gal'] > DEC_LIM_FOR_TRAINING # en degres

data_AGN_train = data_AGN[mask_for_training]
data_AGN_test  = data_AGN[~mask_for_training]

pourcentage_train = len(data_AGN_train)*100/len(data_AGN)
pourcentage_test  = len(data_AGN_test) *100/len(data_AGN)

print(f"\nNombre total d'AGN: {len(data_AGN)}")
print(f"Zone train: {len(data_AGN_train)} >> {pourcentage_train:.1f}%")
print(f"Zone test:  {len(data_AGN_test)} >> {pourcentage_test:.1f}%")

# Xamin

mask_for_training = data_Xamin['PNT_DEC'] > DEC_LIM_FOR_TRAINING # en degres

data_Xamin_train = data_Xamin[mask_for_training]
data_Xamin_test  = data_Xamin[~mask_for_training]

print(f"\nNombre total Xamin: {len(data_Xamin)}")
print(f"Zone train: {len(data_Xamin_train)} >> {len(data_Xamin_train) *100/len(data_Xamin):.1f}%")
print(f"Zone test:  {len(data_Xamin_test)} >> {len(data_Xamin_test) *100/len(data_Xamin):.1f}%")


# //////////// Selection des fenetres ////////////


AllXaminSources = False

if(AllXaminSources):
    list_ID_Xamin_train = data_Xamin_train['ID_Xamin']
    list_ID_Xamin_test  = data_Xamin_test['ID_Xamin']
else:
    list_ID_Xamin_train = CreateListID_Xamin(np.array(data_Xamin_train['ID_Xamin']), list_ID_Xamin_AMAS)
    list_ID_Xamin_test  = CreateListID_Xamin(np.array(data_Xamin_test['ID_Xamin']), list_ID_Xamin_AMAS)

print(f"\nJeu train: {len(list_ID_Xamin_train)}")
print(f"Jeu test:  {len(list_ID_Xamin_test)}")


# //////////// Construction des fenetres ////////////
print(" ")
print(f"=== Construction des fenetres ===")


list_windows_test, info_clusters_test, info_AGN_test = Batisseuse2Fenetres(data_Xamin, 
                                                                           data_AMAS_test, 
                                                                           data_AGN_test, 
                                                                           list_ID_Xamin_test)

print(f"\n✓ Fenêtres test construites")  # ✓ Vert avec reset

list_windows_train, info_clusters_train, info_AGN_train = Batisseuse2Fenetres(data_Xamin, 
                                                                              data_AMAS_train, 
                                                                              data_AGN_train, 
                                                                              list_ID_Xamin_train)

print(f"\n✓ Fenêtres train construites")  # ✓ Vert avec reset


list_windows_test  = list_windows_test[SELECTED_COLUMNS_Xamin + ['window']]
list_windows_train = list_windows_train[SELECTED_COLUMNS_Xamin + ['window']]


# //////////// Selection des fenetres les moins peuplees ////////////


list_windows_test, info_clusters_test, info_AGN_test = GardeFenestronsSousPeuples(list_windows_test, 
                                                                                  info_clusters_test, 
                                                                                  info_AGN_test, 
                                                                                  MAX_Xamin_PAR_FENESTRON)

print(f"\n✓ Fenêtres test reduites")  # ✓ Vert avec reset

list_windows_train, info_clusters_train, info_AGN_train = GardeFenestronsSousPeuples(list_windows_train, 
                                                                                  info_clusters_train, 
                                                                                  info_AGN_train, 
                                                                                  MAX_Xamin_PAR_FENESTRON)

print(f"\n✓ Fenêtres train reduites")  # ✓ Vert avec reset

max_count_sources_train  = CompteSourcesParFenetres(list_windows_train)
max_count_clusters_train = CompteSourcesParFenetres(info_clusters_train)
max_count_AGN_train      = CompteSourcesParFenetres(info_AGN_train)

max_count_sources_test  = CompteSourcesParFenetres(list_windows_test)
max_count_clusters_test = CompteSourcesParFenetres(info_clusters_test)
max_count_AGN_test      = CompteSourcesParFenetres(info_AGN_test)

MAX_SOURCES = max(max_count_sources_train, max_count_sources_test)
MAX_CLUSTERS = max(max_count_clusters_train, max_count_clusters_test)
MAX_AGN = max(max_count_AGN_train, max_count_AGN_test)

print(f"\nMAX_SOURCES : {MAX_SOURCES}")
print(f"MAX_CLUSTERS : {MAX_CLUSTERS}")
print(f"MAX_AGN : {MAX_AGN}")





# //////////// Statistiques ////////////

global_stats_Xamin          = compute_global_stats(vstack([list_windows_train, list_windows_test]), SELECTED_COLUMNS_Xamin)
global_stats_input_clusters = compute_global_stats(vstack([info_clusters_train, info_clusters_test]), SELECTED_COLUMNS_input_clusters)
global_stats_input_AGN      = compute_global_stats(vstack([info_AGN_train, info_AGN_test]), SELECTED_COLUMNS_input_AGN)


# //////////// Rotation des fenetres ////////////
print(" ")
print(f"=== Rotation des fenetres === \n")


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




def save_current_state(windows, clusters, agn, output_dir, chunk_num):
    """Sauvegarde l'état courant dans des fichiers FITS"""
    windows.write(os.path.join(output_dir, f"windows_{chunk_num:04d}.fits"), overwrite=True)
    
    if len(clusters) > 0:
        clusters.write(os.path.join(output_dir, f"clusters_{chunk_num:04d}.fits"), overwrite=True)
    
    if len(agn) > 0:
        agn.write(os.path.join(output_dir, f"agn_{chunk_num:04d}.fits"), overwrite=True)





global_stats_Xamin, global_stats_input_clusters, global_stats_input_AGN = \
    process_rotations_in_chunks(list_windows_test, 
                            info_clusters_test, 
                            info_AGN_test, 
                            total_rotations=30000, 
                            chunk_size=300, 
                            output_dir = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/rotation_output_test",
                            stats_Xamin = global_stats_Xamin, 
                            stats_input_clusters = global_stats_input_clusters, 
                            stats_input_AGN = global_stats_input_AGN)


global_stats_Xamin, global_stats_input_clusters, global_stats_input_AGN = \
    process_rotations_in_chunks(list_windows_train, 
                            info_clusters_train, 
                            info_AGN_train, 
                            total_rotations=30000, 
                            chunk_size=300, 
                            output_dir = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/rotation_output_train",
                            stats_Xamin = global_stats_Xamin, 
                            stats_input_clusters = global_stats_input_clusters, 
                            stats_input_AGN = global_stats_input_AGN)



# //////////// Discretisation des données ////////////
print(" ")
print(f"=== Discretisation des données === \n")


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
        

process_and_save_chunks(
    directory   = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/rotation_output_test",
    output_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_test.txt",
    stats_Xamin          = global_stats_Xamin,
    stats_input_clusters = global_stats_input_clusters,
    stats_input_AGN      = global_stats_input_AGN,
    max_sources  = MAX_SOURCES,
    max_clusters = MAX_CLUSTERS,
    max_agn      = MAX_AGN
)

windows_train = discretise_et_complete(list_windows_train, list_windows_train, int(VOCAB_SIZE-NOMBRE_TOKENS_SPECIAUX), global_stats_Xamin, SELECTED_COLUMNS_Xamin, use_log_scale_Xamin, PAD_TOKEN, MAX_SOURCES)
print("\n      **** TEST  DE DISCRETISATION DIRECTE : OK ***\n")

process_and_save_chunks(
    directory   = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/rotation_output_train",
    output_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/X_train.txt",
    stats_Xamin          = global_stats_Xamin,
    stats_input_clusters = global_stats_input_clusters,
    stats_input_AGN      = global_stats_input_AGN,
    max_sources  = MAX_SOURCES,
    max_clusters = MAX_CLUSTERS,
    max_agn      = MAX_AGN
)




# //////////// Discretisation des données ////////////
print(" ")
print(f"=== Sauvegarde des données === \n")


# Création du dictionnaire
constantes_du_modele = {
    "VOCAB_SIZE": VOCAB_SIZE,
    "PAD_TOKEN": PAD_TOKEN,
    "SEP_TOKEN": SEP_TOKEN,
    "CLS_TOKEN": CLS_TOKEN,
    "SEP_AMAS": SEP_AMAS,
    "SEP_AGN": SEP_AGN,
    "NOMBRE_TOKENS_SPECIAUX": NOMBRE_TOKENS_SPECIAUX,
    "MAX_SOURCES": int(MAX_SOURCES),
    "MAX_CLUSTERS": int(MAX_CLUSTERS),
    "MAX_AGN": int(MAX_AGN)
}

# Sauvegarde en JSON
save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/constantes_du_modele.json"
with open(save_path, 'w') as f:
    json.dump(constantes_du_modele, f, indent=4)

print(f"Dictionnaire sauvegardé dans {save_path}")

# Sauvegarde des statistiques globales

save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/global_stats_Xamin.json"
with open(save_path, 'w') as f:
    json.dump(global_stats_Xamin, f, indent=4)
print(f"Dictionnaire sauvegardé dans {save_path}")

save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/global_stats_input_clusters.json"
with open(save_path, 'w') as f:
    json.dump(global_stats_input_clusters, f, indent=4)
print(f"Dictionnaire sauvegardé dans {save_path}")


global_stats_serializable = convert_numpy_types(global_stats_input_AGN)

save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/global_stats_input_AGN.json"
with open(save_path, 'w') as f:
    json.dump(global_stats_serializable, f, indent=4, ensure_ascii=False)
print(f"Dictionnaire sauvegardé dans {save_path}")



print("\n   ***   THE END   ***   \n")
