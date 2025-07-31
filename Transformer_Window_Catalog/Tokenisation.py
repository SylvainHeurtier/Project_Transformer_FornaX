

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
from Fct_tokenisation import CreateListID_Xamin, Batisseuse2Fenetres, GardeFenestronsSousPeuples, CompteSourcesParFenetres, compute_global_stats, discretise_et_complete, convert_numpy_types, process_rotations_in_chunks, process_and_save_chunks

from Constantes import NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON, TOTAL_ROTATIONS, CHUNK_SIZE
from Constantes import path_list_ID_Xamin_AMAS, path_list_ID_Xamin_AGN
from Constantes import SELECTED_COLUMNS_Xamin, SELECTED_COLUMNS_input_clusters, SELECTED_COLUMNS_input_AGN
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, SEP_AGN, NOMBRE_TOKENS_SPECIAUX
from Constantes import NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON, name_dir
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

global_stats_Xamin, global_stats_input_clusters, global_stats_input_AGN = \
    process_rotations_in_chunks(list_windows_test, 
                            info_clusters_test, 
                            info_AGN_test, 
                            total_rotations=TOTAL_ROTATIONS,
                            chunk_size=CHUNK_SIZE, 
                            output_dir = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/rotation_output_test",
                            stats_Xamin = global_stats_Xamin, 
                            stats_input_clusters = global_stats_input_clusters, 
                            stats_input_AGN = global_stats_input_AGN)


global_stats_Xamin, global_stats_input_clusters, global_stats_input_AGN = \
    process_rotations_in_chunks(list_windows_train, 
                            info_clusters_train, 
                            info_AGN_train, 
                            total_rotations=TOTAL_ROTATIONS, 
                            chunk_size=CHUNK_SIZE, 
                            output_dir = f"/lustre/fswork/projects/rech/wka/ufl73qn/TransformerProject/results/{name_dir}/rotation_output_train",
                            stats_Xamin = global_stats_Xamin, 
                            stats_input_clusters = global_stats_input_clusters, 
                            stats_input_AGN = global_stats_input_AGN)



# //////////// Discretisation des données ////////////
print(" ")
print(f"=== Discretisation des données === \n")



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
