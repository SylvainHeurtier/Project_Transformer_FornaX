

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
from Fct_tokenisation import CreateListID_Xamin, Batisseuse2Fenetres, GardeFenestronsSousPeuples 
from Fct_tokenisation import compute_global_stats, process_rotations_in_chunks, process_and_save_chunks
from Fct_tokenisation import verifier_discretisation, verify_table

from Constantes import NOMBRE_PHOTONS_MIN, MAX_Xamin_PAR_FENESTRON
from Constantes import path_list_ID_Xamin_AMAS, path_list_ID_Xamin_AGN
from Constantes import SELECTED_COLUMNS_Xamin, use_log_scale_Xamin
from Constantes import TOTAL_ROTATIONS, CHUNK_SIZE
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, NOMBRE_TOKENS_SPECIAUX
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

list_windows_test, info_class_test = Batisseuse2Fenetres(data_Xamin, list_ID_Xamin_test, list_ID_Xamin_AMAS)
print(f"\n✓ Fenêtres test construites")
list_windows_train, info_class_train = Batisseuse2Fenetres(data_Xamin, list_ID_Xamin_train, list_ID_Xamin_AMAS)
print(f"\n✓ Fenêtres train construites")

'''
print("\n ///////////   TEST 1    ///////////")
verify_table(list_windows_train, "list_windows_train")
verify_table(info_clusters_train, "info_clusters_train")
verify_table(info_AGN_train, "info_AGN_train")
print("\n //////////////////////////////////")
'''

list_windows_test  = list_windows_test[SELECTED_COLUMNS_Xamin + ['window']]
list_windows_train = list_windows_train[SELECTED_COLUMNS_Xamin + ['window']]

'''
print("\n ///////////   TEST 2    ///////////")
verify_table(list_windows_train, "list_windows_train")
verify_table(info_clusters_train, "info_clusters_train")
verify_table(info_AGN_train, "info_AGN_train")
print("\n //////////////////////////////////")
'''

# //////////// Selection des fenetres les moins peuplees ////////////

list_windows_test, info_class_test = GardeFenestronsSousPeuples(list_windows_test, info_class_test, MAX_Xamin_PAR_FENESTRON)
print(f"\n✓ Fenêtres test reduites")
list_windows_train, info_class_train = GardeFenestronsSousPeuples(list_windows_train, info_class_train, MAX_Xamin_PAR_FENESTRON)
print(f"\n✓ Fenêtres train reduites")

MAX_SOURCES = MAX_Xamin_PAR_FENESTRON
print(f"\nMAX_SOURCES : {MAX_SOURCES}")


 # Affiche la presence de doublons ou non
windows = info_class_train['window'].data  # Récupère la colonne 'window' comme array NumPy
unique_windows, counts = np.unique(windows, return_counts=True) # Trouve les valeurs uniques et leurs comptages
doublons = unique_windows[counts > 1]
if len(doublons) > 0:
    print("⚠️ Doublons trouvés dans 'window'")
else:
    print("✅ Aucun doublon dans 'window'")




# //////////// Statistiques ////////////

global_stats_Xamin = compute_global_stats(vstack([list_windows_train, list_windows_test]), SELECTED_COLUMNS_Xamin)


# //////////// Rotation des fenetres ////////////
print(" ")
print(f"=== Rotation des fenetres === \n")

global_stats_Xamin = process_rotations_in_chunks(list_windows_test, 
                            info_class_test, 
                            total_rotations=TOTAL_ROTATIONS,
                            chunk_size=CHUNK_SIZE, 
                            output_dir = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/rotation_output_test",
                            stats_Xamin = global_stats_Xamin)


global_stats_Xamin = process_rotations_in_chunks(list_windows_train, 
                            info_class_train, 
                            total_rotations=TOTAL_ROTATIONS, 
                            chunk_size=CHUNK_SIZE, 
                            output_dir = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/rotation_output_train",
                            stats_Xamin = global_stats_Xamin)



# //////////// Plot pour verifier la discretisation des données ////////////


verifier_discretisation(list_windows_train, SELECTED_COLUMNS_Xamin, use_log_scale_Xamin,
                        column_to_check = 'EXT', 
                        n_bins = VOCAB_SIZE - NOMBRE_TOKENS_SPECIAUX, 
                        max_sources = MAX_SOURCES,
                        stats_Xamin = global_stats_Xamin)


verifier_discretisation(list_windows_train,  SELECTED_COLUMNS_Xamin, use_log_scale_Xamin,
                        column_to_check = 'PNT_DET_ML', 
                        n_bins = VOCAB_SIZE - NOMBRE_TOKENS_SPECIAUX, 
                        max_sources = MAX_SOURCES,
                        stats_Xamin = global_stats_Xamin)

verifier_discretisation(list_windows_train, SELECTED_COLUMNS_Xamin, use_log_scale_Xamin,
                        column_to_check = 'EXT_LIKE', 
                        n_bins = VOCAB_SIZE - NOMBRE_TOKENS_SPECIAUX, 
                        max_sources = MAX_SOURCES,
                        stats_Xamin = global_stats_Xamin)


verifier_discretisation(list_windows_train, SELECTED_COLUMNS_Xamin, use_log_scale_Xamin,
                        column_to_check = 'EXT_RA', 
                        n_bins = VOCAB_SIZE - NOMBRE_TOKENS_SPECIAUX, 
                        max_sources = MAX_SOURCES,
                        stats_Xamin = global_stats_Xamin)


verifier_discretisation(list_windows_train, SELECTED_COLUMNS_Xamin, use_log_scale_Xamin,
                        column_to_check = 'EXT_DEC', 
                        n_bins = VOCAB_SIZE - NOMBRE_TOKENS_SPECIAUX, 
                        max_sources = MAX_SOURCES,
                        stats_Xamin = global_stats_Xamin)


# //////////// Discretisation des données ////////////
print(" ")
print(f"=== Discretisation des données === \n")

process_and_save_chunks(
    directory   = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/rotation_output_test",
    output_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/X_test.txt",
    stats_Xamin = global_stats_Xamin,
    max_sources = MAX_SOURCES
)

process_and_save_chunks(
    directory   = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/rotation_output_train",
    output_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/X_train.txt",
    stats_Xamin = global_stats_Xamin,
    max_sources = MAX_SOURCES
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
    "NOMBRE_TOKENS_SPECIAUX": NOMBRE_TOKENS_SPECIAUX,
    "MAX_SOURCES": int(MAX_SOURCES)
}

# Sauvegarde en JSON
save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/constantes_du_modele.json"
with open(save_path, 'w') as f:
    json.dump(constantes_du_modele, f, indent=4)

print(f"Dictionnaire sauvegardé dans {save_path}")

# Sauvegarde des statistiques globales

save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/global_stats_Xamin.json"
with open(save_path, 'w') as f:
    json.dump(global_stats_Xamin, f, indent=4)
print(f"Dictionnaire sauvegardé dans {save_path}")


print("\n   ***   THE END   ***   \n")
