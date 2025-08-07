import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle



# Chargement des constantes
from Constantes import SELECTED_COLUMNS_Xamin
from Constantes import VOCAB_SIZE, PAD_TOKEN, SEP_TOKEN, CLS_TOKEN, SEP_AMAS, NOMBRE_TOKENS_SPECIAUX
from Constantes import BATCH_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, name_dir
from Constantes import name_dir, ISCLUSTER, ISNOTCLUSTER



#///////////////////////////////////////////////////////////////////////////////////////////////////////
#                                       Generation des sequences
#///////////////////////////////////////////////////////////////////////////////////////////////////////


titre = "Generation des sequences"
largeur = 80

print("#" * largeur)
print("#" + " " * (largeur-2) + "#")
print("#" + titre.center(largeur-2) + "#")
print("#" + " " * (largeur-2) + "#")
print("#" * largeur)

# Chargement des données
X_train = np.loadtxt(f'/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/X_train.txt', dtype=np.int32)
X_test = np.loadtxt(f'/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/X_test.txt', dtype=np.int32)

print(f"\nDim de X_train: {X_train.shape}") # Devrait être (n_windows,  len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4)
print(f"Dim de X_test: {X_test.shape}") # Devrait être (n_windows,  len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4)


#////////// Load Configuration /////////
with open(f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/constantes_du_modele.json", 'r') as f:
    config = json.load(f)

MAX_SOURCES  = config["MAX_SOURCES"]

print("┌───────────────────────────────┐")
print("│  MODEL CONFIGURATION          │")
print("├───────────────────────────────┤")
for key, value in config.items():
    print(f"│ {key.ljust(15)}: {str(value).rjust(10)}   │")
print("└───────────────────────────────┘")

#///////////////////////////////////////////////////////////////////////////////////////////////////////


# Architecture du Transformer
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mlp = keras.Sequential([
            layers.Dense(d_model, activation="relu"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, x, mask):
        attn_output = self.attn(x, x, attention_mask=mask)
        x = x + attn_output
        x = self.layernorm1(x)
        mlp_output = self.mlp(x)
        x = x + mlp_output
        return self.layernorm2(x)

class AutoregressiveTransformerModel(keras.Model):
    def __init__(self, d_model, num_heads, num_layers, seq_length, vocab_size):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.token_embed = layers.Embedding(vocab_size, d_model)
        self.pos_embed = layers.Embedding(seq_length, d_model)
        self.transformer_blocks = [TransformerBlock(d_model, num_heads) for _ in range(num_layers)]
        self.output_layer = layers.Dense(vocab_size)

    def call(self, x):
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        
        # Masque causal
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]
        
        # Embeddings
        token_embeddings = self.token_embed(x)
        positions = tf.range(seq_len, dtype=tf.int32)
        pos_embeddings = self.pos_embed(positions)
        x = token_embeddings + pos_embeddings

        # Blocs Transformer
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        return self.output_layer(x)


#///////////////////////////////////////////////////////////////////////////////////////////////////////

# Initialisation et chargement du modèle
seq_length = X_train.shape[1]
model = AutoregressiveTransformerModel(
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    seq_length=seq_length,
    vocab_size=VOCAB_SIZE
)

# Chargement des poids sauvegardés
# Création des variables (NÉCESSAIRE avant load_weights)
dummy_input = tf.ones((1, X_train.shape[1]), dtype=tf.int32)
_ = model(dummy_input)  # Cette ligne crée les variables
# Chargement SIMPLE des poids .h5
model.load_weights(f'/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/model_weights.h5')

# Paramètres de génération
#max_length = len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + len(SELECTED_COLUMNS_input_clusters) * MAX_CLUSTERS * 2 + len(SELECTED_COLUMNS_input_AGN) * MAX_AGN * 2 + 4
max_length = len(X_test[0])

#///////////////////////////////////////////////////////////////////////////////////////////////////////

CLASS_TOKENS = [ISCLUSTER, ISNOTCLUSTER]
CLASS_POSITION = len(SELECTED_COLUMNS_Xamin)*MAX_SOURCES + 2

#///////////////////////////////////////////////////////////////////////////////////////////////////////


# Fonction de génération

def generate_sequence(initial_tokens, max_length, class_position, temperature=1.0):
    """
    Génère une séquence et retourne les probabilités de classe uniquement pour le token à class_position
    Version TensorFlow (remplace JAX)
    """
    class_probs = None  # Initialisation à None si la position n'est pas atteinte

    if isinstance(initial_tokens, (list, np.ndarray)):
        initial_tokens = tf.convert_to_tensor(initial_tokens, dtype=tf.int32)

    if len(initial_tokens.shape) == 1:
        current_tokens = tf.expand_dims(initial_tokens, axis=0)
    else:
        current_tokens = initial_tokens

    for i in range(current_tokens.shape[1], max_length):
        try:
            logits = model(current_tokens)

            if len(logits.shape) == 2:
                next_token_logits = logits[0, :]
            elif len(logits.shape) == 3:
                next_token_logits = logits[0, -1, :]
            else:
                raise ValueError(f"Forme inattendue des logits: {logits.shape}")

            current_position = current_tokens.shape[1]  # Position du token qu'on va ajouter

            if current_position == class_position:
                all_probs = tf.nn.softmax(next_token_logits / temperature)  # Softmax global
                class_probs_tensor = tf.gather(all_probs, CLASS_TOKENS)     # Extraction
                class_probs = class_probs_tensor.numpy() / tf.reduce_sum(class_probs_tensor).numpy()  # Renormalisation

                # Sélection du token le plus probable parmi CLASS_TOKENS
                chosen_class_idx = tf.argmax(class_probs_tensor).numpy()
                chosen_class_token = CLASS_TOKENS[chosen_class_idx]
                next_token = chosen_class_token
            else:
                # Échantillonnage du prochain token pour les autres positions
                next_token = tf.random.categorical(
                    tf.expand_dims(next_token_logits / temperature, 0),
                    num_samples=1
                )[0, 0].numpy()

            # Mise à jour des tokens
            current_tokens = tf.concat([
                current_tokens,
                tf.constant([[next_token]], dtype=tf.int32)
            ], axis=1)
            '''
            if next_token == SEP_TOKEN:
                break
            '''

        except Exception as e:
            print(f"Erreur à l'étape {i}: {str(e)}")
            print(f"Current tokens shape: {current_tokens.shape}")
            print(f"Logits shape: {logits.shape if 'logits' in locals() else 'N/A'}")
            break

    return current_tokens[0].numpy(), class_probs



def generate_sequence_batch(initial_tokens_batch, max_length, class_position, temperature=1.0):
    """Version corrigée avec gestion stricte des types"""
    # Conversion initiale en int32
    if isinstance(initial_tokens_batch, (list, np.ndarray)):
        current_tokens = tf.convert_to_tensor(initial_tokens_batch, dtype=tf.int32)
    else:
        current_tokens = tf.cast(initial_tokens_batch, tf.int32)
    
    batch_size = current_tokens.shape[0]
    class_probs_list = [None] * batch_size

    for i in range(current_tokens.shape[1], max_length):
        try:
            logits = model(current_tokens)
            next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            all_probs = tf.nn.softmax(next_token_logits / temperature, axis=-1)
            current_position = current_tokens.shape[1]

            if current_position == class_position:
                class_probs_tensor = tf.gather(all_probs, CLASS_TOKENS, axis=1)
                class_probs_list = (class_probs_tensor / tf.reduce_sum(class_probs_tensor, axis=1, keepdims=True)).numpy()
                
                chosen_class_idxs = tf.argmax(class_probs_tensor, axis=1).numpy()
                next_tokens = tf.constant([CLASS_TOKENS[idx] for idx in chosen_class_idxs], dtype=tf.int32)  # <-- int32 explicite
            else:
                next_tokens = tf.random.categorical(
                    next_token_logits / temperature, 
                    num_samples=1
                )[:, 0]
                next_tokens = tf.cast(next_tokens, tf.int32)  # <-- Conversion ajoutée

            # Conversion de type explicite avant concat
            next_tokens = tf.cast(next_tokens, tf.int32)
            current_tokens = tf.concat([
                current_tokens,
                tf.expand_dims(next_tokens, axis=1)
            ], axis=1)

        except Exception as e:
            print(f"Erreur à l'étape {i}: {str(e)}")
            print(f"Type current_tokens: {current_tokens.dtype}, Type next_tokens: {next_tokens.dtype}")
            break

    return current_tokens.numpy(), class_probs_list


'''

def Pythie(X, min_idx_gen, max_idx_gen, save_list_of_generated_sequences, name):
    if save_list_of_generated_sequences:
        list_of_generated_sequences = []
        list_of_class_probs = []

        index_end_of_Xamin_part = len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + 1 
        print(f"index_end_of_Xamin_part = {index_end_of_Xamin_part}")
        if max_idx_gen + 1 >= len(X):
            max_idx_gen = len(X) - 1

        for i in range(min_idx_gen, max_idx_gen + 1):
            initial_tokens = X[i, :index_end_of_Xamin_part].astype(np.int32)
            generated_sequence, class_probs = generate_sequence(initial_tokens, max_length=max_length, class_position=CLASS_POSITION)
            list_of_generated_sequences.append(generated_sequence)
            list_of_class_probs.append(class_probs)

        # Sauvegarde des résultats
        save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/"
        suffix = "full" if max_idx_gen == len(X) - 1 else f"{min_idx_gen}-{max_idx_gen}"
        with open(f"{save_path}sequence_divinatio_{name}_{suffix}.pkl", 'wb') as f:
            pickle.dump(list_of_generated_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{save_path}proba_divinatio_{name}_{suffix}.pkl", 'wb') as f:
            pickle.dump(list_of_class_probs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        list_of_sequences = X[:len(list_of_generated_sequences)]
        with open(f"{save_path}sequence_veritas_{name}_{suffix}.pkl", 'wb') as f:
            pickle.dump(list_of_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
'''


def Pythie(X, min_idx_gen, max_idx_gen, save_list_of_generated_sequences, name, batch_size=32):
    if save_list_of_generated_sequences:
        list_of_generated_sequences = []
        list_of_class_probs = []

        index_end_of_Xamin_part = len(SELECTED_COLUMNS_Xamin) * MAX_SOURCES + 1 

        if max_idx_gen + 1 >= len(X):
            max_idx_gen = len(X) - 1

        # Nouveau : traitement par lots vectorisés
        for batch_start in range(min_idx_gen, max_idx_gen + 1, batch_size):
            batch_end = min(batch_start + batch_size, max_idx_gen + 1)
            batch = X[batch_start:batch_end, :index_end_of_Xamin_part].astype(np.int32)
            
            # Appel vectorisé (le vrai changement est ici)
            generated_batch, probs_batch = generate_sequence_batch(
                initial_tokens_batch=batch,
                max_length=max_length,
                class_position=CLASS_POSITION,
                temperature=1.0
            )
            
            list_of_generated_sequences.extend(generated_batch)
            list_of_class_probs.extend(probs_batch)

            # Optionnel : afficher la progression
            print(f"Traitement des séquences {batch_start}-{batch_end-1} terminé")

        # Sauvegarde (identique à votre version originale)
        save_path = f"/lustre/fswork/projects/rech/wka/ufl73qn/Project_Transformer_FornaX/Transformer_Window_Center_Classifier/results/{name_dir}/"
        suffix = "full" if max_idx_gen == len(X) - 1 else f"{min_idx_gen}-{max_idx_gen}"
        
        with open(f"{save_path}sequence_divinatio_{name}_{suffix}.pkl", 'wb') as f:
            pickle.dump(list_of_generated_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f"{save_path}proba_divinatio_{name}_{suffix}.pkl", 'wb') as f:
            pickle.dump(list_of_class_probs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        list_of_sequences = X[:len(list_of_generated_sequences)]
        with open(f"{save_path}sequence_veritas_{name}_{suffix}.pkl", 'wb') as f:
            pickle.dump(list_of_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

# Exécution
Pythie(X_test, 0, 100000, True, "test")


