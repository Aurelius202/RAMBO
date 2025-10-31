import os
import pandas as pd

def load_data(file_paths):
    """
    Charge les données des fichiers CSV spécifiés.

    :param file_paths: Liste des chemins de fichiers CSV à charger.
    :return: Dictionnaire contenant les données chargées.
    """
    data = {}

    for file_path in file_paths:
        # Lire les résultats
        df = pd.read_csv(file_path)

        # Extraire les positions, les intensités et les noms de fichiers des pics
        peaks = df['Peak Position'].tolist()
        intensities = df['Peak Height'].tolist()

        # Ajouter les pics et les intensités au dictionnaire
        data[os.path.basename(file_path)] = {
            'Peak Position': peaks,
            'Peak Height': intensities
        }

    return data

def save_results(results_df, output_folder, filename='gag_sulfate_max_intensities.xlsx'):
    """
    Sauvegarde les résultats dans un fichier Excel.

    :param results_df: DataFrame contenant les résultats à sauvegarder.
    :param output_folder: Chemin du dossier de sortie.
    :param filename: Nom du fichier de sortie.
    """
    output_file = os.path.join(output_folder, filename)
    results_df.to_excel(output_file, index=False)
