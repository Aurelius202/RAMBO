import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from typing import Tuple
import sys
sys.path.append('/opt/anaconda3/lib/python3.12/site-packages')
from orpl.baseline_removal import bubblefill
import plotly.graph_objs as go 
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def normalize_spectrum_minmax(spectrum: np.ndarray, quantile: float = 1) -> np.ndarray:
    """
    Normalise l'intensité d'un spectre.
    :param spectrum: Tableau 2D avec la première colonne comme dimension indépendante et la deuxième comme intensité.
    :param quantile: Quantile pour la normalisation.
    :return: Spectre avec l'intensité normalisée.
    """
    # Assurez-vous que seul l'axe Y est normalisé
    raman_data = spectrum[1, :]

    # Décalage pour s'assurer que toutes les valeurs sont positives
    min_val = np.min(raman_data)
    if min_val < 0:
        raman_data = raman_data - min_val

    # Calcul de la valeur du quantile
    quantile_value = np.quantile(raman_data, quantile)

    # Normalisation
    normalized_raman_data = raman_data / quantile_value

    # Retourne un tableau 2D avec l'axe X inchangé et l'axe Y normalisé
    return np.vstack((spectrum[0, :], normalized_raman_data))


def normalize_spectrum_sum(spectrum):
    """Normalise le spectre en utilisant la somme des intensités."""
    sum_intensities = np.sum(spectrum[1, :])
    if sum_intensities != 0:
        spectrum[1, :] = spectrum[1, :] / sum_intensities
    return spectrum

def normalize_spectrum_reference(spectrum, reference_peak):
    """Normalise le spectre en utilisant un pic de référence."""
    reference_index = np.argmin(np.abs(spectrum[0, :] - reference_peak))
    reference_intensity = spectrum[1, reference_index]
    if reference_intensity != 0:
        spectrum[1, :] = spectrum[1, :] / reference_intensity
    return spectrum


def normalize_spectrum_vect(spectrum):
    """
    Normalize the spectrum vector.

    Parameters:
    spectrum (np.ndarray): A 2D array where the first row is the x-axis and the second row is the y-axis (intensity).

    Returns:
    np.ndarray: The normalized spectrum.
    """
    # Extract the intensity values
    intensities = spectrum[1, :]

    # Calculate the L2 norm of the intensity vector
    norm = np.linalg.norm(intensities)

    # Normalize the intensity values
    normalized_intensities = intensities / norm

    # Return the spectrum with normalized intensities
    return np.vstack((spectrum[0, :], normalized_intensities))


def find_raman_peaks(spectrum: np.ndarray, height: float = None, distance: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identifie les pics dans un spectre Raman.

    Paramètres :
    - spectrum : np.ndarray, le spectre Raman (axe X et données Raman).
    - height : float, la hauteur minimale des pics à détecter.
    - distance : int, la distance minimale entre deux pics.

    Retourne :
    - peaks_positions : np.ndarray, les positions des pics sur l'axe X.
    - peaks_heights : np.ndarray, les hauteurs des pics.
    """
    xaxis = spectrum[0, :]
    raman_data = spectrum[1, :]

    # Utilisation de find_peaks pour détecter les pics
    peaks, properties = find_peaks(raman_data, height=height, distance=distance, prominence=0.1, width=2)
    peaks_positions = xaxis[peaks]
    peaks_heights = properties['peak_heights']

    print(f"Found {len(peaks_positions)} peaks.")

    return peaks_positions, peaks_heights

def polynomial_fitting(spectrum: np.ndarray, poly_order: int = 6, precision: float = 0.005, max_iter: int = 1000, imod: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    xaxis = spectrum[0, :]
    raman_data = spectrum[1, :]

    if xaxis.ndim != 1 or raman_data.ndim != 1:
        raise ValueError("xaxis and raman_data must be 1D arrays.")

    converged = False
    std_dev = np.inf

    for i in range(1, max_iter + 1):
        poly_coeffs = np.polyfit(xaxis, raman_data, poly_order)
        poly_fit = np.polyval(poly_coeffs, xaxis)

        residual = raman_data - poly_fit
        previous_std_dev = std_dev
        std_dev = np.std(residual)

        if imod:
            mask = raman_data > poly_fit + std_dev
        else:
            mask = raman_data > poly_fit

        raman_data = np.where(mask, poly_fit + std_dev * imod, raman_data)

        if np.abs((std_dev - previous_std_dev) / std_dev) < precision:
            converged = True
            break

    baseline = poly_fit
    raman = spectrum[1, :] - baseline
    return raman, baseline

def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def plot_spectrum_with_peaks(spectrum: np.ndarray, peaks_positions: np.ndarray, peaks_heights: np.ndarray):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spectrum[0], y=spectrum[1], mode='lines', name='Spectrum'))
    fig.add_trace(go.Scatter(x=peaks_positions, y=peaks_heights, mode='markers', name='Peaks'))

    for position, height in zip(peaks_positions, peaks_heights):
        fig.add_annotation(
            x=position,
            y=height,
            text=f'{position:.1f}, {height:.2f}',
            showarrow=True,
            arrowhead=1
        )

    fig.update_layout(
        title='Raman Spectrum with Identified Peaks',
        xaxis_title='Raman Shift (cm^-1)',
        yaxis_title='Intensity'
    )

    return fig

def process_file(file_path: str, poly_order: int = 4, bubble_width: int = 50):
    df = pd.read_csv(file_path, delimiter=',', skiprows=1)
    data = df.to_numpy().T
    data = data[:, data[0] >= 350]

    results_poly = []
    results_bubble = []
    for col in range(1, data.shape[0]):
        spectrum = data[[0, col]]

        raman_poly, _ = polynomial_fitting(spectrum, poly_order=poly_order)
        raman_poly_smoothed = savgol_filter(raman_poly, window_length=11, polyorder=3)
        normalized_spectrum_poly = normalize_spectrum(np.array([data[0], raman_poly_smoothed]))
        peaks_positions_poly, peaks_heights_poly = find_raman_peaks(normalized_spectrum_poly, height=0.01, distance=20)

        raman_bubble, _ = bubblefill(spectrum[1], bubble_width)
        raman_bubble_smoothed = savgol_filter(raman_bubble, window_length=11, polyorder=3)
        normalized_spectrum_bubble = normalize_spectrum(np.array([data[0], raman_bubble_smoothed]))
        peaks_positions_bubble, peaks_heights_bubble = find_raman_peaks(normalized_spectrum_bubble, height=0.01, distance=20)

        for position, height in zip(peaks_positions_poly, peaks_heights_poly):
            results_poly.append({
                'File': os.path.basename(file_path),
                'Spectrum Column': col,
                'Peak Position': position,
                'Peak Height': height
            })

        for position, height in zip(peaks_positions_bubble, peaks_heights_bubble):
            results_bubble.append({
                'File': os.path.basename(file_path),
                'Spectrum Column': col,
                'Peak Position': position,
                'Peak Height': height
            })

        if len(peaks_positions_poly) > 200 or len(peaks_positions_bubble) > 200:
            return results_poly, results_bubble, True, peaks_positions_poly, peaks_heights_poly, peaks_positions_bubble, peaks_heights_bubble, normalized_spectrum_poly, normalized_spectrum_bubble

    return results_poly, results_bubble, False, peaks_positions_poly, peaks_heights_poly, peaks_positions_bubble, peaks_heights_bubble, normalized_spectrum_poly, normalized_spectrum_bubble
