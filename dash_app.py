import dash
from dash import dcc, html, Input, Output, State, ALL, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import tempfile
import zipfile
import json
import logging
from io import BytesIO, StringIO
from scipy.signal import savgol_filter
from tensorflow.keras.models import load_model
from src.analysis.spectrum_processing import (
    polynomial_fitting,
    bubblefill,
    find_raman_peaks,
    baseline_als_optimized,
    normalize_spectrum_minmax,
    normalize_spectrum_vect,
    normalize_spectrum_sum,
    normalize_spectrum_reference
)
# Configuration and global variables
DEFAULT_DELIMITER = ','
DEFAULT_SKIPROWS = 1
DEFAULT_X_MIN = 0
DEFAULT_X_MAX = None
tissues = ['Adipose tissue', 'Bone', 'Cartilage', 'Skeletal Muscle', 'Tendon']
# Load peak assignments
with open('data/peak_assignments.json', 'r') as file:
    peak_assignments = json.load(file)
# Load model
model = load_model('predict.h5')
# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap'])
app.title = "RAMBO"
app.config.suppress_callback_exceptions = True
# Global variable for storing file data
file_data = {}
def create_input_field(label_text, input_id, input_type='text', value=None, style=None, placeholder=None):
    """Create an input field with a label."""
    input_element = dcc.Input(
        id=input_id,
        type=input_type,
        value=value,
        style=style,
        placeholder=placeholder
    )
    return html.Div([
        html.Span(label_text, style={'margin-right': '5px'}),
        input_element
    ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'})
# Define the layout of the app
app.layout = dbc.Container([
    # Stores for global data
    dcc.Store(id='selected-files-store', data=[]),
    dcc.Store(id='predictions-store', data={}),

    # Main container with three columns: left (controls), center (graphs), right (table)
    html.Div([
        # Left column: Upload, settings, and controls
        html.Div([
            # App logo 
            html.Div([
                html.Img(src='/assets/logo.png', style={'height':'150px', 'width':'auto', 'margin': 'auto', 'display': 'block'}),
            ], style={'text-align': 'center', 'width': '100%'}),

            # App title and description
            html.H2("Raman-Assisted Molecular Biological Observations", style={'margin-top': '4px', 'text-align': 'center'}),
            html.P("Upload your Raman spectra files and analyze them with various preprocessing options.", style={'text-align': 'center'}),

            # Advanced CSV settings section
            html.Div([
                dbc.Button(
                    "Advanced CSV Settings",
                    id="collapse-button",
                    color="secondary",
                    className="mb-3",
                    style={'width': '80%'} 
                ),
                dbc.Collapse(
                    html.Div([
                        create_input_field("CSV delimiter", 'csv-delimiter', value=DEFAULT_DELIMITER),
                        create_input_field("Skip rows", 'skiprows', input_type='number', value=DEFAULT_SKIPROWS),
                        create_input_field("X-axis range (Min)", 'x-min', input_type='number', placeholder='Min'),
                        create_input_field("X-axis range (Max)", 'x-max', input_type='number', placeholder='Max'),
                    ]),
                    id="collapse",
                    is_open=False,
                ),
            ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'width': '100%'}),  # Center all children horizontally

            # File upload section 
            html.Div([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or Select CSV Files'
                    ]),
                    style={
                        'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                        'textAlign': 'center', 'margin': '10px'
                    },
                    multiple=True
                ),
                html.Div(id='feedback-message', style={'margin-top': '10px', 'text-align': 'center'})
            ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'center', 'width': '100%'}),  

            # File selection
            html.Div([
                dcc.Dropdown(
                id='file-dropdown',
                multi=True,
                placeholder="Select files to analyze",
                className="dropdown-menu position-static d-grid gap-1 p-2 rounded-3 mx-0 shadow w-220px",
                style={'z-index': 1050, 'position': 'relative', 'width': '80%'}
                ),
                dbc.Button("Select All", id="select-all-button", n_clicks=0, color="primary", style={'margin-top': '10px', 'width': '20%'})
            ], style={'display': 'flex', 'justify-content': 'center'}), 

            html.Hr(),

            # Savitzky-Golay filter section
            html.Div([
                dbc.Checklist(
                    id='savgol-checkbox',
                    options=[{'label': 'Apply Savitzky-Golay Filter', 'value': 'apply'}],
                    value=['apply'],
                    inline=True
                ),
                dbc.Tooltip(
                    "Smooths the data to reduce noise. Reference: Savitzky, A., Golay, M. J. E. (1964).",
                    target="savgol-checkbox"
                ),
                html.Div(id='savgol-params-container', children=[
                    dbc.Row([
                        dbc.Col(dbc.Input(id='savgol-window', type='number', min=1, max=100, value=11), width=3),
                        dbc.Col(dbc.FormFeedback(id='savgol-window-feedback', type='invalid')),
                        dbc.Col(dbc.Tooltip(
                            "Window size for the Savitzky-Golay filter",
                            target="savgol-window"
                        )),
                        dbc.Col(html.Span('(Window size)'), width='auto'),
                        dbc.Col(dbc.Input(id='savgol-order', type='number', min=1, max=10, value=3), width=3),
                        dbc.Col(dbc.Tooltip(
                            "Polynomial order for the Savitzky-Golay filter",
                            target="savgol-order"
                        )),
                        dbc.Col(html.Span('(Polynomial order)'), width='auto')
                    ], style={'display':'flex', 'align-items' : 'center', 'margin-top': '10px'}),
                ]),
            ]),
            html.Hr(),

            # Baseline removal section
            html.Div([
                dbc.Checklist(
                    id='baseline-checkbox',
                    options=[{'label': 'Apply Baseline Removal', 'value': 'apply'}],
                    value=['apply'],
                    inline=True
                ),
                html.Div(id='baseline-params-container', children=[
                    dcc.Dropdown(
                        id='baseline-method',
                        options=[
                            {'label': 'Polynomial Fitting', 'value': 'poly'},
                            {'label': 'Bubble Fill', 'value': 'bubble'},
                            {'label': 'Asymmetric Least Squares', 'value': 'als'}
                        ],
                        value='poly',
                        className="dropdown-menu position-static d-grid gap-1 p-2 rounded-3 mx-0 shadow w-220px",
                        style={'z-index': 1050, 'position': 'relative'}
                    ),
                    html.Div(id='baseline-method-params', children=[
                        dbc.Row([
                            dbc.Col(dbc.Input(id='poly-order', type='number', min=1, max=10, value=4, style={'width': '100px'}), width=2),
                            dbc.Col(html.Span(" (Polynomial Order)", style={'margin-left': '5px'}), width='auto'),
                            dbc.Col(dbc.Tooltip(
                                "Polynomial order for baseline fitting",
                                target="poly-order"
                            )),
                        ])
                    ]),
                ]),
            ]),
            html.Hr(),

            # Normalization section
            html.Div([
                dbc.Checklist(
                    id='norm-checkbox',
                    options=[{'label': 'Apply Normalization', 'value': 'apply'}],
                    value=['apply'],
                    inline=True
                ),
                html.Div(id='norm-params-container', children=[
                    dcc.Dropdown(
                        id='norm-method-combined',
                        options=[
                            {'label': 'Max - Min-Max', 'value': 'max_minmax'},
                            {'label': 'Max - Vector', 'value': 'max_vector'},
                            {'label': 'Quantile - Min-Max', 'value': 'quantile_minmax'},
                            {'label': 'Quantile - Vector', 'value': 'quantile_vector'},
                            {'label': 'Sum of Intensities', 'value': 'sum'},
                            {'label': 'Reference Peak', 'value': 'reference'}
                        ],
                        value='max_minmax',
                        clearable=False
                    ),
                    dbc.Input(id='quantile-input', type='text', value='0.95', style={'display': 'none'}),
                    dbc.Input(id='reference-peak-input', type='text', placeholder='Reference Peak Position', style={'display': 'none'}),
                ]),
            ]),
            html.Hr(),

            # Peak detection section
            html.Div([
                html.Label('Peak Detection Method:'),
                dcc.Dropdown(
                    id='peak-detection-method',
                    options=[
                        {'label': 'Select by prominence', 'value': 'auto'},
                        {'label': 'Select by Tissue', 'value': 'tissue'}
                    ],
                    value='tissue',
                    placeholder="Select Method",
                    className="dropdown-menu position-static d-grid gap-1 p-2 rounded-3 mx-0 shadow w-220px",
                    style={'z-index': 1050, 'position': 'relative'}
                ),
                dcc.Dropdown(
                    id='tissue-dropdown',
                    options=[{'label': html.Div([
                        html.Img(src=f"/assets/Icons/{tissue}.png", style={'width': '20px', 'height': '20px', 'vertical-align': 'middle'}),
                        f" {tissue}"
                    ]), 'value': tissue} for tissue in tissues],
                    placeholder="Select Tissue",
                    className="dropdown-menu position-static d-grid gap-1 p-2 rounded-3 mx-0 shadow w-220px",
                    style={'display': 'none', 'z-index': 0, 'position': 'relative'}
                ),
                dbc.Row([
                    dbc.Col(dbc.Input(id='peak-height', type='text', value='0.5'), width=3),
                    dbc.Col(html.Span(" (Minimum Height)", style={'margin-left': '5px'}), width='auto'),
                    dbc.Col(dbc.Tooltip(
                        "Minimum height of peaks to detect",
                        target="peak-height"
                    )),
                    dbc.Col(dbc.Input(id='peak-distance', type='number', min=1, max=100, value=20), width=3),
                    dbc.Col(html.Span(" (Minimum Distance)", style={'margin-left': '5px'}), width='auto'),
                    dbc.Col(dbc.Tooltip(
                        "Minimum distance between detected peaks",
                        target="peak-distance"
                    )),
                ]),
            ]),
            html.Hr(),

            # Peak search section
            html.Div([
                html.Label('Search Peaks at Specific Positions:'),
                dbc.Row([
                    dbc.Col(dbc.Input(id='peak-positions-input', type='text', value='', placeholder='e.g., 1000, 1200, 1500', style={'width': '200px'}), width=4),
                    dbc.Col(html.Span(" (Comma-separated peak positions)", style={'margin-left': '5px'}), width='auto'),
                    dbc.Col(dbc.Tooltip(
                        "Enter the peak positions to search for, separated by commas",
                        target="peak-positions-input"
                    )),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Input(id='tolerance-input', type='text', value='5.0', style={'width': '100px'}), width=2),
                    dbc.Col(html.Span(" (Tolerance Value)", style={'margin-left': '5px'}), width='auto'),
                    dbc.Col(dbc.Tooltip(
                        "Tolerance for matching peaks",
                        target="tolerance-input"
                    )),
                ]),
                dbc.Button("Export Specific Peaks", id="export-peak-table-button", n_clicks=0, color="primary", style={'margin-top': '10px', 'width': '80%'}),  # Centered button
                dcc.Download(id="download-peak-table-data")
            ]),
            html.Hr(),

            # Export buttons (centered)
            html.Div([
                dbc.Button("Export Raw Data", id="export-raw-button", n_clicks=0, color="success", style={'margin-top': '10px', 'width': '80%'})
            ], style={'display': 'flex', 'justify-content': 'center'}),

            html.Div([
                dbc.Button("Export Processed Data", id="export-processed-button", n_clicks=0, color="info", style={'margin-top': '10px', 'width': '80%'})
            ], style={'display': 'flex', 'justify-content': 'center'}),

            html.Div([
                dbc.Button("Export All Processed Data", id="export-all-processed-button", n_clicks=0, color="danger", style={'margin-top': '10px', 'width': '80%'})
            ], style={'display': 'flex', 'justify-content': 'center'}),

            html.Hr(),

            # SNR output section
            html.Div([
                html.Label('Signal to Noise Ratio (SNR):'),
                html.Div(id='snr-output', style={'margin-top': '10px'})
            ]),
            html.Hr(),

            # Predicted tissue types section
            html.Div([
                html.Label('Predicted Tissue Types:'),
                html.Div(id='tissue-type-output', style={'margin-top': '10px'})
            ]),
            html.Hr(),

            # Bottom logos (centered)
            html.Div([
                html.Img(src='/assets/HTlogo.png', style={'height':'100px', 'width':'auto', 'margin': '10px'}),
                html.Img(src='/assets/UniBaslogo.png', style={'height':'100px', 'width':'auto', 'margin': '10px'}),
            ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'gap': '20px'})
        ], style={'width': '30%', 'height': '90vh', 'overflow-y': 'auto', 'margin': '10px', 'padding': '10px', 'box-shadow': '0 0 10px rgba(0,0,0,0.1)'}),  # Left column style

        # Center column: Graphs
        html.Div([
            dcc.Graph(
                id='original-spectrum',
                style={'width': '100%', 'height': '45vh', 'display': 'inline-block'},
                config={'displayModeBar': True, 'displaylogo': False},
                figure={
                    'layout': {
                        'margin': {'t': 40, 'b': 40, 'l': 60, 'r': 20},
                        'legend': {'x': 1, 'y': 1}
                    }
                }
            ),
            dcc.Graph(
                id='processed-spectrum',
                style={'width': '100%', 'height': '45vh', 'display': 'inline-block'},
                config={'displayModeBar': True, 'displaylogo': False},
                figure={
                    'layout': {
                        'margin': {'t': 40, 'b': 40, 'l': 60, 'r': 20},
                        'legend': {'x': 1, 'y': 1}
                    }
                }
            ),
            html.Hr(),
            html.Label('Spectra Colors:'),
            html.Div(id='color-pickers', style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}),  # Center color pickers
        ], style={'width': '35%', 'height': '90vh', 'overflow-y': 'auto', 'margin': '10px', 'padding': '10px', 'box-shadow': '0 0 10px rgba(0,0,0,0.1)'}),  # Center column style

        # Right column: Peaks table
        html.Div([
            html.Label('Select Spectrum for Peaks Table:'),
            html.Div([
                dcc.Dropdown(
                    id='spectrum-dropdown',
                    placeholder="Select Spectrum",
                    clearable=False,
                    className="dropdown-menu position-static d-grid gap-1 p-2 rounded-3 mx-0 shadow w-220px"
                )
            ], style={'z-index': 1050, 'position': 'relative'}),
            html.Table(
                id='peaks-table',
                children=[
                    html.Thead(html.Tr([
                        html.Th("Peak Position"),
                        html.Th("Peak Height"),
                        html.Th("Variability Range"),
                        html.Th("Biochemical Component"),
                        html.Th("Tissue")
                    ])),
                    html.Tbody(id='table-body')
                ],
                style={'width': '100%', 'display': 'block', 'overflow-y': 'scroll', 'height': '100%'}
            ),

            html.Div([
                dbc.Button("Export Table", id="export-table-button", n_clicks=0, color="primary", style={'margin-top': '10px', 'width': '100%'}),
                dbc.Button("Export All Tables", id="export-all-tables-button", n_clicks=0, color="warning", style={'margin-top': '10px', 'width': '100%'})
            ], style={'display': 'flex', 'justify-content': 'center'}),  # Centered button

            dcc.Download(id="download-table-data"),
            dcc.Download(id="download-all-tables-data"),
        ], style={'width': '30%', 'height': '90vh', 'overflow-y': 'auto', 'margin': '10px', 'padding': '10px', 'box-shadow': '0 0 10px rgba(0,0,0,0.1)'})  # Right column style
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-between', 'height': '100vh'})  # Main row container
], fluid=True, className='dashboard-container', style={'height': '100vh'})  # Main app container


# Define callbacks for toggling visibility of parameter inputs
@app.callback(
    Output('savgol-params-container', 'style'),
    [Input('savgol-checkbox', 'value')]
)
def toggle_savgol_params(checkbox_value):
    if checkbox_value and 'apply' in checkbox_value:
        return {'display': 'block', 'margin-top': '15px', 'padding-top': '15px'}
    else:
        return {'display': 'none'}
@app.callback(
    Output('baseline-params-container', 'style'),
    [Input('baseline-checkbox', 'value')]
)
def toggle_baseline_params(checkbox_value):
    if checkbox_value and 'apply' in checkbox_value:
        return {'display': 'block', 'margin-top': '15px', 'padding-top': '15px'}
    else:
        return {'display': 'none'}
@app.callback(
    Output('norm-params-container', 'children'),
    [Input('norm-checkbox', 'value')]
)
def toggle_norm_params(checkbox_value):
    if checkbox_value and 'apply' in checkbox_value:
        return dbc.Row([
            dbc.Col(dcc.Dropdown(
                id='norm-method-combined',
                options=[
                    {'label': 'Max - Min-Max', 'value': 'max_minmax'},
                    {'label': 'Max - Vector', 'value': 'max_vector'},
                    {'label': 'Quantile - Min-Max', 'value': 'quantile_minmax'},
                    {'label': 'Quantile - Vector', 'value': 'quantile_vector'},
                    {'label': 'Sum of Intensities', 'value': 'sum'},
                    {'label': 'Reference Peak', 'value': 'reference'}
                ],
                value='max_minmax',
                clearable=False
            ), width=8),
            dbc.Col(dbc.Input(id='quantile-input', type='text', value='0.95', style={'display': 'none'}), width=2),
            dbc.Col(dbc.Input(id='reference-peak-input', type='text', placeholder='Reference Peak Position', style={'display': 'none'}), width=2)
        ])
    else:
        return []
@app.callback(
    [Output('quantile-input', 'style'),
     Output('reference-peak-input', 'style')],
    [Input('norm-method-combined', 'value')]
)
def update_norm_input_visibility(norm_method):
    quantile_style = {'display': 'none'}
    reference_style = {'display': 'none'}
    if norm_method in ['quantile_minmax', 'quantile_vector']:
        quantile_style = {'display': 'block'}
    elif norm_method == 'reference':
        reference_style = {'display': 'block'}
    return quantile_style, reference_style
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open
@app.callback(
    Output('baseline-method-params', 'children'),
    [Input('baseline-method', 'value')]
)
def update_baseline_params_layout(method):
    poly_order_style = {'display': 'none'}
    bubble_width_style = {'display': 'none'}
    als_lambda_style = {'display': 'none'}
    als_p_style = {'display': 'none'}
    if method == 'poly':
        poly_order_style = {'display': 'block'}
    elif method == 'bubble':
        bubble_width_style = {'display': 'block'}
    elif method == 'als':
        als_lambda_style = {'display': 'block'}
        als_p_style = {'display': 'block'}
    return [
        dbc.Row([
            dbc.Col(dbc.Input(id='poly-order', type='number', min=1, max=10, value=4, style={'width': '100px', **poly_order_style}), width=2),
            dbc.Col(html.Span(" (Polynomial Order)", style={'margin-left': '5px', **poly_order_style}), width='auto'),
            dbc.Col(dbc.Tooltip("Polynomial order for baseline fitting", target="poly-order"), style=poly_order_style),
        ]),
        dbc.Row([
            dbc.Col(dbc.Input(id='bubble-width', type='number', min=1, max=100, value=50, style={'width': '100px', **bubble_width_style}), width=2),
            dbc.Col(html.Span(" (Bubble Width)", style={'margin-left': '5px', **bubble_width_style}), width='auto'),
            dbc.Col(dbc.Tooltip("Width of the bubble for baseline removal", target="bubble-width"), style=bubble_width_style),
        ]),
        dbc.Row([
            dbc.Col(html.Span("Lambda (λ)", id='lambda-label', style={'margin-left': '5px', **als_lambda_style}), width='auto'),
            dbc.Col(dcc.Input(id='als-lambda', type='number', value=100000, style={'width': '100px', **als_lambda_style}), width=2),
            dbc.Col(dbc.Tooltip("Smoothness parameter for ALS", target="als-lambda"), style=als_lambda_style),
            dbc.Col(html.Span("P", id='p-label', style={'margin-left': '5px', **als_p_style}), width='auto'),
            dbc.Col(dcc.Input(id='als-p', type='number', value=0.01, style={'width': '100px', **als_p_style}), width=2),
            dbc.Col(dbc.Tooltip("Asymmetry parameter for ALS", target="als-p"), style=als_p_style),
        ])
    ]
@app.callback(
    Output('tissue-dropdown', 'style'),
    Input('peak-detection-method', 'value'),
)
def toggle_tissue_dropdown(selected_method):
    if selected_method == 'tissue':
        return {'position': 'relative', 'z-index': 1051 }
    return {'display': 'none'}
def preprocess_spectrum(data):
    """Preprocess the spectrum by applying various operations."""
    if data.empty:
        raise ValueError("Data is empty")
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    wavenumber = data.iloc[:, 0].values
    spectrum = data.iloc[:, 1].values
    mask = (wavenumber >= 300) & (wavenumber <= 3000)
    wavenumber = wavenumber[mask]
    spectrum = spectrum[mask]
    x = np.arange(len(spectrum))
    p = np.polyfit(x, spectrum, 3)
    baseline = np.polyval(p, x)
    spectrum = spectrum - baseline
    spectrum = savgol_filter(spectrum, window_length=11, polyorder=3)
    spectrum = (spectrum - np.mean(spectrum)) / np.std(spectrum)
    expected_length = 1662
    if len(spectrum) < expected_length:
        spectrum = np.pad(spectrum, (0, expected_length - len(spectrum)), mode='constant')
    elif len(spectrum) > expected_length:
        spectrum = spectrum[:expected_length]
    return spectrum
def predict_tissue_type(selected_files, tissue_types=None):
    if tissue_types is None:
        tissue_types = ['Adipose tissue', 'Bone', 'Cartilage', 'Skeletal Muscle', 'Tendon']
    spectra = []
    filenames = []
    for filename in selected_files:
        if filename in file_data:
            try:
                data = file_data[filename]
                processed_spectrum = preprocess_spectrum(data)
                spectra.append(processed_spectrum)
                filenames.append(filename)
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
                continue
    if not spectra:
        return {}
    spectra_array = np.array(spectra)
    spectra_array = spectra_array.reshape((spectra_array.shape[0], spectra_array.shape[1], 1))
    predictions = model.predict(spectra_array)
    predicted_types = [tissue_types[np.argmax(pred)] for pred in predictions]
    predictions_dict = {filename: tissue_type for filename, tissue_type in zip(filenames, predicted_types)}
    return predictions_dict
@app.callback(
    Output('file-dropdown', 'options'),
    Output('file-dropdown', 'value'),
    Output('selected-files-store', 'data'),
    Output('predictions-store', 'data'),
    Input('upload-data', 'contents'),
    Input('select-all-button', 'n_clicks'),
    State('upload-data', 'filename'),
    State('file-dropdown', 'value'),
    State('selected-files-store', 'data'),
    State('csv-delimiter', 'value'),
    State('skiprows', 'value'),
)
def update_file_dropdown(contents, n_clicks, filenames, current_values, stored_values, delimiter, skiprows):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    global file_data
    triggered_id = ctx.triggered[0]['prop_id']
    if 'upload-data' in triggered_id:
        if contents is None:
            return [], [], {}, {}
        options = [{'label': filename, 'value': filename} for filename in filenames]
        current_values = current_values or stored_values or []
        file_data = {}
        for filename, content in zip(filenames, contents):
            try:
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                data = pd.read_csv(BytesIO(decoded), delimiter=delimiter, skiprows=skiprows)
                file_data[filename] = data
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
                continue
        if file_data:
            predictions_dict = predict_tissue_type(list(file_data.keys()))
        else:
            predictions_dict = {}
        return options, current_values, {filename: content for filename, content in zip(filenames, contents)}, predictions_dict
    elif 'select-all-button' in triggered_id:
        if n_clicks > 0 and filenames:
            options = [{'label': filename, 'value': filename} for filename in filenames]
            current_values = [option['value'] for option in options]
            file_data = {}
            for filename, content in zip(filenames, contents):
                try:
                    content_type, content_string = content.split(',')
                    decoded = base64.b64decode(content_string)
                    data = pd.read_csv(BytesIO(decoded), delimiter=delimiter, skiprows=skiprows)
                    file_data[filename] = data
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")
                    continue
            if file_data:
                predictions_dict = predict_tissue_type(list(file_data.keys()))
            else:
                predictions_dict = {}
            return options, current_values, {filename: content for filename, content in zip(filenames, contents)}, predictions_dict
        else:
            return [], [], {}, {}
    return [], [], {}, {}
@app.callback(
    Output('spectrum-dropdown', 'options'),
    Input('file-dropdown', 'value'),
)
def update_spectrum_dropdown(selected_files):
    if not selected_files:
        return []
    options = [{'label': filename, 'value': filename} for filename in selected_files]
    return options
@app.callback(
    Output('color-pickers', 'children'),
    Input('file-dropdown', 'value'),
    State('upload-data', 'filename')
)
def update_color_pickers(selected_files, filenames):
    selected_files = selected_files if selected_files is not None else []
    filenames = filenames if filenames is not None else []
    if not selected_files:
        selected_files = filenames
    color_pickers = [
        html.Div([
            html.P(filename, style={'margin-right': '10px'}),
            dbc.Input(id={'type': 'color-picker', 'index': filename}, type='color', value='#1f77b4', style={'width': '50px', 'height': '30px'})
        ], style={'margin-bottom': '10px', 'margin-right': '10px'})
        for filename in selected_files
    ]
    return color_pickers
def process_spectrum(spectrum, savgol_checked, savgol_window, savgol_order, baseline_checked, baseline_method, poly_order=None, bubble_width=None, als_lambda=None, als_p=None, norm_checked=None, norm_method_combined=None, quantile=None, reference_peak=None):
    if savgol_checked and savgol_window and savgol_order:
        try:
            spectrum[1, :] = savgol_filter(spectrum[1, :], savgol_window, savgol_order)
        except Exception as e:
            print(f"Error applying Savitzky-Golay filter: {e}")
    if baseline_checked:
        if baseline_method == 'poly' and poly_order:
            spectrum[1, :], _ = polynomial_fitting(spectrum, poly_order)
        elif baseline_method == 'bubble' and bubble_width:
            baseline = bubblefill(spectrum[1, :], bubble_width)[1]
            spectrum[1, :] = spectrum[1, :] - baseline
        elif baseline_method == 'als' and als_lambda is not None and als_p is not None:
            baseline = baseline_als_optimized(spectrum[1, :], lam=als_lambda, p=als_p)
            spectrum[1, :] = spectrum[1, :] - baseline
    if norm_checked and norm_method_combined:
        if norm_method_combined in ['max_minmax', 'quantile_minmax'] and quantile is not None:
            spectrum = normalize_spectrum_minmax(spectrum, quantile=float(quantile) if norm_method_combined == 'quantile_minmax' else 1.0)
        elif norm_method_combined in ['max_vector', 'quantile_vector']:
            spectrum = normalize_spectrum_vect(spectrum)
        elif norm_method_combined == 'sum':
            spectrum = normalize_spectrum_sum(spectrum)
        elif norm_method_combined == 'reference' and reference_peak is not None:
            spectrum = normalize_spectrum_reference(spectrum, reference_peak)
    return spectrum
def calculate_snr(spectrum):
    signal = np.mean(spectrum[1, :])
    noise = np.std(spectrum[1, :])
    return signal / noise if noise != 0 else 0
def update_graph_and_table(
    contents, selected_files, savgol_checked, savgol_window, savgol_order,
    baseline_checked, baseline_method, poly_order, bubble_width, als_lambda, als_p,
    norm_checked, norm_method_combined, quantile, reference_peak,
    detection_method, selected_tissue, peak_height,
    peak_distance, selected_spectrum, color_values,
    timestamp, color_pickers, delimiter, skiprows, x_min, x_max,
    figure, table_rows
):
    ctx = dash.callback_context
    if not ctx.triggered or contents is None or selected_files is None:
        return go.Figure(), go.Figure(), [], html.Div("Please upload files to begin analysis.", style={'textAlign': 'center',
                    'padding': '10px',
                    'backgroundColor': '#f8f9fa',
                    'margin': '10px 0',
                }), True, False, ""
    fig_original = go.Figure()
    fig_processed = go.Figure()
    feedback_message = "Files processed successfully."
    savgol_valid, savgol_invalid, savgol_feedback = True, False, ""
    if not color_values or len(color_values) != len(color_pickers):
        color_values = ['#1f77b4'] * len(selected_files)
    all_peaks_data = {}
    for idx, filename in enumerate(selected_files):
        try:
            content_index = selected_files.index(filename)
            content = contents[content_index]
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                tmp_file.write(decoded)
                tmp_file.flush()
                data = pd.read_csv(tmp_file.name, delimiter=delimiter, skiprows=skiprows)
            if data.empty:
                continue
            x_axis = data.iloc[:, 0].values
            original_spectrum = data.iloc[:, 1].values
            spectrum = np.vstack((x_axis, original_spectrum))
            if x_min is not None or x_max is not None:
                mask = np.ones_like(x_axis, dtype=bool)
                if x_min is not None and x_max is not None:
                    mask = (x_axis >= x_min) & (x_axis <= x_max)
                elif x_min is not None:
                    mask = x_axis >= x_min
                elif x_max is not None:
                    mask = x_axis <= x_max
                x_axis_truncated = x_axis[mask]
                original_spectrum_truncated = original_spectrum[mask]
                spectrum = spectrum[:, mask]
            else:
                x_axis_truncated = x_axis
                original_spectrum_truncated = original_spectrum
            # Ensure parameters are not None before passing them
            savgol_window = savgol_window if savgol_window is not None else 11
            savgol_order = savgol_order if savgol_order is not None else 3
            baseline_method = baseline_method if baseline_method else 'poly'
            poly_order = poly_order if poly_order is not None else 4
            bubble_width = bubble_width if bubble_width is not None else 50
            als_lambda = als_lambda if als_lambda is not None else 100000
            als_p = als_p if als_p is not None else 0.01
            spectrum = process_spectrum(
                spectrum,
                'apply' in savgol_checked if savgol_checked else False,
                savgol_window,
                savgol_order,
                'apply' in baseline_checked if baseline_checked else False,
                baseline_method,
                poly_order,
                bubble_width,
                als_lambda,
                als_p,
                'apply' in norm_checked if norm_checked else False,
                norm_method_combined,
                quantile,
                reference_peak,
            )
            peaks_positions, peaks_heights = [], []
            if detection_method == 'auto':
                peaks_positions, peaks_heights = find_raman_peaks(spectrum, height=float(peak_height) if peak_height else 0.5, distance=peak_distance if peak_distance else 20)
            elif detection_method == 'tissue' and selected_tissue:
                for substance, details in peak_assignments.items():
                    if selected_tissue in details.get("tissue", []):
                        variability_range = details.get("variability_range", [])
                        peak_position = details.get("peak_position")
                        if isinstance(variability_range, list) and len(variability_range) == 2:
                            lower_bound, upper_bound = variability_range
                            peak_indices = np.where((x_axis >= lower_bound) & (x_axis <= upper_bound))[0]
                            if len(peak_indices) > 0:
                                peak_index = peak_indices[np.argmax(spectrum[1, peak_indices])]
                                peaks_positions.append(x_axis[peak_index])
                                peaks_heights.append(spectrum[1, peak_index])
            all_peaks_data[filename] = (peaks_positions, peaks_heights)
            fig_original.add_trace(go.Scatter(
                x=x_axis_truncated,
                y=original_spectrum_truncated,
                mode='lines',
                name=f'Original Spectrum - {filename}',
                line=dict(color=color_values[idx])
            ))
            fig_processed.add_trace(go.Scatter(
                x=x_axis_truncated,
                y=spectrum[1, :],
                mode='lines',
                name=f'Processed Spectrum - {filename}',
                line=dict(color=color_values[idx])
            ))
            fig_processed.add_trace(go.Scatter(
                x=peaks_positions,
                y=[spectrum[1, np.where(x_axis_truncated == pos)[0][0]] for pos in peaks_positions],
                mode='markers',
                marker=dict(size=8, color='red'),
                name='Detected Peaks',
                customdata=np.stack((peaks_positions, peaks_heights), axis=-1),
                hovertemplate='<b>Peak Position:</b> %{customdata[0]:.2f}<br><b>Peak Height:</b> %{customdata[1]:.2f}'
            ))
        except Exception as e:
            logging.error(f"Error processing spectrum for file {filename}: {e}")
            continue
    num_spectra = len(fig_processed.data) // 2
    legend_y_position = -0.3 - (num_spectra * 0.05)
    fig_original.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=40, b=80),
        xaxis_title="Raman Shift (cm⁻¹)",
        yaxis_title="Intensity (a.u.)"
    )
    fig_processed.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=40, b=80),
        xaxis_title="Raman Shift (cm⁻¹)",
        yaxis_title="Intensity (a.u.)"
    )
    if selected_spectrum and selected_spectrum in all_peaks_data:
        peaks_positions, peaks_heights = all_peaks_data[selected_spectrum]
        table_rows = []
        for position, height in zip(peaks_positions, peaks_heights):
            matches = []
            for substance, details in peak_assignments.items():
                variability_range = details.get("variability_range", [])
                peak_position = details.get("peak_position")
                if isinstance(variability_range, list) and len(variability_range) == 2:
                    lower_bound, upper_bound = variability_range
                    if lower_bound <= position <= upper_bound:
                        difference = abs(position - peak_position)
                        matches.append((difference, details, substance))
            if matches:
                tissue_icons = []
                for difference, details, substance in matches:
                    for tissue in details.get("tissue", []):
                        tissue_icons.append(html.Img(src=f"/assets/Icons/{tissue}.png", title=tissue, style={'width': '20px', 'height': '20px', 'padding': '2px'}))
                best_match = matches[0][1]
                variability_range = f'{best_match["variability_range"][0]} - {best_match["variability_range"][1]}'
                biochemical_component = best_match.get("biochemical_component", "N/A")
                biochemical_component = " ; ".join(biochemical_component) if isinstance(biochemical_component, list) else biochemical_component
                table_rows.append(html.Tr([
                    html.Td(f'{position:.2f}', id={'type': 'table-row', 'index': str(position)}),
                    html.Td(f'{height:.2f}'),
                    html.Td(variability_range),
                    html.Td(biochemical_component),
                    html.Td(tissue_icons)
                ], id={'type': 'table-row', 'index': str(position)}))
    return fig_original, fig_processed, table_rows, feedback_message, savgol_valid, savgol_invalid, savgol_feedback
def update_snr(selected_files, baseline_method, savgol_checked, savgol_window, savgol_order,
               norm_checked, norm_method_combined, quantile, reference_peak,
               contents, filenames, delimiter, skiprows, x_min, x_max,
               baseline_checked, poly_order, bubble_width, als_lambda, als_p):
    if not selected_files or not contents:
        return []
    snr_values = []
    for filename, content in zip(filenames, contents):
        if filename in selected_files:
            try:
                content_type, content_string = content.split(',')
                decoded = base64.b64decode(content_string)
                with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
                    tmp_file.write(decoded)
                    tmp_file.flush()
                    data = pd.read_csv(tmp_file.name, delimiter=delimiter, skiprows=skiprows).to_numpy()
                x_axis = data[:, 0]
                original_spectrum = data[:, 1]
                spectrum = np.vstack((x_axis, original_spectrum))
                if x_min is not None or x_max is not None:
                    if x_min is not None and x_max is not None:
                        mask = (x_axis >= x_min) & (x_axis <= x_max)
                    elif x_min is not None:
                        mask = x_axis >= x_min
                    elif x_max is not None:
                        mask = x_axis <= x_max
                    else:
                        mask = np.ones_like(x_axis, dtype=bool)
                    x_axis_truncated = x_axis[mask]
                    original_spectrum_truncated = original_spectrum[mask]
                    spectrum = spectrum[:, mask]
                spectrum = process_spectrum(
                    spectrum,
                    'apply' in savgol_checked if savgol_checked else False,
                    savgol_window,
                    savgol_order,
                    'apply' in baseline_checked if baseline_checked else False,
                    baseline_method,
                    poly_order,
                    bubble_width,
                    als_lambda,
                    als_p,
                    'apply' in norm_checked if norm_checked else False,
                    norm_method_combined,
                    quantile,
                    reference_peak
                )
                snr = calculate_snr(spectrum)
                snr_values.append((filename, snr))
            except Exception as e:
                logging.error(f"Error calculating SNR for file {filename}: {e}")
                continue
    snr_output = [html.P(f"{filename}: {snr:.2f}") for filename, snr in snr_values]
    return snr_output
@app.callback(
    [
        Output('original-spectrum', 'figure'),
        Output('processed-spectrum', 'figure'),
        Output('table-body', 'children'),
        Output('feedback-message', 'children'),
        Output('savgol-window', 'valid'),
        Output('savgol-window', 'invalid'),
        Output('savgol-window-feedback', 'children'),
        Output('snr-output', 'children'),
        Output('tissue-type-output', 'children')
    ],
    [
        Input('upload-data', 'contents'),
        Input('file-dropdown', 'value'),
        Input('savgol-checkbox', 'value'),
        Input('savgol-window', 'value'),
        Input('savgol-order', 'value'),
        Input('baseline-checkbox', 'value'),
        Input('baseline-method', 'value'),
        Input('poly-order', 'value'),
        Input('bubble-width', 'value'),
        Input('als-lambda', 'value'),
        Input('als-p', 'value'),
        Input('norm-checkbox', 'value'),
        Input('norm-method-combined', 'value'),
        Input('quantile-input', 'value'),
        Input('reference-peak-input', 'value'),
        Input('peak-detection-method', 'value'),
        Input('tissue-dropdown', 'value'),
        Input('peak-height', 'value'),
        Input('peak-distance', 'value'),
        Input('spectrum-dropdown', 'value'),
        Input({'type': 'color-picker', 'index': ALL}, 'value'),
        Input('table-body', 'n_clicks_timestamp'),
    ],
    [
        State('color-pickers', 'children'),
        State('csv-delimiter', 'value'),
        State('skiprows', 'value'),
        State('x-min', 'value'),
        State('x-max', 'value'),
        State('processed-spectrum', 'figure'),
        State('table-body', 'children'),
        State('predictions-store', 'data')
    ],
    prevent_initial_call=True
)
def update_output_and_snr(
    contents, selected_files, savgol_checked, savgol_window, savgol_order,
    baseline_checked, baseline_method, poly_order, bubble_width, als_lambda, als_p,
    norm_checked, norm_method_combined, quantile, reference_peak,
    detection_method, selected_tissue, peak_height,
    peak_distance, selected_spectrum, color_values,
    timestamp, color_pickers, delimiter, skiprows, x_min, x_max,
    figure, table_rows, predictions
):
    fig_original, fig_processed, table_rows, feedback_message, savgol_valid, savgol_invalid, savgol_feedback = update_graph_and_table(
        contents, selected_files, savgol_checked, savgol_window, savgol_order,
        baseline_checked, baseline_method, poly_order, bubble_width, als_lambda, als_p,
        norm_checked, norm_method_combined, quantile, reference_peak,
        detection_method, selected_tissue, peak_height,
        peak_distance, selected_spectrum, color_values,
        timestamp, color_pickers, delimiter, skiprows, x_min, x_max,
        figure, table_rows
    )
    snr_output = update_snr(
        selected_files, baseline_method, savgol_checked, savgol_window, savgol_order,
        norm_checked, norm_method_combined, quantile, reference_peak,
        contents, selected_files, delimiter, skiprows, x_min, x_max,
        baseline_checked, poly_order, bubble_width, als_lambda, als_p
    )
    tissue_type_output = []
    if predictions:
        tissue_type_output = html.Div([
            html.Ul([html.Li(f"{filename}: {tissue_type}") for filename, tissue_type in predictions.items()])
        ])
    return fig_original, fig_processed, table_rows, feedback_message, savgol_valid, savgol_invalid, savgol_feedback, snr_output, tissue_type_output
@app.callback(
    Output("download-raw-data", "data"),
    Input("export-raw-button", "n_clicks"),
    State('file-dropdown', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('csv-delimiter', 'value'),
    State('skiprows', 'value'),
    prevent_initial_call=True
)
def export_raw_data(n_clicks, selected_files, contents, filenames, delimiter, skiprows):
    if n_clicks is None or not selected_files:
        raise PreventUpdate
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for filename in selected_files:
            content = contents[filenames.index(filename)]
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            data = pd.read_csv(BytesIO(decoded), delimiter=delimiter, skiprows=skiprows)
            csv_buffer = StringIO()
            data.to_csv(csv_buffer, index=False)
            zip_file.writestr(f"{filename.split('.')[0]}_raw.csv", csv_buffer.getvalue())
    zip_file.close()
    return dcc.send_bytes(zip_buffer.getvalue(), "raw_data.zip")
@app.callback(
    Output("download-processed-data", "data"),
    Input("export-processed-button", "n_clicks"),
    State('file-dropdown', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('csv-delimiter', 'value'),
    State('skiprows', 'value'),
    State('savgol-checkbox', 'value'),
    State('savgol-window', 'value'),
    State('savgol-order', 'value'),
    State('baseline-checkbox', 'value'),
    State('baseline-method', 'value'),
    State('poly-order', 'value'),
    State('bubble-width', 'value'),
    State('als-lambda', 'value'),
    State('als-p', 'value'),
    State('norm-checkbox', 'value'),
    State('norm-method-combined', 'value'),
    State('quantile-input', 'value'),
    State('reference-peak-input', 'value'),
    State('x-min', 'value'),
    State('x-max', 'value'),
    prevent_initial_call=True
)
def export_processed_data(n_clicks, selected_files, contents, filenames, delimiter, skiprows, savgol_checked, savgol_window, savgol_order, baseline_checked, baseline_method, poly_order, bubble_width, als_lambda, als_p, norm_checked, norm_method_combined, quantile, reference_peak, x_min, x_max):
    if n_clicks is None or not selected_files:
        raise PreventUpdate
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for filename in selected_files:
            content = contents[filenames.index(filename)]
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            data = pd.read_csv(BytesIO(decoded), delimiter=delimiter, skiprows=skiprows)
            x_axis = data.iloc[:, 0].values
            original_spectrum = data.iloc[:, 1].values
            spectrum = np.vstack((x_axis, original_spectrum))
            if x_min is not None or x_max is not None:
                mask = np.ones_like(x_axis, dtype=bool)
                if x_min is not None and x_max is not None:
                    mask = (x_axis >= x_min) & (x_axis <= x_max)
                elif x_min is not None:
                    mask = x_axis >= x_min
                elif x_max is not None:
                    mask = x_axis <= x_max
                spectrum = spectrum[:, mask]
            spectrum = process_spectrum(spectrum, 'apply' in savgol_checked, savgol_window, savgol_order, 'apply' in baseline_checked, baseline_method, poly_order, bubble_width, als_lambda, als_p, 'apply' in norm_checked, norm_method_combined, quantile, reference_peak)
            processed_data = pd.DataFrame({'Raman Shift': spectrum[0, :], 'Intensity': spectrum[1, :]})
            csv_buffer = StringIO()
            processed_data.to_csv(csv_buffer, index=False)
            zip_file.writestr(f"{filename.split('.')[0]}_processed.csv", csv_buffer.getvalue())
    zip_file.close()
    return dcc.send_bytes(zip_buffer.getvalue(), "processed_data.zip")
@app.callback(
    Output("download-all-processed-data", "data"),
    Input("export-all-processed-button", "n_clicks"),
    State('file-dropdown', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('csv-delimiter', 'value'),
    State('skiprows', 'value'),
    State('savgol-checkbox', 'value'),
    State('savgol-window', 'value'),
    State('savgol-order', 'value'),
    State('baseline-checkbox', 'value'),
    State('baseline-method', 'value'),
    State('poly-order', 'value'),
    State('bubble-width', 'value'),
    State('als-lambda', 'value'),
    State('als-p', 'value'),
    State('norm-checkbox', 'value'),
    State('norm-method-combined', 'value'),
    State('quantile-input', 'value'),
    State('reference-peak-input', 'value'),
    State('x-min', 'value'),
    State('x-max', 'value'),
    prevent_initial_call=True
)
def export_all_processed_data(n_clicks, selected_files, contents, filenames, delimiter, skiprows, savgol_checked, savgol_window, savgol_order, baseline_checked, baseline_method, poly_order, bubble_width, als_lambda, als_p, norm_checked, norm_method_combined, quantile, reference_peak, x_min, x_max):
    if n_clicks is None or not selected_files:
        raise PreventUpdate
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for filename in selected_files:
            content = contents[filenames.index(filename)]
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            data = pd.read_csv(BytesIO(decoded), delimiter=delimiter, skiprows=skiprows)
            x_axis = data.iloc[:, 0].values
            original_spectrum = data.iloc[:, 1].values
            spectrum = np.vstack((x_axis, original_spectrum))
            if x_min is not None or x_max is not None:
                mask = np.ones_like(x_axis, dtype=bool)
                if x_min is not None and x_max is not None:
                    mask = (x_axis >= x_min) & (x_axis <= x_max)
                elif x_min is not None:
                    mask = x_axis >= x_min
                elif x_max is not None:
                    mask = x_axis <= x_max
                spectrum = spectrum[:, mask]
            spectrum = process_spectrum(spectrum, 'apply' in savgol_checked, savgol_window, savgol_order, 'apply' in baseline_checked, baseline_method, poly_order, bubble_width, als_lambda, als_p, 'apply' in norm_checked, norm_method_combined, quantile, reference_peak)
            processed_data = pd.DataFrame({'Raman Shift': spectrum[0, :], 'Intensity': spectrum[1, :]})
            csv_buffer = StringIO()
            processed_data.to_csv(csv_buffer, index=False)
            zip_file.writestr(f"{filename.split('.')[0]}_processed.csv", csv_buffer.getvalue())
    zip_file.close()
    return dcc.send_bytes(zip_buffer.getvalue(), "all_processed_data.zip")
@app.callback(
    Output("download-table-data", "data"),
    Input("export-table-button", "n_clicks"),
    State('spectrum-dropdown', 'value'),
    State('table-body', 'children'),
    prevent_initial_call=True
)
def export_table_data(n_clicks, selected_spectrum, table_rows):
    if n_clicks is None or not selected_spectrum:
        raise PreventUpdate
    data = []
    for row in table_rows:
        cells = row.get('props', {}).get('children', [])
        if len(cells) >= 5:
            peak_position = cells[0].get('props', {}).get('children', '')
            peak_height = cells[1].get('props', {}).get('children', '')
            variability_range = cells[2].get('props', {}).get('children', '')
            biochemical_component = cells[3].get('props', {}).get('children', '')
            tissue = cells[4].get('props', {}).get('children', '')
            data.append({
                'Peak Position': peak_position,
                'Peak Height': peak_height,
                'Variability Range': variability_range,
                'Biochemical Component': biochemical_component,
                'Tissue': tissue
            })
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, f"{selected_spectrum}_peaks_table.csv", index=False)
@app.callback(
    Output("download-all-tables-data", "data"),
    Input("export-all-tables-button", "n_clicks"),
    State('file-dropdown', 'value'),
    State('table-body', 'children'),
    prevent_initial_call=True
)
def export_all_tables_data(n_clicks, selected_files, table_rows):
    if n_clicks is None or not selected_files:
        raise PreventUpdate
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for filename in selected_files:
            data = []
            for row in table_rows:
                cells = row.get('props', {}).get('children', [])
                if len(cells) >= 5:
                    peak_position = cells[0].get('props', {}).get('children', '')
                    peak_height = cells[1].get('props', {}).get('children', '')
                    variability_range = cells[2].get('props', {}).get('children', '')
                    biochemical_component = cells[3].get('props', {}).get('children', '')
                    tissue = cells[4].get('props', {}).get('children', '')
                    data.append({
                        'Peak Position': peak_position,
                        'Peak Height': peak_height,
                        'Variability Range': variability_range,
                        'Biochemical Component': biochemical_component,
                        'Tissue': tissue
                    })
            df = pd.DataFrame(data)
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            zip_file.writestr(f"{filename.split('.')[0]}_peaks_table.csv", csv_buffer.getvalue())
    zip_file.close()
    return dcc.send_bytes(zip_buffer.getvalue(), "all_peaks_tables.zip")
if __name__ == '__main__':
    app.run(debug=True)