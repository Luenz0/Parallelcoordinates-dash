import pandas as pd
import plotly.graph_objects as go

from matplotlib.ticker import FuncFormatter
import io
#%matplotlib inline

import os

#from jupyter_dash import JupyterDash
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output


# Jupyter notebook to create ParallelCoordinates (PC) from Parametric runs in IDAICE
# the IDA ICE output should be stored as .csv inside a folder named IDAICE_results
# the PC will be stored in a new folder



color_var = -1


def preprocessing(df):
    #print(df)
    ## PREPROCESSING ##
    
    # Find the column name matching the regex pattern '^Unnamed'
    empty_column_name = df.columns[df.columns.to_series().str.match('^Unnamed')].max()

    if empty_column_name is not None:
        # Find the index of the empty column
        empty_index = df.columns.get_loc(empty_column_name)

        # Split the columns into inputs and outputs based on the empty_index
        inputs = list(df.columns[1:empty_index])
        n_inputs =  len(inputs)
        outputs = list(df.columns[empty_index + 1:])
        n_outputs = len(outputs)
        #print("Inputs:", len(inputs))
        #print("Outputs:", len(outputs))
    #else:
        #print("No column matching the pattern found.")

    #Drop unnecessary columns
    df.drop(columns=(['Name'] + list(df.filter(regex='Unnamed'))), inplace = True)

    #identify categorical columns
    cols_to_label = set(df.columns) - set(df._get_numeric_data().columns)
    cols = list(cols_to_label)

    # Get labels to encode
    dict_labels = {col: {n: cat for n, cat in enumerate(df[col].astype('category').cat.categories)} for col in df[cols]}
    #print(dict_labels)

    #Label-Encode categorical columns
    df[cols] = pd.DataFrame({col: df[col].astype('category').cat.codes for col in df[cols]}, index=df.index)

    return df, cols, dict_labels, n_inputs, n_outputs

    

def get_data_PC(df, cat_cols, dict_labels, n_inputs, n_outputs):
    
    cols = cat_cols

    ## Data for parallel coordinates ##

    #Empty start
    data_PC = []
    
    ################################ PC 8 inputs and 8 outpus ###########################################
    
    #Empty dimension for missing variables
    empty_dimension = [dict(range = [0, 10], label = "--", tickvals=[0, 10],  ticktext = [' ', ' '],values = pd.Series([5] * len(df[df.columns[0]])))]

    #Define Empty inputs
    emp_inp = 8 - n_inputs
    
    #Define Empty outputs
    emp_out = 8 - n_outputs
    
        
    #Fill Empty inputs
    for i in range(emp_inp):
        data_PC = data_PC + empty_dimension
        
    #Drop extra logged inputs, maximum 8, drops the last items
    if emp_inp < 0:
        columns_to_drop = df.columns[8:8+abs(emp_inp)]  
        df.drop(columns=columns_to_drop, inplace=True) 
        n_inputs = 8
        #print(df)
        
    #Drop extra logged outputs, maximum 8, drops the last items
    if emp_out < 0:
        columns_to_drop = df.columns[emp_inp:]
        #print("Dropping")
        #print(columns_to_drop)
        df.drop(columns=columns_to_drop, inplace=True) 
        n_outputs = 8
        #print(df)    
        
    
    # Fill the PC with available variables max 8 inputs and max 8 outputs-    
    for n, col in enumerate(df):
                         
        val_max = round(df[col].max() + 0.045, 1) if df[col].max() < 1 else round(df[col].max() + 0.45, 0)
        val_min = abs(round(df[col].min()-0.45,0))
        
        
        
        ticktext = list(dict_labels[col].values()) if col in cat_cols else None
        tickvals = list(range(0, len(ticktext))) if ticktext and len(ticktext) > 0 else None
        
        
        new_dimension = [dict(range = [val_min, val_max], label = col, values = df[col] , tickvals = tickvals, ticktext = ticktext)]
        
        ###### Separate inputs from outputs #######
        if n == n_inputs-1:
            
            ### Aproach A: the last input is repeated with no ticktext
            extra = [dict(range = [val_min, val_max], 
                          label = " ", 
                          values = df[col] , 
                          tickvals = [val_min, val_max], 
                          ticktext = [' ', ' '])]
            new_dimension = new_dimension + extra
            

        data_PC = data_PC + new_dimension
     
    #Fill Empty outputs
    for i in range(emp_out):
        data_PC = data_PC + empty_dimension
        
    return df, data_PC




def Plot_parcoords(filename, color_col, reverse_color):
    
    reverse = True if 'reverse' in reverse_color else False
    
    #Read the file
    df = pd.read_csv(os.path.join(idaice_results_directory, filename)) 
        
    #preprocess (return df and cat columns and its labels)
    df, cat_cols, dict_labels, n_inputs, n_outputs = preprocessing(df)    #df:values, cat_cols:categorical index, dictlabels:text 
    df, data_PC = get_data_PC(df, cat_cols, dict_labels, n_inputs, n_outputs)

    #line = set_linecolor(df.columns[color_var]) # For the color scale the last column is used as reference
    line = dict(color = df[color_col],
                colorscale = 'Electric',
                reversescale = reverse,
                showscale = True,
                          cmin = df[color_col].min(),
                          cmax = df[color_col].max()
               )



    ## Figure ##
    fig = go.Figure(data=
    go.Parcoords(

            line=line,
            dimensions = data_PC,
            labelangle = 15,   
       )
    )
    
    return fig



# assign directory
# the IDA ICE output should be stored as .csv inside a folder named IDAICE_results'
current_directory = os.path.dirname(os.path.abspath(__file__))
idaice_results_directory = os.path.join(current_directory, 'IDAICE_results')
save_directory = 'ParallelCoordinates_Plots'

# Create a Dash app
app = dash.Dash(__name__)
server = app.server


file_dropdown_options = [{'label': filename, 'value': filename} for filename in os.listdir(idaice_results_directory) if filename.endswith('.csv')]
#Starting point
first_file = file_dropdown_options[0]['value']

#print(first_file)
df = pd.read_csv(os.path.join(idaice_results_directory, first_file))
print(df)


##################################################################
# Define the layout of the app
app.layout = html.Div([
    html.H1("Parallel Coordinates Plots - LS"),
    
    html.Br(),
    html.Label("Select File"),
    dcc.Dropdown(
        id='file_dropdown',
        options=file_dropdown_options,
        value=first_file
    ),
    
    html.Br(),
    html.Label("Select Color"),
    dcc.Dropdown(
        id='col_dropdown',
    ),
    
    html.Br(),
        html.Label("Reverse Colorscale"),
        dcc.Checklist(
            id='reverse_color',
            options=[{'label': 'Reverse', 'value': 'reverse'}],
            value=[]
        ),

    
    dcc.Graph(
        id='parcoord-graph',
        figure=Plot_parcoords(first_file, df.columns[8], [False]),
        #style={'width': '1500px', 'height': '600px'}
    ),
    
     html.Button("Save as HTML", id="save_as_html", n_clicks=0)
])
##################################################################



##################################################################
# Define callback to update the file based on dropdown selection
@app.callback(
    Output('col_dropdown', 'options'),
    Output('col_dropdown', 'value'),
    Input('file_dropdown', 'value')
)

def update_dropdown(selected_filename):
    df = pd.read_csv(os.path.join(idaice_results_directory, selected_filename))
    df, cat_cols, dict_labels, n_inputs, n_outputs = preprocessing(df)
    df, data_PC = get_data_PC(df, cat_cols, dict_labels, n_inputs, n_outputs)
    #print(df)
    col_dropdown_options = [{'label': col, 'value': col} for col in df.columns]
    default_value = df.columns[-1] if len(df.columns) > 0 else None
    return col_dropdown_options, default_value

##################################################################
# Define callback to update the color and figure based on dropdown selection
@app.callback(
    Output('parcoord-graph', 'figure'),
    Input('file_dropdown', 'value'),
    Input('col_dropdown', 'value'),
    Input('reverse_color', 'value')
)


def update_figure(selected_filename, selected_column, reverse_color):
    print(selected_filename)
    print(selected_column)
    return Plot_parcoords(selected_filename, selected_column, reverse_color)

##################################################################
# Callback to save as HTML file
@app.callback(
    Output('save_as_html', 'n_clicks'),
    Input('save_as_html', 'n_clicks'),
    Input('parcoord-graph', 'figure'),
    prevent_initial_call=True
)
def save_as_html(n_clicks, fig):
    if n_clicks is not None and n_clicks > 0:
        

        figure=go.Figure(fig)
        
        
        if isinstance(figure, go.Figure):
            figure.write_html('current_figure_export.html')
            print("HTML file saved successfully")
        else:
            print("Error: Invalid figure object received")

        
    return n_clicks



# Run the app in the notebook
app.run_server(debug=True)