import wfdb
import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy import signal
import json
import pickle
import time
from sklearn.decomposition import PCA

from bokeh.plotting import figure, curdoc
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.models import ColumnDataSource, PreText, Div, LegendItem, DataRange1d,RadioButtonGroup, Spinner, Select, HoverTool, CrosshairTool, CustomJS
from bokeh.layouts import gridplot, column, row
from bokeh import events
from bokeh.events import RangesUpdate

## Variable
SIGNAL_SIZE = 2000

## Functions
def filt_bandpass(sig,fs):
    '''
    Filtro Butterworth passa banda, ordem 3. Frequência de corte 1hz e 50hz.
    :param sig: sinal a ser filtrado
    :param fs: frequência de amostragem do sinal.
    :return sig_filt: sinal filtrado.
    '''
    
    nyq = 0.5*fs # frequência de Nyquist
    ordem = 3 # ordem do filtro
    fh = 50
    fl = 1
    high = fh/nyq
    low = fl/nyq
    b, a = signal.butter(ordem, [low, high], btype='band')
    sig_filt = signal.filtfilt(b, a, sig, axis = 0)
        
    return sig_filt

def make_matrix(signal):
    '''
    Função que recebe um sinal e retorna a matriz de distâncias.
    :param signal: sinal a ser transformado em matriz de distâncias.
    :return l: matriz de distâncias.
    '''
    if switch2.active == 0:
        signal = signal[:,:12]
    elif switch2.active == 1:
        signal = signal[:,12:]
    elif switch2.active == 2:
        sig = signal[:,:12]
        pca = PCA(n_components=12)
        pca.fit(sig)
        somatorio = np.cumsum(pca.explained_variance_ratio_)
        num_channels = np.argmax(somatorio > 0.94)
        signal = signal[:,:num_channels]

    diffs = signal[:, np.newaxis, :] - signal[np.newaxis, :, :]
    dists = norm(diffs, axis=2)
    l = None
    del l
    # Concatenate distances vertically
    l = np.concatenate((l, dists), axis=0) if 'l' in locals() else dists
    return l

def read_signal(patient, hea, n_channels = 15):
    '''
    Função que recebe o nome do paciente e o nome do exame e retorna o sinal e os campos do exame.	
    :param patient: nome do paciente.
    :param hea: nome do exame.
    :param n_channels: número de canais do exame.
    :return signal: sinal do exame.
    '''
    signal, fields = wfdb.rdsamp(f'data/ptb-diagnostic-ecg-database-1.0.0/{patient}/{hea}', channels=[*range(n_channels)])
    return signal, fields


def generate_hist(event):
    '''
    Função que recebe um evento e atualiza o histograma.
    :param event: evento que ocorreu.
    '''
    global quantile_vbars
    global matrix
    x_range, y_range = p4.x_range, p4.y_range
    xmin, xmax = x_range.start, x_range.end
    ymax, ymin = y_range.start, y_range.end
    
    xmin = max(0, int(xmin))
    xmax = min(SIGNAL_SIZE, int(xmax))
    ymin = max(0, int(ymin))   
    ymax = min(SIGNAL_SIZE, int(ymax))
    
    # Update the histogram data based on the current image view
    rounded_signal = np.round(matrix,3)
    rounded_signal = rounded_signal[int(ymin):int(ymax), int(xmin):int(xmax)]

    ## Update Metrics
    quants = [0.1, 0.25, 0.5, 0.75, 0.9]
    max_value = np.max(rounded_signal)
    min_value = np.min(rounded_signal) 
    quantiles = np.quantile(rounded_signal, quants)
    p.text =f"Valor max: {max_value:.2f}\nValor min: {min_value:.2f}\n\
10%:{quantiles[0]:.2f}\n25%: {quantiles[1]:.2f}\n50%: {quantiles[2]:.2f}\n\
75%:{quantiles[3]:.2f}\n90%: {quantiles[4]:.2f}"

    hist_values, hist_bins = np.histogram(rounded_signal,bins=6200)

    for i, barv in enumerate(quantile_vbars):
        barv.glyph.x = quantiles[i]
        barv.glyph.top = max(hist_values)
        new_label = f"Quantile {quants[i]*100}%: {quantiles[i]:.2f}"
        p5.legend[0].items[i] = (LegendItem(label=new_label, renderers=[barv]))

    hist = p5.renderers[0] 
    hist.data_source.data = dict(top=hist_values, bottom=[0] * len(hist_values), left=hist_bins[:-1], right= hist_bins[1:])


def update_spinner_end(attrname, old, new):
    '''
    Função que recebe um evento e atualiza o spinner de fim.
    '''
    global spinner_end
    if spinner_start.value + SIGNAL_SIZE <= spinner_start.high:
        spinner_end = spinner_start.value + SIGNAL_SIZE
    else: 
        spinner_end = spinner_start.high


def callback(event):
    '''
    Função que recebe um evento e atualiza o gráfico de distâncias.
    '''
    global matrix
    if event.x and event.y:
        x_value = event.y 
        y_value = event.x  # Invert y due to plot coordinate system

        p11_line = p11.renderers[0] 
        p11_line.data_source.data = dict(x=matrix[:, int(y_value)],y=list(range(0,SIGNAL_SIZE)))

        p1_line = p1.renderers[0] 
        p1_line.data_source.data = dict(x=list(range(0,SIGNAL_SIZE)),y=matrix[int(x_value),:])

def update_patient(attrname, old, new):
    '''
    Função que recebe um evento e atualiza o dropdown de pacientes.
    '''
    selected_patology = dropdown_patology.value
    if selected_patology != "All":  
        dropdown_pat.options = ["None"]+ list(df[df['Reason for admission'] == selected_patology]["idP"])
    else:
        dropdown_pat.options = ["None"]+ list(df["idP"])

def update_heas(attrname, old, new):
    '''
    Função que recebe um evento e atualiza o dropdown de exames.
    '''
    selected_patient = dropdown_pat.value
    dropdown_exam.options = ["None"]+data[selected_patient]["heas"]

def update_signal(attrname, old, new):
    '''
    Função que recebe um evento e atualiza o sinal.
    '''
    global signals_t
    global len_signal
    global spinner_end

    selected_signal = dropdown_exam.value
    filtered = switch3.active
    if spinner_start.value == spinner_end:
        signals_t = np.zeros((SIGNAL_SIZE,15))
    elif selected_signal != "None":
        signals_t,_ = read_signal(dropdown_pat.value,selected_signal.replace(".hea",""))
        if filtered == 1:
            signals_t = filt_bandpass(signals_t,1000)
    else:
        signals_t = np.zeros((SIGNAL_SIZE,15))

def generate_matrix(attrname, old, new):
    '''
    Função que recebe um evento e atualiza a matriz de distâncias.
    '''
    global signals_t
    global matrix
    global spinner_end

    start_time = time.time()
    signals = signals_t[spinner_start.value:spinner_end]
    signal_channel = signals[:, spinner_channel.value]
    source.data = dict(x=np.arange(len(signal_channel)), y=signal_channel)
    matrix = make_matrix(signals)
    #matrix = np.flipud(matrix1)
    hover_tool.callback = CustomJS(code=hover_code, args={'matrix': matrix})
    if spinner_end - spinner_start.value == SIGNAL_SIZE or spinner_end - spinner_start.value == 0: 
        p4.title.text = f"{dropdown_pat.value} | {dropdown_patology.value} | {dropdown_exam.value} | Channel: {spinner_channel.value} | Start: {spinner_start.value} | End: {spinner_end}"
        if spinner_end - spinner_start.value == 0:
            p4.image(image=[np.zeros((SIGNAL_SIZE,SIGNAL_SIZE))], x=0, y=0, dw=SIGNAL_SIZE, dh=SIGNAL_SIZE, palette="Turbo256")
        else:
            p4.image(image=[matrix], x=0, y=0, dw=SIGNAL_SIZE, dh=SIGNAL_SIZE, palette="Turbo256")

        p2_line = p2.renderers[0] 
        p2_line.data_source.data = dict(x=np.arange(len(signal_channel)), y=signal_channel)

        p3_line = p3.renderers[0] 
        p3_line.data_source.data = dict(x=signal_channel, y=np.arange(len(signal_channel)))

        switch.active = 0
        generate_hist(event=None)
    end_time = time.time()
    print(f"Tempo de execução: {end_time - start_time}")

def generate_matrix_thresh(attrname, old, new):
    '''
    Função que recebe um evento e atualiza a matriz de distâncias com limiar.
    '''
    global matrix
    global spinner_end
    if switch.active == 1:
        mask = (matrix < thresh_value_up.value) & (matrix > thresh_value_down.value)
        mask = (~mask*255).astype(np.uint8)

        p4.title.text = f"{dropdown_pat.value} | {dropdown_patology.value} | {dropdown_exam.value} | Channel: {spinner_channel.value} | Start: {spinner_start.value} | End: {spinner_end}"
        p4.image(image=[mask], x=0, y=0, dw=SIGNAL_SIZE, dh=SIGNAL_SIZE, palette="Greys256")
    else:
        p4.image(image=[matrix], x=0, y=0, dw=SIGNAL_SIZE, dh=SIGNAL_SIZE, palette="Turbo256")


## MAIN
global signals_t
global quantile_vbars
global spinner_end

## Styles
spinner_style = {
    'font': 'helvetica',   # Change to your preferred font
    'font_size': '14pt',    # Adjust font size
    'text_color': '#333333',  # Set text color
    'margin': '5px',         # Add margin
    'width': '200px',        # Set width
}
dropdown_style = {
    'font': 'helvetica',   # Change to your preferred font
    'font_size': '14pt',    # Adjust font size
    'text_color': '#333333',  # Set text color
    'margin': '5px',         # Add margin
    'width': '200px',        # Set width
}
div_style = {
    'font': 'helvetica',   # Change to your preferred font
    'font_size': '20pt',    # Adjust font size
    'text_color': '#333333',  # Set text color
    'margin': '5px',         # Add margin
    'width': '1000px',        # Set width
}


## Reading info data
df = pd.read_csv('data/Record_info.csv')
with open('data/patient_heas.json') as json_file:
    data = json.load(json_file)

## Texts
div_form = Div(text="<b>Selecione as opções para gerar a visualização</b>", width=500, height=20,styles=div_style)
div_visualization = Div(text="<b>Visualização do resultado</b>", width=300, height=20,styles=div_style)

## Switch
LABELS = ["UTRP", "TRP"]
label_div1 = Div(text="<b>Tipos de matriz</b>", width=300, height=20)
switch = RadioButtonGroup(labels=LABELS, active=0)

LABELS2 = ["12d", "3d","PCA"]
label_div2 = Div(text="<b>Canais</b>", width=300, height=20)
switch2 = RadioButtonGroup(labels=LABELS2, active=0)

LABELS3 = ["Sem filtro","Filtro passa banda"]
# Add a label using a Div element
label_div3 = Div(text="<b>Filtro</b>", width=300, height=20)
switch3 = RadioButtonGroup(labels=LABELS3, active=0)
 
## Reading the signal and creating recurrence plot
fs = 1000
signals_t,_ = read_signal(list(data.keys())[0],data[list(data.keys())[0]]["heas"][0].replace(".hea",""))
signals = signals_t[0:SIGNAL_SIZE]
signal_channel = signals[:, 0]
matrix = make_matrix(signals)
#matrix1 = make_matrix(signals)
#matrix = np.flipud(matrix1) 
rounded_signal = np.round(matrix,3)
hist_values, hist_bins = np.histogram(rounded_signal,bins=6200)
max_value = np.max(matrix)
min_value = np.min(matrix)
quantiles = np.quantile(matrix, [0.1, 0.25, 0.5, 0.75, 0.9])

# Save matrix in a raw file
'''file_path = "matrix_data.pkl"
output = open(file_path, 'wb')
pickle.dump(matrix, output)
output.close()'''

# Creating Widgets for app

## Hover tool
hover_code = """
    const geometry = cb_data.geometry;
    const x = geometry.x;
    const y = geometry.y;
"""

hover_tool = HoverTool(tooltips=[("Value", "@image"),("(x,y)", "($x, $y)")],callback=CustomJS(args={'matrix': matrix}, code=hover_code))

## Dropdown patology
valid_options_patology = pd.notna(df['Reason for admission'])
dropdown_patology = Select(title="Selecione a Patologia", value="All", options=["All"] + list(df.loc[valid_options_patology, 'Reason for admission'].unique()),styles=dropdown_style)

## Dropdown patient
dropdown_pat = Select(title="Selecione o Paciente", value=list(df['idP'])[0], options=list(df['idP']),styles=dropdown_style)
selected_patient = dropdown_pat.value

## Dropdown exam
dropdown_exam = Select(title="Selecione o Exame", value=data[selected_patient]["heas"][0], options=data[selected_patient]["heas"],styles=dropdown_style)


## Spinner
spinner_start = Spinner(title="Selecione o ponto de inicio do sinal:", low=0, high=signals_t.shape[0], step=1, value=0, styles=spinner_style)
spinner_end = spinner_start.value + SIGNAL_SIZE# Spinner(title="S o ponto de término do sinal:", low=0, high=signals_t.shape[0], step=1, value=2000)
spinner_channel = Spinner(title="Selecione um Canal :", low=0, high=signals_t.shape[1]-1, step=1, value=0, styles=spinner_style)
thresh_value_down = Spinner(title="Escolha um limiar inferior para o gráfico de recorrência",low=-1, high=20, step=0.01, value = -1, styles=spinner_style)
thresh_value_up = Spinner(title="Escolha um limiar superior para o gráfico de recorrência",low=0, high=20, step=0.01, value = 0.3, styles=spinner_style)

## Crosshair
crosshair = CrosshairTool(dimensions='both') 

# Configure callbacks
dropdown_patology.on_change('value', update_patient)
dropdown_pat.on_change('value', update_heas)

dropdown_exam.on_change('value', update_signal)
dropdown_exam.on_change('value', generate_matrix)

spinner_start.on_change('value', update_spinner_end)
spinner_start.on_change('value', generate_matrix)

#spinner_end.on_change('value', generate_matrix)
spinner_channel.on_change('value', generate_matrix)

thresh_value_up.on_change('value', generate_matrix_thresh)
thresh_value_down.on_change('value', generate_matrix_thresh)

switch.on_change('active', generate_matrix_thresh)
switch2.on_change('active', generate_matrix)
switch3.on_change('active', update_signal)
switch3.on_change('active', generate_matrix)

# Chosing a random signal to initiate the app
## Create a Bokeh ColumnDataSource with the initial signal data
hist_source = ColumnDataSource(data=dict(top=[], bins=[]))
source = ColumnDataSource(data=dict(x=np.arange(len(signal_channel)), y=signal_channel))
y_range = DataRange1d(start=0, end=len(signal_channel)-1)

tools = ["pan","wheel_zoom","box_zoom","reset","save","help"]

## Create figures for each object
p1 = figure(title="Gráfico de distâncias horizontal",width=800, height=400,tools=tools)
p11 = figure(title="Gráfico de distâncias vertical",width=400, height=800,tools=tools)
p2 = figure(title="Sinal na horizontal", width=800, height=300,tools=tools)
p3 = figure(title="Sinal na vertical", width=300, height=800,tools=tools)
p4 = figure(title="Matriz de distância", width=800, height=800, x_range=(0, SIGNAL_SIZE), y_range=(SIGNAL_SIZE,0),tools=tools) 
p5 = figure(title="Histograma da imagem",width=800, height=300,tools=tools)
## Mudanças futuras: Max e min da imagem, localização, e quantiles
p = PreText(text=f"Valor max: {max_value:.2f}\nValor min: {min_value:.2f}\n\
10%:{quantiles[0]:.2f}\n25%: {quantiles[1]:.2f}\n50%: {quantiles[2]:.2f}\n\
75%:{quantiles[3]:.2f}\n90%: {quantiles[4]:.2f}",width=200, height=100)

## Defining figure type for each plot
p1.line(list(range(0,SIGNAL_SIZE)), np.zeros(SIGNAL_SIZE), line_width=2)
p11.line(np.zeros(SIGNAL_SIZE),list(range(0,SIGNAL_SIZE)), line_width=2)
p2.line(np.arange(len(signal_channel)), signal_channel)
p3.line(signal_channel, np.arange(len(signal_channel)))
p4.image(image=[matrix], x=0, y=0, dw=SIGNAL_SIZE, dh=SIGNAL_SIZE,palette="Turbo256")
p5.quad(top=hist_values, bottom=0, left=hist_bins[:-1], right=hist_bins[1:], fill_color="navy", line_color="navy")
quantile_vbars = [p5.vbar(x=quantiles[i], top=max(hist_values), width=0.01, color="red", legend_label=f"Quantile {q*100}%: {quantiles[i]:.2f}") for i, q in enumerate([0.1, 0.25, 0.5, 0.75, 0.9])]

## plots configs
p3.y_range.flipped = True
p3.x_range.flipped = True
p11.y_range.flipped = True
p11.x_range.flipped = True
p2.x_range = p4.x_range
p3.y_range = p4.y_range

p11.y_range = p4.y_range
p1.x_range = p4.x_range

p2.add_tools(crosshair)
p3.add_tools(crosshair)
p4.add_tools(crosshair)
p11.add_tools(crosshair)
p1.add_tools(crosshair)

p4.on_event(events.Tap, callback)
p4.on_event(RangesUpdate, generate_hist)

p4.add_tools(hover_tool)
##p4.on_event(Pan, generate_hist)
## Making a grid
grid = gridplot([[p,p2,p5], [p3, p4, p11],[None,p1]])

# Create sub-columns for related elements
column0 = column(div_form)
column1 = column(dropdown_patology, dropdown_pat, dropdown_exam,label_div2, switch2)
column2 = column(spinner_channel, spinner_start,label_div1, switch,label_div3, switch3)
column3 = column(thresh_value_up, thresh_value_down)
column4 = column(div_visualization)

# Combine the sub-columns into rows
row0 = row(column0)
row1 = row(column1, column2)
row2 = row(column3)
row3 = row(column4)

# Combine rows into a final column layout
final_layout = column(row0,row1, row2,row3, grid)

# Add the final layout to the document
curdoc().add_root(final_layout)

#curdoc().add_root(column(dropdown_patology,dropdown_pat,dropdown_exam,spinner_channel,spinner_start,label_div1,switch,label_div2,switch2,label_div3,switch3,thresh_value_up,thresh_value_down,grid))

## Activate python venv before running the app with powershell: .\venv\Scripts\Activate.ps1
##To run the app use the command: bokeh serve --show data_app.py
