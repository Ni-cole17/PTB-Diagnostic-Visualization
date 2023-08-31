import wfdb
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from wfdb import processing
from numpy.linalg import norm
import json
import pickle

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, curdoc
from bokeh.sampledata.autompg import autompg_clean as df
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models import ColumnDataSource, Paragraph, DataRange1d,RadioButtonGroup,Slider,Line,Range1d, Spinner, Select, HoverTool, RangeSlider, CrosshairTool, Span, CustomJS, TextInput
from bokeh.layouts import gridplot, column
from bokeh import events
from functools import partial
from bokeh.events import MouseWheel,RangesUpdate


## Functions
def make_matrix(signal):
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
    signal, fields = wfdb.rdsamp(f'data/ptb-diagnostic-ecg-database-1.0.0/{patient}/{hea}', channels=[*range(n_channels)])
    return signal, fields


def generate_hist(event):
    global matrix1
    x_range, y_range = p4.x_range, p4.y_range
    xmin, xmax = x_range.start, x_range.end
    ymax, ymin = y_range.start, y_range.end
    
    xmin = max(0, int(xmin))
    xmax = min(1500, int(xmax))
    ymin = max(0, int(ymin))   
    ymax = min(1500, int(ymax))
    
    # Update the histogram data based on the current image view
    rounded_signal = np.round(matrix1,3)
    rounded_signal = rounded_signal[int(ymin):int(ymax), int(xmin):int(xmax)]
    hist_values, hist_bins = np.histogram(rounded_signal,bins=6200)

    hist = p5.renderers[0] 
    hist.data_source.data = dict(top=hist_values, bottom=[0] * len(hist_values), left=hist_bins[:-1], right= hist_bins[1:])

def update_spinner_end(attrname, old, new):
    if spinner_start.value + 1500 <= spinner_start.high:
        spinner_end.value = spinner_start.value + 1500
        spinner_end.high = spinner_start.value + 1500
    else: 
        spinner_end.value = spinner_start.high


def callback(event):
    global matrix1
    if event.x and event.y:
        x_value = event.y 
        y_value = event.x  # Invert y due to plot coordinate system

        p11_line = p11.renderers[0] 
        p11_line.data_source.data = dict(x=matrix1[:, int(y_value)],y=range(0,1500))

        p1_line = p1.renderers[0] 
        p1_line.data_source.data = dict(x=range(0,1500),y=matrix1[int(x_value),:])

        #print(f"Image value: {matrix1[int(x_value), int(y_value)]}")
        #print(f"value: {int(x_value)} {x_value} {int(y_value)} {y_value}")
        

def update_patient(attrname, old, new):
    selected_patology = dropdown_patology.value
    if selected_patology != "All":  
        dropdown_pat.options = ["None"]+ list(df[df['Reason for admission'] == selected_patology]["idP"])
    else:
        dropdown_pat.options = ["None"]+ list(df["idP"])

def update_heas(attrname, old, new):
    selected_patient = dropdown_pat.value
    dropdown_exam.options = ["None"]+data[selected_patient]["heas"]

def update_signal(attrname, old, new):
    global signals_t
    global len_signal
    selected_signal = dropdown_exam.value
    if spinner_start.value == spinner_end.value:
        signals_t = np.zeros((1500,15))
    elif selected_signal != "None":
        signals_t,_ = read_signal(dropdown_pat.value,selected_signal.replace(".hea",""))
    else:
        signals_t = np.zeros((1500,15))
    len_signal = signals_t.shape[0]
    spinner_start.value = 0
    spinner_start.high = len_signal

def generate_matrix(attrname, old, new):
    global signals_t
    global matrix
    global matrix1

    signals = signals_t[spinner_start.value:spinner_end.value]
    signal_channel = signals[:, spinner_channel.value]
    source.data = dict(x=np.arange(len(signal_channel)), y=signal_channel)
    matrix1 = make_matrix(signals)
    matrix = np.flipud(matrix1)
    hover_tool.callback = CustomJS(code=hover_code, args={'matrix1': matrix1})
    if spinner_end.value - spinner_start.value == 1500 or spinner_end.value - spinner_start.value == 0: 
        p4.title.text = f"{dropdown_pat.value} | {dropdown_patology.value} | {dropdown_exam.value} | Channel: {spinner_channel.value} | Start: {spinner_start.value} | End: {spinner_end.value}"
        if spinner_end.value - spinner_start.value == 0:
            p4.image(image=[np.zeros((1500,1500))], x=0, y=1500, dw=1500, dh=1500, palette="Turbo256")
        else:
            p4.image(image=[matrix], x=0, y=1500, dw=1500, dh=1500, palette="Turbo256")

        p2_line = p2.renderers[0] 
        p2_line.data_source.data = dict(x=np.arange(len(signal_channel)), y=signal_channel)

        p3_line = p3.renderers[0] 
        p3_line.data_source.data = dict(x=signal_channel, y=np.arange(len(signal_channel)))

        switch.active = 0

def generate_matrix_thresh(attrname, old, new):
    global matrix
    global matrix1
    if switch.active == 1:
        mask = matrix < thresh_value.value
        mask = (~mask*255).astype(np.uint8)

        p4.title.text = f"{dropdown_pat.value} | {dropdown_patology.value} | {dropdown_exam.value} | Channel: {spinner_channel.value} | Start: {spinner_start.value} | End: {spinner_end.value}"
        p4.image(image=[mask], x=0, y=1500, dw=1500, dh=1500, palette="Greys256")
    else:
        p4.image(image=[matrix], x=0, y=1500, dw=1500, dh=1500, palette="Turbo256")


## MAIN

global signals_t
global matrix1

## Reading info data
df = pd.read_csv('data/Record_info.csv')
with open('data/patient_heas.json') as json_file:
    data = json.load(json_file)

## Switch
LABELS = ["UTRP", "TRP"]
switch = RadioButtonGroup(labels=LABELS, active=0)

LABELS2 = ["12d", "3d","PCA"]
switch2 = RadioButtonGroup(labels=LABELS2, active=0)

## Hover tool
hover_code = """
    const geometry = cb_data.geometry;
    const x = geometry.x;
    const y = geometry.y;
"""
hover_tool = HoverTool(tooltips=[("Value", "@image"),("(x,y)", "($x, $y)")],callback=CustomJS(args={'matrix1': matrix1}, code=hover_code))

## Reading the signal and creating recurrence plot
signals_t,_ = read_signal(list(data.keys())[0],data[list(data.keys())[0]]["heas"][0].replace(".hea",""))
signals = signals_t[0:1500]
signal_channel = signals[:, 0]
matrix1 = make_matrix(signals)
matrix = np.flipud(matrix1) 
rounded_signal = np.round(matrix,3)
hist_values, hist_bins = np.histogram(rounded_signal,bins=6200)

# Save matrix in a raw file
file_path = "matrix_data3.pkl"
output = open(file_path, 'wb')
pickle.dump(matrix1, output)
output.close()

# Creating Widgets for app
## Dropdown patology
valid_options_patology = pd.notna(df['Reason for admission'])
dropdown_patology = Select(title="Select Patology", value="All", options=["All"] + list(df.loc[valid_options_patology, 'Reason for admission'].unique()))

## Dropdown patient
dropdown_pat = Select(title="Select Patient", value=list(df['idP'])[0], options=list(df['idP']))
selected_patient = dropdown_pat.value

## Dropdown exam
dropdown_exam = Select(title="Select Exam", value=data[selected_patient]["heas"][0], options=data[selected_patient]["heas"])

## Slider
thresh_value = Slider(start=0, end=1, value=0.1, step=0.01, title="Choose threshold value for recurrence plot")

## Spinner
spinner_start = Spinner(title="Signal start:", low=0, high=signals_t.shape[0], step=1, value=0)
spinner_end = Spinner(title="Signal end:", low=0, high=signals_t.shape[0], step=1, value=1500)
spinner_channel = Spinner(title="Signal channel:", low=0, high=signals_t.shape[1]-1, step=1, value=0)
thresh_value = Spinner(title="Choose threshold value for recurrence plot",low=0, high=1, step=0.01, value = 0.3)

## Crosshair
crosshair = CrosshairTool(dimensions='both') 

# Configure callbacks
dropdown_patology.on_change('value', update_patient)
dropdown_pat.on_change('value', update_heas)

dropdown_exam.on_change('value', update_signal)
dropdown_exam.on_change('value', generate_matrix)

spinner_start.on_change('value', update_spinner_end)
spinner_start.on_change('value', generate_matrix)

spinner_end.on_change('value', generate_matrix)
spinner_channel.on_change('value', generate_matrix)

thresh_value.on_change('value', generate_matrix_thresh)

switch.on_change('active', generate_matrix_thresh)
switch2.on_change('active', generate_matrix)

# Chosing a random signal to initiate the app
## Create a Bokeh ColumnDataSource with the initial signal data
hist_source = ColumnDataSource(data=dict(top=[], bins=[]))
source = ColumnDataSource(data=dict(x=np.arange(len(signal_channel)), y=signal_channel))
y_range = DataRange1d(start=0, end=len(signal_channel)-1)

tools = ["pan","wheel_zoom","box_zoom","reset","save","help"]

## Create figures for each object
p1 = figure(title="Distances plot horizontal",width=800, height=400,tools=tools)
p11 = figure(title="Distances plot vertical",width=400, height=800,tools=tools)
p2 = figure(title="1x1500 Plot", width=800, height=300,tools=tools)
p3 = figure(title="1500x1 Plot", width=300, height=800,tools=tools)
p4 = figure(title="1500x1500 Plot", width=800, height=800, x_range=(0, 1500), y_range=(1500,0),tools=tools) 
p5 = figure(title="Histograma da imagem",width=800, height=300,tools=tools)
## Mudanças futuras: Max e min da imagem, localização, e quantiles
##p = Paragraph(text=f"""Valor visualizado""",width=200, height=100)

## Defining figure type for each plot
p1.line(range(0,1500), np.zeros(1500), line_width=2)
p11.line(np.zeros(1500),range(0,1500), line_width=2)
p2.line(np.arange(len(signal_channel)), signal_channel)
p3.line(signal_channel, np.arange(len(signal_channel)))
p4.image(image=[matrix], x=0, y=1500, dw=1500, dh=1500,palette="Turbo256")
p5.quad(top=hist_values, bottom=0, left=hist_bins[:-1], right=hist_bins[1:], fill_color="navy", line_color="navy")

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
grid = gridplot([[p, p2,p5], [p3, p4, p11],[None,p1]])

## Adding to document
curdoc().add_root(column(dropdown_patology,dropdown_pat,dropdown_exam,spinner_channel,spinner_start,spinner_end,switch,switch2,thresh_value,grid))

##To run the app use the command: bokeh serve --show data_app.py