# PTB-Diagnostic-Visualization
This repository aims to enhance the visualization of recurrence plots from the PTB Diagnostic ECG database.

# Steps to run app
### Clone this repo

    git clone https://github.com/Ni-cole17/PTB-Diagnostic-Visualization.git

### Install Requirements

    pip install -r requirements.txt

### Download/Unzip data

Before run data_app.py you need to download the data and go to data folder and unzip ptb-diagnostic-ecg-database-1.0.0.zip directly on data folder.
Link to download the data: [https://physionet.org/content/ptbdb/1.0.0/](https://physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip)
After unzip data folder will be:

data/ptb-diagnostic-ecg-database-1.0.0/patient001

data/ptb-diagnostic-ecg-database-1.0.0/patient002

data/ptb-diagnostic-ecg-database-1.0.0/...

### Run data_app.py

    bokeh serve --show data_app.py

# App visualization
Ilustration of app forms and signal visualization
![image](https://github.com/Ni-cole17/PTB-Diagnostic-Visualization/assets/65842535/f97a8eac-6201-4abd-bacd-9a8edb345af0)
![image](https://github.com/Ni-cole17/PTB-Diagnostic-Visualization/assets/65842535/2950a76b-7336-4d8e-a9e4-3209a9f62cd8)
