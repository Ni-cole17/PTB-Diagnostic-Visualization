# PTB-Diagnostic-Visualization
This repository aims to enhance the visualization of recurrence plots from the PTB Diagnostic ECG database.

# Steps to run app
### Clone this repo

    git clone https://github.com/Ni-cole17/PTB-Diagnostic-Visualization.git

### Install Requirements

    pip install -r requirements.txt

### Download/Unzip data

Before run data_app.py you need to download the data, go to data folder and unzip ptb-diagnostic-ecg-database-1.0.0.zip directly on data folder.
Link to download the data: [https://physionet.org/content/ptbdb/1.0.0/](https://physionet.org/static/published-projects/ptbdb/ptb-diagnostic-ecg-database-1.0.0.zip)
After unzip data folder will be:

data/ptb-diagnostic-ecg-database-1.0.0/patient001

data/ptb-diagnostic-ecg-database-1.0.0/patient002

data/ptb-diagnostic-ecg-database-1.0.0/...

### Run data_app.py

    bokeh serve --show data_app.py

# App visualization
Ilustration of app forms and signal visualization
![image](https://github.com/Ni-cole17/PTB-Diagnostic-Visualization/assets/65842535/93d86651-465f-4896-9deb-9dcb4f5e8667)
![image](https://github.com/Ni-cole17/PTB-Diagnostic-Visualization/assets/65842535/0928eba5-fc11-43d9-bdb0-b4ba0efde6bf)
![image](https://github.com/Ni-cole17/PTB-Diagnostic-Visualization/assets/65842535/0ce97fce-a84a-452c-ab05-6126316f8f8c)
![image](https://github.com/Ni-cole17/PTB-Diagnostic-Visualization/assets/65842535/cc00dbf2-e9d9-4250-9044-a281eaf854be)

In this Bokeh application, you can visualize various plots and analyses for 12 signal derivations, VCG (Vectorcardiogram) with 3 derivations, and results from PCA (Principal Component Analysis). The application offers the following features:

1. Signal Plot: View plots for the 12 signal derivations, VCG, and PCA analysis, available for each patient in the PTB Diagnostic ECG Database.
    
2. Recurrence Plot: Explore both unthresholded and thresholded recurrence plots.
    
3. Interactive Tap: When you tap on an image plot, you can inspect distances both vertically and horizontally from the tapped point.
    
4. Histogram Plot: In the top left corner, there's a histogram plot that dynamically updates as you change the image. It provides insights such as maximum and minimum values, as well as quantiles (0.1, 0.25, 0.5, 0.75, 0.9).

This interactive Bokeh application offers a comprehensive way to analyze and visualize your data, enabling a deeper understanding of your signal and its characteristics.

### Limitations
Since plotting longer signals would require a significant amount of RAM and take a considerable amount of time to load, for now, this code is limited to a signal time interval of 1.5 seconds. However, there may be modifications in the future to accommodate signals shorter than 1.5 seconds.
