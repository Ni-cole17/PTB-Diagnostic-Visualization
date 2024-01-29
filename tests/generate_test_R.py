import wfdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from tqdm import tqdm
import math
from wfdb import processing
from numpy.linalg import norm
import cv2
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import pandas as pd
from scipy import signal
import json
import os

def float_to_unint(flot_img):
    '''
    Função para converter uma imagem de float para uint8.
    :param flot_img: imagem em float.
    :return uint8_image: imagem em uint8.
    '''
    # Clip values to the range [0, 1]
    clipped_image = np.clip(flot_img, 0, 1)

    # Scale to the range [0, 255] and convert to uint8
    uint8_image = (clipped_image * 255).astype(np.uint8)
    return uint8_image

def generate_image(signal):
  '''
  Função para gerar a matriz de distância a partir de um sinal utilizando a distância euclidiana.
  :param signal: sinal a ser utilizado.
  '''
  diffs = signal[:, np.newaxis, :] - signal[np.newaxis, :, :]
  dists = norm(diffs, axis=2)
  l = np.concatenate((l, dists), axis=0) if 'l' in locals() else dists
  return l

def show_matrix(matrix,map='jet',show=True,return_img=False):
    '''
    Função para plotar uma matriz de valores.
    :param matrix: matriz de valores.
    :param map: cmap do matplotlib.
    :param show: se True, plota a imagem.
    :param return_img: se True, retorna a imagem.
    '''
    # Obtain the colormap
    colormap = plt.cm.get_cmap(map)
    # Normalize the intensity values
    norm = plt.Normalize(vmin=np.min(matrix), vmax=np.max(matrix))
    # Map intensity values to RGBA using the colormap
    rgb_image = colormap(norm(np.abs(matrix)))[:, :, :3]  # Discard the alpha channel
    if show:
        fig, ax = plt.subplots()
        ax.imshow(np.abs(matrix), cmap=map)
        plt.show()
    if return_img:
        return rgb_image

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

def testing_R_R(matrix_dist,sig,outpath,save = True,channel = 0):
    '''
    Função para testar o algoritmo de detecção de picos R-R, ao final ela salva os resultados em um arquivo .json na mesma pasta das imagens.
    :param matrix_dist: matriz de distância.
    :param sig: sinal a ser utilizado.
    :param outpath: caminho para salvar a imagem.
    :param save: se True, salva a imagem.
    :param channel: canal do sinal a ser utilizado no plot da imagem.
    '''
    #quantiles = np.quantile(matrix_dist, [0.25, 0.5, 0.95])
    ## Applying a threshold to the matrix on 0.95 quantile
    #matrix_dist_t = (matrix_dist < quantiles[2])*255
    #matrix_dist_bin = 255 - matrix_dist_t

    # Reshape the image to a 2D array of pixels (rows, columns, channels)
    image = float_to_unint(show_matrix(matrix_dist,return_img=True,show=False))
    pixels = image.reshape((-1, 3))  # Assuming it's a 3-channel (RGB) image
    # Choose the number of clusters (k)
    k = 5
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixels)
    # Reshape the labels to the original image shape
    labels_reshaped = labels.reshape(image.shape[:2])

    Back_mask,QRS_mask,res_masks,orig_masks = get_back_qrs_masks(labels_reshaped,image,k)
    matrix_dist_bin = cv2.cvtColor(float_to_unint(show_matrix(QRS_mask,return_img=True,show=False)), cv2.COLOR_RGB2GRAY)
    sums = []
    for i in range(matrix_dist_bin.shape[0]):
        sums.append(sum(matrix_dist_bin[:, i]))

    peaks = []
    #limiar = np.percentile(sums, 98)
    limiar = max(sums)*0.75
    for i in range(matrix_dist_bin.shape[0]):
        if sum(matrix_dist_bin[:, i]) > limiar:
            peaks.append(i)
            
    # Data vector
    data = np.array(peaks)

    # Reshape the vector to be a two-dimensional matrix
    data_reshaped = data.reshape(-1, 1)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=100, min_samples=2)
    labels = dbscan.fit_predict(data_reshaped)

    # Find the mean for each cluster
    unique_labels = np.unique(labels)
    centroids = []
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = data[labels == label]
        cluster_mean = np.mean(cluster_points)
        centroids.append(cluster_mean)

    if save:
        ## Plot matrix_dist, bin_image, signal with line on centroids and sums
        fig, axs = plt.subplots(2, 2, figsize=(20,20))
        axs[0, 0].imshow(matrix_dist, cmap='jet')
        axs[0, 0].set_title('Matriz de distância')
        axs[0, 1].imshow(matrix_dist_bin, cmap='gray')
        axs[0, 1].set_title('Matriz binária')

        axs[1, 0].plot(sums)
        axs[1, 0].set_title('Soma das colunas')
        axs[1, 0].axhline(y=limiar, color='r', linestyle='--')
        ## Plot centroids with a line on signal in axs[1,1]
        t = np.linspace(0, len(sig), len(sig))
        axs[1,1].plot(t, sig[:,channel], label='Sinal original')
        for i in centroids:
            axs[1,1].axvline(x=i, color='r', linestyle='--')
        axs[1, 1].set_title('Canal 1 do sinal, com linhas nos picos R-R')
        axs[1,1].set_xlabel('Tempo (1000*s)')
        axs[1,1].set_ylabel('Amplitude')
        ## Save the figure
        plt.savefig(outpath)
        #close the figure
        plt.close()
    else:
        return centroids

def get_back_qrs_masks(labels_reshaped,image,k):
    '''
    Função para obter as máscaras de QRS e background.
    :param labels_reshaped: matriz de labels.
    :param image: imagem.
    :param k: número de clusters.
    :return masks: Background mask,QRS mask,remaining masks,original masks
    '''
    masks_img = []
    masks = []
    for i in range(k):
        label_choose = i
        mask = (labels_reshaped == label_choose)
        mask_img = show_matrix(mask*255,map = 'gray',show=False,return_img=True)
        masks.append(mask_img)

        masked_image = image.copy()
        masked_image[~mask] = 0
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        black_mask = gray_image > 0

        # Calcular a média da intensidade de cor para cada canal
        red_intensity = np.mean(masked_image[:, :, 0])  # Canal Vermelho
        green_intensity = np.mean(masked_image[:, :, 1])  # Canal Verde
        blue_intensity = np.mean(masked_image[:, :, 2])  # Canal Azul

        brightness = np.mean(gray_image[black_mask])
        max_intensity_channel = np.argmax([red_intensity, green_intensity, blue_intensity])

        masks_img.append([mask,masked_image,red_intensity,green_intensity,blue_intensity,max_intensity_channel,brightness])

    masks_img = np.array(masks_img, dtype=object)
    try:
        reds_array = masks_img[:,5]
        reds_inds = np.where(reds_array==0)[0]
        reds = masks_img[:,2][reds_inds]
        max_v = max(reds)
        ## Index of max and min values
        max_idx = reds_inds[np.where(reds == max_v)[0][0]]
    except:
        reds_array = masks_img[:,2]
        max_v = max(reds_array)
        ## Index of max and min values
        max_idx = np.where(reds_array == max_v)[0][0]

    last = masks_img[:,6]
    min_v = min(last)
    min_idx = np.where(last == min_v)[0][0]

    all_masks = masks_img[:,1]
    all_masks = np.array(all_masks)
    res_masks = np.delete(all_masks, [min_idx,max_idx],axis=0)
    orig_masks = np.delete(masks, [min_idx,max_idx],axis=0)

    return masks_img[min_idx][0],masks_img[max_idx][0], res_masks, orig_masks

def comparingRR(matrix_dist,sig,outpath):
    '''
    Função para comparar os picos R-R obtidos pelo algoritmo com os picos R-R obtidos pelo WFDB.
    :param matrix_dist: matriz de distância.
    :param sig: sinal a ser utilizado.
    :param outpath: caminho para salvar a imagem.
    '''
    ## Calculating R-R interval and heart rate
    rr = []
    hr = []
    fs = 1000
    ## WFDB qrs detection
    for channel in range(0,12):
      qrs_inds = processing.xqrs_detect(sig=sig[:,channel], fs=fs,verbose=False)
      if len(qrs_inds) != 0:
        break
    centroids = testing_R_R(matrix_dist,sig,outpath,save = False)
    for i in range(len(centroids)-1):
        diff = (centroids[i+1] - centroids[i])/fs
        rr.append(diff)
        hr.append(60/diff)
    ## Calculating R-R interval and heart rate
    rr_qrs = []
    hr_qrs = []
    for i in range(len(qrs_inds)-1):
        diff = (qrs_inds[i+1] - qrs_inds[i])/fs
        rr_qrs.append(diff)
        hr_qrs.append(60/diff)
    ## Plot centorids and qrs_inds on signal
    t = np.linspace(0, len(sig), len(sig))
    fig, axs = plt.subplots(1,2,figsize=(20,10))
    axs[0].plot(t, sig[:,channel], label='Sinal original')
    for i in centroids:
        axs[0].axvline(x=i, color='r', linestyle='--', label='Centroids',linewidth=2)
    for j in qrs_inds:
        axs[0].axvline(x=j, color='g', linestyle='--', label='QRS',linewidth=1)
    axs[0].set_title(f'Canal {channel} do sinal, com linhas nos picos R-R')
    axs[0].set_xlabel('Tempo (1000*s)')
    axs[0].set_ylabel('Amplitude')
    ## Set a legend for red and green lines
    red_patch = plt.Line2D([0], [0], color='r', linestyle='--', label='Centroids')
    green_patch = plt.Line2D([0], [0], color='g', linestyle='--', label='QRS')
    axs[0].legend(handles=[red_patch,green_patch])

    axs[1].plot(t, sig[:,0], label='Sinal original')
    for k in centroids:
        axs[1].axvline(x=k, color='r', linestyle='--', label='Centroids',linewidth=2)
    for m in qrs_inds:
        axs[1].axvline(x=m, color='g', linestyle='--', label='QRS',linewidth=1)
    axs[1].set_title('Canal 0 do sinal, com linhas nos picos R-R')
    axs[1].set_xlabel('Tempo (1000*s)')
    axs[1].set_ylabel('Amplitude')
    ## Set a legend for red and green lines
    red_patch = plt.Line2D([0], [0], color='r', linestyle='--', label='Centroids')
    green_patch = plt.Line2D([0], [0], color='g', linestyle='--', label='QRS')
    axs[1].legend(handles=[red_patch,green_patch])
    ## Save the figure
    plt.tight_layout()
    plt.savefig(outpath)
    ## Save json with results
    plt.close('all')
    return rr,hr,rr_qrs,hr_qrs, qrs_inds, centroids

def comparingRRv2(sig,qrs_inds,centroids,outpath=''):
    '''
    Função versão 2 para comparar os picos R-R obtidos pelo algoritmo com os picos R-R obtidos pelo WFDB.
    :param sig: sinal a ser utilizado.
    :param qrs_inds: picos R-R obtidos pelo WFDB.
    :param centroids: picos R-R obtidos pelo algoritmo.
    :param outpath: caminho para salvar a imagem.
    '''
    ## Calculating R-R interval and heart rate
    rr = []
    hr = []
    fs = 1000
    json_filename = "comparing.json"
    for i in range(len(centroids)-1):
        diff = (centroids[i+1] - centroids[i])/fs
        rr.append(diff)
        hr.append(60/diff)
    ## Calculating R-R interval and heart rate
    rr_qrs = []
    hr_qrs = []
    for i in range(len(qrs_inds)-1):
        diff = (qrs_inds[i+1] - qrs_inds[i])/fs
        rr_qrs.append(diff)
        hr_qrs.append(60/diff)

    ## Read json file
    output_path_root = os.path.dirname(outpath)
    if os.path.exists(os.path.join(output_path_root, json_filename)):
        with open(os.path.join(output_path_root, json_filename)) as json_file:
            data = json.load(json_file)
    else:
        data = {"TP":0,"FP":0,"FN":0,"TN":0}

    matrix = [[0,0],[0,0]]
    if len(centroids) == 0 and len(qrs_inds) == 0:
        pass
    elif len(qrs_inds) > 0 and len(centroids) > 0:
        for i in qrs_inds:
            tp = False
            for j in centroids:
                if i + 60 >= j and i - 60 <= j:
                    ## True positive
                    matrix[0][0] += 1
                    tp = True
                    break
            if not tp:
                ## False negative
                matrix[1][0] += 1
        ## False positive
        matrix[0][1] = len(centroids) - matrix[0][0]
    elif len(qrs_inds) == 0 and len(centroids) > 0:
        matrix[0][1] = len(centroids)
    elif len(centroids) == 0 and len(qrs_inds) > 0:
        matrix[1][0] = len(qrs_inds)

    t = np.linspace(0, len(sig), len(sig))
    fig, axs = plt.subplots(1,1,figsize=(20,10))
    axs.plot(t, sig[:,0], label='Sinal original')
    for i in centroids:
        axs.axvline(x=i, color='r', linestyle='--', label='Centroids',linewidth=2)
    for j in qrs_inds:
        axs.axvline(x=j, color='g', linestyle='--', label='QRS',linewidth=1)
        axs.fill_between(t, np.min(sig[:,0]), np.max(sig[:,0]), where=((t >= j-60) & (t <= j+60)), color='gray', alpha=0.3)
    ## Make a annotation on the figure with the matrix
    axs.annotate(f'Matrix: [{matrix[0][0],matrix[0][1]}]\n            [{matrix[1][0],matrix[1][1]}] ', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=16,
                horizontalalignment='left', verticalalignment='top')
    axs.set_title(f'Canal {0+1} do sinal, com linhas nos picos R-R')
    axs.set_xlabel('Tempo (1000*s)')
    axs.set_ylabel('Amplitude')
    ## Set a legend for red and green lines
    red_patch = plt.Line2D([0], [0], color='r', linestyle='--', label='Centroids')
    green_patch = plt.Line2D([0], [0], color='g', linestyle='--', label='QRS')
    axs.legend(handles=[red_patch,green_patch])

    ## Save the figure
    plt.tight_layout()
    plt.savefig(outpath)
    ## Save json with results
    plt.close('all')

    ## Create a json
    dict = {"TP":data['TP']+matrix[0][0],"FP":data['FP']+matrix[0][1],"FN":data['FN']+matrix[1][0],"TN":data['TN']+matrix[1][1]}
    with open(os.path.join(output_path_root, json_filename), 'w') as fp:
        json.dump(dict, fp)
        
if __name__ == '__main__':
    ## Read csv
    df_patients = pd.read_csv('patients_sample.csv')
    dic_pat = {}
    count = 0
    ## Iter belong the csv
    for index, row in df_patients.iterrows():
        print(f'Patient {count}')
        count += 1
        patient = row['idP']
        issue = row['Reason for admission']
        record = row['hea'].replace(".hea","")
        ## Read the signal
        sign, fields = wfdb.rdsamp(f'ptb-diagnostic-ecg-database-1.0.0/{patient}/{record}', channels=[*range(15)])
        ## Read the annotation
        signal_size = fields['sig_len']
        ## Get points from start columns from dataframe, there is ten columns
        points = row[3:].dropna().values
        for point in points:
            out = f'tests/T_wave/T_wave_v4/{patient}-{issue}-{record}-{point}-{point+2000}'
            out_path = f'{out}.png'
            if os.path.exists(out_path):
                continue
            # Extract the first 2000 samples from the first 12 channels
            sig_f = filt_bandpass(sign,1000)
            sig = sig_f[:, :12][point:point+2000]
            matrix_dist = generate_image(sig)
            matrix_img = show_matrix(matrix_dist,return_img=True,show=False)
            image = float_to_unint(matrix_img)
            centroids = testing_R_R(matrix_dist,sig,out_path,save = False)
                        
            min_len = 0
            for channel in range(0,12):
                qrs_inds_i = processing.xqrs_detect(sig=sig_f[:,channel], fs=1000,verbose=False)
                if len(qrs_inds_i) >= min_len:
                    min_len = len(qrs_inds_i)
                    qrs_inds = qrs_inds_i

            qrs_inds = np.array(qrs_inds)
            #print("PONTOS:")
            if centroids[-1] < 1940: ## Max value 2000 minus 60 from QRS duration
                if centroids[0] > 60:
                    qrs_trues = qrs_inds[(qrs_inds >= point) & (qrs_inds <= point+2000)]
                    sig_r = sig_f[:, :12][point:point+2000]
                    x = 0
                    #print(point,point+2000)
                else:
                    qrs_trues = qrs_inds[(qrs_inds >= point-60) & (qrs_inds <= point+2000)]
                    sig_r = sig_f[:, :12][point-60:point+2000]
                    x = +60
                    #print(point-60,point+2000)
            else:
                if centroids[0] > 60:
                    qrs_trues = qrs_inds[(qrs_inds >= point) & (qrs_inds <= point+2060)]
                    sig_r = sig_f[:, :12][point:point+2060]
                    x = 0
                    #print(point,point+2060)
                else:
                    qrs_trues = qrs_inds[(qrs_inds >= point-60) & (qrs_inds <= point+2060)]
                    sig_r = sig_f[:, :12][point-60:point+2060]
                    x = +60
                    #print(point-60,point+2060)
            ## Get just values between point and point+2000
            #print(point,point+2000)
            #print(point,x)
            qrs_trues = [k-(point - x) for k in qrs_trues]
            centroids = [k+x for k in centroids]
            comparingRRv2(sig_r,qrs_trues,centroids,out_path)


