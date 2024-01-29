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

def dilate_imag(img,iterations=10,kernel=(5,5),show = True):
    '''
    Função para aplicar a dilatação em uma imagem.
    :param img: imagem a ser dilatada.
    :param iterations: número de iterações.
    :param kernel: tamanho do kernel.
    :param show: se True, plota a imagem.
    :return dilation_result: imagem dilatada.
    '''
    kernel = np.ones(kernel, np.uint8)
    dilation_result = cv2.dilate(img, kernel, iterations=iterations)

    if show:
        # Exibir as imagens original e resultante
        plt.subplot(121), plt.imshow(img,cmap='gray')
        plt.subplot(122), plt.imshow(dilation_result, cmap='gray'), plt.title('Com Dilatação')
        plt.show()
    else:
        return dilation_result
    
def cluster_image_kmeans(matrix, n_clusters=4, return_img=False):
    '''
    Função para aplicar o algoritmo de clusterização K-means em uma imagem.
    :param matrix: matriz de valores.
    :param n_clusters: número de clusters.
    :param return_img: se True, retorna a imagem.
    :return labels_reshaped: imagem com os clusters.
    '''
    # Reshape the image to a 2D array of pixels (rows, columns, channels)
    image = float_to_unint(show_matrix(matrix,return_img=True,show=False))
    pixels = image.reshape((-1, 3))  # Assuming it's a 3-channel (RGB) image

    # Choose the number of clusters (k)
    k = n_clusters

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(pixels)

    # Reshape the labels to the original image shape
    labels_reshaped = labels.reshape(image.shape[:2])

    if return_img:
        return labels_reshaped, image
    else:
        return labels_reshaped

## Apply erosion on v image
def erode_img(img,iterations = 10,kernel=(5,5),show=True):
    '''
    Função para aplicar a erosão em uma imagem.
    :param img: imagem a ser erodida.
    :param iterations: número de iterações.
    :param kernel: tamanho do kernel.
    :param show: se True, plota a imagem.
    :return erosion_result: imagem erodida.
    '''
    kernel = np.ones(kernel, np.uint8)
    erosion_result = cv2.erode(img, kernel, iterations=iterations)

    if show:
        # Exibir as imagens original e resultante
        plt.subplot(121), plt.imshow(img,cmap='gray')
        plt.subplot(122), plt.imshow(erosion_result, cmap='gray'), plt.title('Com Erosão')
        plt.show()
    else:
        return erosion_result

def opening_img(img,iterations = 10,kernel=(5,5),show=True):
    '''
    Função para aplicar a abertura em uma imagem.
    :param img: imagem a ser aberta.
    :param iterations: número de iterações.
    :param kernel: tamanho do kernel.
    :param show: se True, plota a imagem.
    :return opening_result: imagem aberta.
    '''
    kernel = np.ones(kernel, np.uint8)
    opening_result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

    if show:
        # Exibir as imagens original e resultante
        plt.subplot(121), plt.imshow(img,cmap='gray')
        plt.subplot(122), plt.imshow(opening_result, cmap='gray'), plt.title('Com Opening')
        plt.show()
    else:
        return opening_result
    
def closing_img(img,iterations = 10,kernel=(5,5),show=True):
    '''
    Função para aplicar o fechamento em uma imagem.
    :param img: imagem a ser fechada.
    :param iterations: número de iterações.
    :param kernel: tamanho do kernel.
    :param show: se True, plota a imagem.
    :return closing_result: imagem fechada.
    '''
    kernel = np.ones(kernel, np.uint8)
    closing_result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    if show:
        # Exibir as imagens original e resultante
        plt.subplot(121), plt.imshow(img,cmap='gray')
        plt.subplot(122), plt.imshow(closing_result, cmap='gray'), plt.title('Com Closing')
        plt.show()
    else:
        return closing_result

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

def get_back_qrs_masks(labels_reshaped,image,k):
    '''
    Função para obter as máscaras de background e QRS.
    :param labels_reshaped: imagem com os clusters.
    :param image: imagem original.
    :param k: número de clusters.
    :return Back_mask: máscara de background.
    :return QRS_mask: máscara de QRS.
    :return res_masks: máscaras de QRS.
    :return orig_masks: máscaras de QRS.
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

def normalize_array(vec):
    '''
    Função para normalizar um vetor.
    :param vec: vetor a ser normalizado.
    :return vec_new: vetor normalizado.
    '''
    vec_new = [(i - min(vec))/(max(vec) - min(vec)) for i in vec]
    return vec_new

def apply_erode_dilate(orig_masks,iterations=7,kernel=(5,5)):
    '''
    Função para aplicar erosão e dilatação nas máscaras de QRS.
    :param orig_masks: máscaras de QRS.
    :param iterations: número de iterações.
    :param kernel: tamanho do kernel.
    :return masks_bests_proc: máscaras de QRS processadas.
    '''
    masks_bests_proc = []
    while len(masks_bests_proc) == 0:
        for m in orig_masks:    
            t_wave_erode = erode_img(m,iterations=iterations,kernel=kernel,show=False)
            t_wave_dilate = dilate_imag(t_wave_erode,iterations=iterations,kernel=kernel,show=False)
            if np.sum(t_wave_dilate) > 0:
                    masks_bests_proc.append(t_wave_dilate)
        iterations -= 2
    return masks_bests_proc
        
def get_T_mask(res_masks,orig_masks):
    '''
    Função para obter a máscara da onda T.
    :param res_masks: máscaras de QRS.
    :param orig_masks: máscaras de QRS.
    :return masks_bests_proc[best_med]: máscara da onda T.
    '''
    if len(res_masks) == 1:
        return res_masks[0]
    elif len(res_masks) > 1:
        metric = []
        means = []
        for r in res_masks:
            gray_image = cv2.cvtColor(r, cv2.COLOR_RGB2GRAY)
            black_mask = gray_image > 0
            ## Apply black mask on matrix_dist
            matrix_mask = matrix_dist[black_mask]
            metric.append(np.sum(matrix_mask))
            brightness = np.mean(gray_image[black_mask])
            means.append(brightness)

        ## Order masks mean
        means = np.array(normalize_array(means))
        metric = np.array(normalize_array(metric))
        metric_ord_inds = np.argsort(metric)[::-1]
        metric_ord = np.sort(metric)[::-1]
        #best_metric = metric_ord[:2]
        best_inds = metric_ord_inds[:2]
        #masks_2_best = res_masks[best_inds]
        orig_masks = np.array(orig_masks)[best_inds]
        ## Process masks (erode and dilate)
        masks_bests_proc = apply_erode_dilate(orig_masks)
        if len(masks_bests_proc) == 1:
            return masks_bests_proc[0]
        else:
            best_metric = metric[best_inds]
            best_means = means[best_inds]
            meds = 0.8*best_means + 0.2*best_metric
            best_med = np.argmax(meds)
            masks_bests_proc = np.array(masks_bests_proc)

    return masks_bests_proc[best_med]

def get_T_peaks(T_mask):
    '''
    Função para obter os picos da onda T.
    :param T_mask: máscara da onda T.
    :return centroids: picos da onda T.
    '''
    ## Sums of masks_bests_proc[0]
    sums = []
    centroids = []
    ## Turn to one channel
    T_mask_gray = cv2.cvtColor(float_to_unint(T_mask), cv2.COLOR_RGB2GRAY)
    for i in range(T_mask_gray.shape[0]):
        sums.append(sum(T_mask_gray[i]))

    ## Identifying peaks from the sum of the matrix columns
    peaks = []
    first_last = {}
    limiar = max(sums)*0.85
    for i in range(T_mask_gray.shape[0]):
        if sum(T_mask_gray[:, i]) > limiar:
            peaks.append(i)
            
    # Data vector
    data = np.array(peaks)

    # Reshape the vector to be a two-dimensional matrix
    data_reshaped = data.reshape(-1, 1)

    # Apply DBSCAN
    try:
        dbscan = DBSCAN(eps=100, min_samples=2)
        labels = dbscan.fit_predict(data_reshaped)

        # Find the mean for each cluster
        unique_labels = np.unique(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            cluster_points = data[labels == label]
            cluster_mean = np.mean(cluster_points)
            centroids.append(cluster_mean)
            first_last[label] = [min(cluster_points), max(cluster_points)]

        return centroids
    except:
        return []

def testing_cluster(matrix_dist,sig,outpath):
    '''
    Função para testar o algoritmo de clusterização.
    :param matrix_dist: matriz de distância.
    :param sig: sinal a ser utilizado.
    :param outpath: caminho para salvar a imagem.
    '''
    # Ger labels from kmeans
    labels_reshaped = cluster_image_kmeans(matrix_dist, n_clusters=5)

    # Assign unique colors to each cluster label
    unique_labels = np.unique(labels_reshaped)
    num_clusters = len(unique_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))[:, :3]  # Use viridis colormap

    # Create a colored image based on cluster labels
    colored_image = colors[labels_reshaped]

    Back_mask,QRS_mask,res_masks,orig_masks = get_back_qrs_masks(labels_reshaped,image,k)
    T_mask = get_T_mask(res_masks,orig_masks)
    centroids = get_T_peaks(T_mask)

    # Display the original and clustered images
    fig, axs = plt.subplots(2, 3, figsize=(30,30))

    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original Image')

    axs[0, 1].imshow(colored_image)
    axs[0, 1].set_title('Clustered Image')

    t = np.linspace(0, len(sig), len(sig))
    axs[0, 2].plot(t, sig[:,0], label='Sinal original')
    for i in centroids:
        axs[0, 2].axvline(x=i, color='r', linestyle='--')
    axs[0, 2].set_title('Sinal com onda T identificada')
    axs[0, 2].set_xlabel('Tempo (s)')
    axs[0, 2].set_ylabel('Amplitude')

    axs[1, 0].imshow(QRS_mask, cmap='gray')
    axs[1, 0].title.set_text('Mask QRS')
    axs[1, 1].imshow(Back_mask, cmap='gray')
    axs[1, 1].title.set_text('Mask background')

    axs[1, 2].imshow(T_mask, cmap='gray')
    axs[1, 2].title.set_text('Mask T wave')

    '''ax1.scatter(range(len(data)), data, c=labels, cmap='viridis', s=50, alpha=0.8)
    ax1.scatter(idxs, [data[i] for i in idxs], marker='X', color='red', s=100)
    ax1.set_title('DBSCAN Clustering with Centroids')
    ax1.set_xlabel('Index in the Vector')
    ax1.set_ylabel('Values')'''

    ## Save the figure
    plt.savefig(outpath)
    #plt.show()
    plt.close()

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
            testing_cluster(matrix_dist,sig,out_path)


