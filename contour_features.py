import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing

def load_image(img_name):
    img = cv.imread(img_name)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def process_image(img, clahe_tile=(19,19), kernel_size=(37,37)):
    # Preprocess
    #smoothed = cv.blur(img,(5,5))
    denoise = cv.fastNlMeansDenoising(img,None,21,7)
    # Convert to gray
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # CLAHE
    clahe_filter = cv.createCLAHE(clipLimit=3,tileGridSize=clahe_tile)
    clahe = clahe_filter.apply(img_gray)
    #clahe = cv.equalizeHist(img_gray)
    # Otsu
    _, otsu = cv.threshold(clahe,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    # Closing (Dilate+Erode)
    kernel = np.ones(kernel_size,np.uint8)
    closing = cv.morphologyEx(otsu, cv.MORPH_OPEN, kernel)
    # Fill holes    
    holes = closing.copy()
    contours,_ = cv.findContours(holes,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    limit = img.shape[0]*img.shape[1]*0.05
    for contour in contours:
        area = cv.contourArea(contour)
        if area<limit:
            cv.drawContours(holes,[contour],0,0, thickness=cv.FILLED)
    return [img,denoise,clahe,otsu,closing,holes]


def percentage_white_pixels(image, height):
    img_cut1 = image[0:height+1,:]
    total_pixels_section = img_cut1.shape[0]*img_cut1.shape[1]
    pixels1 = np.count_nonzero(img_cut1 == 255)
    pwp1 = (pixels1*100)/total_pixels_section
    
    img_cut2 = image[height:image.shape[0],:]
    total_pixels_section = img_cut2.shape[0]*img_cut2.shape[1]
    pixels2= np.count_nonzero(img_cut2 == 255)
    pwp2 = (pixels2*100)/total_pixels_section
    return [pwp1, pwp2, pixels1, pixels2]

def cut_section(image, return_process=False, calculate_features=False):
    contours,_ = cv.findContours(image,cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)
    img = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
    biggest_contour = []
    biggest_area = 0
    final_data = {}
    for contour in contours:
        area = cv.contourArea(contour)
        if area>biggest_area:
            biggest_contour = contour
            biggest_area = area
    if biggest_area>0:        
        rows,cols = img.shape[:2]
        
        min_col0 = rows
        max_col0 = 0
        min_cols = rows
        max_cols = 0
        contour = []
        for point in biggest_contour:
            if point[0][0]==0:
                if point[0][1]> max_col0:
                    max_col0 = point[0][1]
                if point[0][1]< min_col0:
                    min_col0 = point[0][1]
            if point[0][0]==(cols-1):
                if point[0][1]> max_cols:
                    max_cols = point[0][1]
                if point[0][1]< min_cols:
                    min_cols = point[0][1]
            contour.append([point[0][0], point[0][1]])
        final_data['biggest_contour'] = contour
        
        diff_ymax = abs(rows-max_col0)
        if diff_ymax>min_col0:
            y_value = max_col0
        else:
            y_value = min_col0
        if y_value<image.shape[0] and y_value>0:
            pwp1, pwp2, _, _ = percentage_white_pixels(image, int(y_value))        
            if pwp1>pwp2:
                final_data['sign'] = 1
                begin_point = [0,max_col0]
                end_point = [cols-1,max_cols]
            else: 
                final_data['sign'] = -1
                begin_point = [0,min_col0]
                end_point = [cols-1,min_cols]
            if begin_point in contour and end_point in contour:
                index_begin = contour.index(begin_point)
                index_end = contour.index(end_point)
                contour_line = contour[index_begin:index_end+1]
                if contour_line[1][0]==0:
                    contour_line = contour[index_end:] + contour[:index_begin+1]
                final_data['contour_line'] = contour_line

                min_line = begin_point[1]
                max_line = 0
                for point in contour_line:
                    if point[1]<min_line:
                        min_line = point[1]
                    if point[1]>max_line:
                        max_line = point[1]

                final_data['y1'] = min_line
                final_data['y2'] = max_line
                final_data['dist'] = max_line - min_line

                if return_process:
                    cv.rectangle(img,(0,min_line),(cols-1,max_line),(255,0,0),2)
        if return_process:
            if not calculate_features:
                cv.drawContours(img, [biggest_contour], 0, (0,255,0), 2)
            final_data['img'] = img            
    return final_data

def calculate_features(image, final_data, return_process=False):
    
    line_point = np.array(final_data['contour_line'] , dtype=np.int32)
    X = np.reshape(line_point[:,0], (len(line_point), -1))
    y = line_point[:,1]
    reg = LinearRegression().fit(X, y)
    final_data['slope'] = reg.coef_[0]

    middle = int((final_data['y1']+final_data['y2'])/2)
    validation = True
    if final_data['sign'] == 1:
        if (final_data['y2']+1)<image.shape[0]:
            _, _, pixels_limit ,  _= percentage_white_pixels(image, int(final_data['y2']+1))
            _, _, pixels_middle ,  _= percentage_white_pixels(image, int(middle+1))
        else:
            validation = False
    else:
        if final_data['y1']>0:
            _, _, _,  pixels_limit= percentage_white_pixels(image, int(final_data['y1']-1))
            _, _, _,  pixels_middle= percentage_white_pixels(image, int(middle-1))
        else:
            validation = False
    if validation:
        total_pixels = image.shape[0]*image.shape[1]
        final_data['pwp_limit']  =  pixels_limit/total_pixels
        final_data['pwp_middle']  = pixels_middle/total_pixels

        if final_data['sign'] == 1:
            beginning_point = [0,final_data['y1']]
            ending_point = [image.shape[1]-1,final_data['y1']]
        else:
            beginning_point = [0,final_data['y2']]
            ending_point = [image.shape[1]-1,final_data['y2']]

        if final_data['contour_line'][0][0] == 0:
            complete_contour = np.array([beginning_point] + final_data['contour_line'] + [ending_point], dtype=np.int32)
        else:
            complete_contour = np.array([ending_point] + final_data['contour_line'] + [beginning_point], dtype=np.int32)

        area = cv.contourArea(complete_contour)
        x,y,w,h = cv.boundingRect(complete_contour)
        rect_area = w*h
        extent = float(area)/rect_area
        final_data['extent'] = extent       

        if return_process:
            cv.drawContours(final_data['img'], [complete_contour], 0, (0,255,0), 2)
            cv.rectangle(final_data['img'],(x,y),(x+w,y+h),(255,0,0),2)
            
def cut_image(image, y1, y2):
    y1 = y1-50
    y1 = 0 if y1<0 else y1
    y2 = y2+50
    y2 = (image.shape[0]-1) if y2>(image.shape[0]-1) else y2
    return image[y1:y2,:]

def show_process(imgs):
    fig, axs = plt.subplots(1, 7, figsize=(16,5)) 
    axs[0].imshow(imgs[0])
    axs[0].axis('off')
    axs[0].set_title('(a) Original')
    axs[1].imshow(imgs[1])
    axs[1].axis('off')
    axs[1].set_title('(b) Preprocess')
    axs[2].imshow(imgs[2], cmap='gray')
    axs[2].axis('off')
    axs[2].set_title('(c) Gray and CLAHE')
    axs[3].imshow(imgs[3], cmap='gray')
    axs[3].axis('off')
    axs[3].set_title('(d) Otsu')
    axs[4].imshow(imgs[4], cmap='gray')
    axs[4].axis('off')
    axs[4].set_title('(e) Closing')
    axs[5].imshow(imgs[5], cmap='gray')
    axs[5].axis('off')
    axs[5].set_title('(f) Fill holes')
    axs[6].imshow(imgs[6])
    axs[6].axis('off')
    axs[6].set_title('(g) Contour features')
    plt.show()
    
def save_process(imgs, file_name):
    cv.imwrite('data/process/'+file_name+'_0_original.jpg', cv.cvtColor(imgs[0], cv.COLOR_RGB2BGR))
    cv.imwrite('data/process/'+file_name+'_1_preprocess.jpg', cv.cvtColor(imgs[1], cv.COLOR_RGB2BGR))
    cv.imwrite('data/process/'+file_name+'_2_clahe.jpg', imgs[2])
    cv.imwrite('data/process/'+file_name+'_3_otsu.jpg', imgs[3])
    cv.imwrite('data/process/'+file_name+'_4_closing.jpg', imgs[4])
    cv.imwrite('data/process/'+file_name+'_5_holes.jpg', imgs[5])
    cv.imwrite('data/process/'+file_name+'_6_contours.jpg', cv.cvtColor(imgs[6], cv.COLOR_RGB2BGR))