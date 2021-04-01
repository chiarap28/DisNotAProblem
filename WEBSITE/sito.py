from flask import Flask, render_template, url_for, request
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import pickle
from nltk.util import pad_sequence
from nltk.util import bigrams, trigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
from nltk import word_tokenize, sent_tokenize 
import os
import requests
import io
import re
import glob
import pandas as pd


app = Flask(__name__)

# ------------ ALFABETO --------------------- #

alf_UPP = {0: 'A',
           1: 'B',
           2: 'C',
           3: 'D',
           4: 'E',
           5: 'F',
           6: 'G',
           7: 'H',
           8: 'I',
           9: 'J',
           10: 'K',
           11: 'L',
           12: 'M',
           13: 'N',
           14: 'O',
           15: 'P',
           16: 'Q',
           17: 'R',
           18: 'S',
           19: 'T',
           20: 'U',
           21: 'V',
           22: 'W',
           23: 'X',
           24: 'Y',
           25: 'Z'}

alf_LOW = {0: 'a',
           1: 'b',
           2: 'c',
           3: 'd',
           4: 'e',
           5: 'f',
           6: 'g',
           7: 'h',
           8: 'i',
           9: 'j',
           10: 'k',
           11: 'l',
           12: 'm',
           13: 'n',
           14: 'o',
           15: 'p',
           16: 'q',
           17: 'r',
           18: 's',
           19: 't',
           20: 'u',
           21: 'v',
           22: 'w',
           23: 'x',
           24: 'y',
           25: 'z'}


 # ----------- WORD SEGMENTATION ------------ #
 
def prepareImg(img, height):
    "convert given image to grayscale image (if needed) and resize to desired height"
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts


# ---------- FINE WORD SEGMENTATION -----------#


# ---------- CHAR SEGMENTATION ----------------#

THRESH1 = 8
THRESH2 = 4

def average_col(pc):
    
    sp = []
    s = 0
    n = 0
    flag = False
    # compute average between closed black columns
    for i in range(len(pc)-1):
    
        if(pc[i+1]-pc[i] <= 3):

            s += pc[i]
            n += 1

            if(i == len(pc)-2):
                flag = True
                s += pc[i+1]
                n += 1
                sp.append(s//n)
        else:
            if(n>0):
                sp.append(s//n)
                n = 0
                s = 0
            else:                
                sp.append(pc[i])
                
    if not flag and sp!=[]:
        sp.append(pc[-1])
        
    return sp

def find_black_col(letterGray,h):
    """
    Find columns with all pixel black (background)
    from a binary inverted image; average closed columns;
    remove columns which are more closed than a threshold
    
    img: a binary inverted image with normalized pixel
    
    return a list with indexs of black columns
    """
    bc = []
    # sum each column pixel 
    col = np.sum(letterGray==0, axis=0)
    # save index of black columns
    black_col = np.where(col==h)
    black_col = black_col[0]
    
    bc = average_col(black_col)
        
    return bc

def char_seg(letterGray,a,b):
    """
    a: int
        start point
    b: int
        end point   
        
    return a list of potential segmentation points
    """
        
    col = np.sum(letterGray==1,axis=0)
    psp = []
    sp = []
    
    for i in range(a, b-1):
        
        if col[i+1]-col[i] >= THRESH2:

            psp.append(i)
    
    sp = average_col(psp)
            
    return sp

# ---------- FINE CHAR SEGMENTATION -----------#

@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/predict', methods=['POST'])
#def predict():
    
#    return render_template('index.html',prediction_text='BLA BLA BLABLAAAA')

@app.route('/generic')
def generic():
    return render_template('generic.html')

@app.route('/elements')
def elements():
    return render_template('elements.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/photo', methods=['POST'])
def photo():
    output = [x for x in request.form.values()]
    testo = ''
    
    if(output==[]):
        
        return render_template('elements.html', error='Carica prima l\'immagine :-)')
    
    if(output!=[]):
        src = 'images/'+output[-1]
        img = prepareImg(cv2.imread(src, cv2.IMREAD_GRAYSCALE), 1000)
        blur = cv2.GaussianBlur(img,(5,5),0)
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKernel)
        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        # Compute the Scharr gradient along the y-axis of the blackhat image
        gradY = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradY = np.absolute(gradY)
        # We then take this gradient image and scale it back into the range [0, 255] using min/max scaling
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        (minVal, maxVal) = (np.min(gradY), np.max(gradY))
        gradY = (255 * ((gradY - minVal) / (maxVal - minVal))).astype("uint8")
        close_img2 = cv2.morphologyEx(gradY+gradX, cv2.MORPH_CLOSE, rectKernel)
        (_, imgThres) = cv2.threshold(close_img2, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        thresh2 = cv2.erode(imgThres, None, iterations=1)
        (_, imgThres) = cv2.threshold(thresh2, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        
        components = grab_contours(cv2.findContours(imgThres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE))
        minArea= 100
        # append components to result   
        res = []
        i = 0

        for c in components:
            # skip small word candidates
            i += 1
            if cv2.contourArea(c) < minArea:
                continue
            # append bounding box and image of word to result list
            currBox = cv2.boundingRect(c) # returns (x, y, w, h)
            (x, y, w, h) = currBox
            
            #if h > 80:
            #    continue
            
            currImg = img[y:y+h, x:x+w]
            res.append((currBox, currImg))

        #sort segmented images
        list1 = sorted(res, key=lambda x: x[0][1])
        list2 = []

        k = 0

        for i in range(len(list1)-1):
            
            if list1[i][0][0] < list1[i+1][0][0]:
                
                list2.append(sorted(list1[k:i+1], key=lambda x: x[0][0]))
                k = i+1
        
        # to segment in character
        
        if(output[0]=='maiuscolo'):
        
            model_upp = load_model('model3_UPP_byclass_noDA.h5')
            
            
            for i in range(len(res)):

                gray = res[i][1]
                th, letterGray = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
                h = letterGray.shape[0]
                w = letterGray.shape[1]
                letterGray = letterGray/255
                
                bc = find_black_col(letterGray,h)
                
                new_col = []
                no_black_col = False

                if(bc!=[]):
                    
                    if(bc[0]==0):
                        bc.remove(0)
                    
                    if(bc!=[] and bc[-1]==w-1):
                        bc.remove(w-1)
                    
                    if(bc!=[] and bc[0] < THRESH1):

                        bc.remove(bc[0])

                    for i in range(len(bc)-1):

                        if(i==0):

                            if(bc[i] > THRESH1):

                                new_col.append(char_seg(letterGray,0, bc[i]))
                        
                        if(bc[i+1]-bc[i] > THRESH1):

                            new_col.append(char_seg(letterGray,bc[i], bc[i+1]))
                            
                    if(bc!=[] and (w-1)-bc[-1] > THRESH1):

                        new_col.append(char_seg(letterGray, bc[-1],w))
                        
                    if(len(bc)==1):
                        
                        new_col.append(char_seg(letterGray,0,bc[0]))
                        new_col.append(char_seg(letterGray,bc[0],w-1))
                        
                else:
                    
                    no_black_col = True
                    
                    new_col.append(char_seg(letterGray,0, w-1))
                    
                new_sp = []
    
                for i in range(len(new_col)):

                    for x in new_col[i]:

                        new_sp.append(x)
                        
                if(new_sp!=[]):
                    
                    new_sp = (sorted(list(set(new_sp))))
                    

                    if(new_sp[0] < THRESH1):
                        
                        new_sp.remove(new_sp[0])
                    
                if(new_sp!=[]):
                    
                    if((w-1)-new_sp[-1] < THRESH2):
                        
                        new_sp.remove(new_sp[-1])
                        
                final_col = []
                b = False

                for i in range(len(new_sp)):
                    
                    b = False
                    
                    for j in range(len(bc)):
                        
                        if(abs(bc[j]-new_sp[i]) < THRESH1):
                            
                            b=True
                            
                    if(not b):    
                        
                        final_col.append(new_sp[i])
            
                final_col = final_col+bc
                final_col = sorted(set(final_col))
                
                img_letters = []
                
                for i in range(len(final_col)-1):
                    
                    if i==0:
                        seg = gray[:,0:final_col[i]]
                        img_letters.append(seg)
                                                      
                    if i==len(final_col)-1:
                        seg = gray[:,final_col[i]:w]
                        img_letters.append(seg)
                                                      
                    else:
                        seg = gray[:,final_col[i]:final_col[i+1]]
                        img_letters.append(seg)
                
                #print(len(img_letters))
                for i in range(len(img_letters)):
                    
                    l = cv2.resize(img_letters[i], (28,28))
                    
                    letter = l.reshape((1,28,28,1)).astype('float32')
                    
                    y_pred = model_upp.predict(letter)
                    
                    y = y_pred.argmax()
                    
                    testo += alf_UPP[y]
                    
                testo += ' '
            
        elif output[0]=='minuscolo':    
            
            model_low = load_model('model1LOW_byclass_noDA.h5')

            for i in range(len(res)):
                
                gray = res[i][1]
                th, letterGray = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
                # normalize image pixel in range [0,1]
                letterGray = letterGray/255
                from skimage import io, color, morphology
                letterGray = morphology.thin(letterGray)
                
                h = letterGray.shape[0]
                w = letterGray.shape[1]
                
                # sum each column pixel 
                col = np.sum(letterGray==True,axis=0)
            
                csc = np.where(col<=1)
                csc = csc[0]
                
                sp = []
                s = 0
                n = 0
                flag = False
                # compute average between closed black columns
                for i in range(len(csc)-1):

                    if(csc[i+1]-csc[i] <= 5):

                        s += csc[i]
                        n += 1

                        if(i == len(csc)-2):
                            flag = True
                            s += csc[i+1]
                            n += 1
                            sp.append(s//n)
                    else:
                        if(n>0):
                            sp.append(s//n)
                            n = 0
                            s = 0
                        else:
                            sp.append(csc[i])
                        
                if not flag and sp!=[]:
                    sp.append(csc[-1])
                    
                img_letters = []
                
                if len(sp) > 1:
                    
                    if(sp[0]<=5):
                        
                        sp.remove(sp[0])
                        
                    elif(sp[-1]>=(w-5)):
                        
                        sp.remove(sp[-1])
                
                for i in range(len(sp)-1):
                    
                    if i==0:
                        seg = gray[:,0:sp[i]]
                        img_letters.append(seg)
                                                      
                    if i==len(sp)-1:
                        seg = gray[:,sp[i]:w]
                        img_letters.append(seg)
                                                      
                    else:
                        seg = gray[:,sp[i]:sp[i+1]]
                        img_letters.append(seg)
                        
                for i in range(len(img_letters)):
                    
                    l = cv2.resize(img_letters[i], (28,28))
                    
                    letter = l.reshape((1,28,28,1)).astype('float32')
                    
                    y_pred = model_low.predict(letter)
                    
                    y = y_pred.argmax()
                    
                    testo += alf_LOW[y]
                    
                testo += ' '
    
            
    return render_template('elements.html', message='La foto caricata è {}'.format(output[-1]), predict_text='TESTO: {}'.format(testo))

@app.route('/text', methods=['POST'])
def text():
    testo = [x for x in request.form.values()]
    input_text = testo[0]
    input_text = re.sub(r'\n', '' , input_text)
    input_text = re.sub(r'\t', '' , input_text)
    
    tok_test = word_tokenize(input_text.lower())
    trig = list(trigrams(tok_test))
    
    vocab_file = pd.read_csv('DIZIONARIO.csv', sep=';')
    vocab = list(vocab_file['TOKEN'])
    
    model = pickle.load(open('langmod.pickle', 'rb'))
    
    err = 0 
    
    for t in trig:
        
        if(t[2] not in vocab):
            err +=1
            continue
        if(model.score(t[2],[t[0],t[1]])==0):
            err +=1
    
    if(err<=2):
        punteggio = 'Sembra che non ci siano errori ortografici tipici della disortografia :-)' 
    elif(err>2 and err<=5):
        punteggio = 'Forse c\'è qualche errore ortografico che potrebbe far pensare alla presenza di disortografia. Controlla le doppie, lo scambio, l\'inserimento o la traslazione di lettere/sillabe, l\'H nei verbi e gli accenti!'
        
    elif(err>5):
        punteggio = 'Gli errori ortografici sembrano essere un po\' tantini e tipici della disortografia! Non preoccuparti, respira e - dopo aver controllato altri testi - consulta uno specialista :-)'
    
    return render_template('elements.html',score=punteggio)
    
if __name__=='__main__':
    app.run(debug=True)
