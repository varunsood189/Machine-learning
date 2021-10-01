#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:29:23 2021

@author: varun
"""
import os
import cv2
import math
import numpy as np
os.chdir("/home/varun/Downloads/ml_assignment/1/ELL784 Assignment 1_files")


#%% 
#240,352,3

def gaussian(x ,mean, sd):
    return     1/math.sqrt(math.pow(2*math.pi*abs(sd),3)) * (math.exp(-1/2*(x-mean)*(x-mean)/sd))

#%% 
def calculate_rpo(alpha, x, mean, sd):
    return alpha* gaussian(x,mean,sd)

#%%
 
def update_mean(rho,x,mean):
    return (1-rho)*mean+rho*x

#%%
def update_sd(rho,x,mean,sd):
    return (1-rho)*math.pow(sd,2)+rho*math.pow((x-mean),2)

#%%

# m = 1 for matched 0 for unmatched
def update_weight(alpha,weight,m):
    return (1-alpha)*weight+alpha*m
    
    
#%%

#creating a model of the the 
def sort_index(mean,weights,variance):
    sq_variance = [math.sqrt(var) for var in variance]
    index = np.argsort(weights/sq_variance)
#    print(weights[index[::-1]],sum(weights))
    weights = weights[index[::-1]]
    variance = weights[index[::-1]]
    mean = mean[index[::-1]]
    return mean, weights, variance
    
# sorr on the basis of w/varianve, all the weights

#%%

#creating a model of the the 
def sort_weight(mean,weights,variance):
    index = np.argsort(weights)
#    print(weights[index[::-1]],sum(weights))
    weights = weights[index[::-1]]
    variance = weights[index[::-1]]
    mean = mean[index[::-1]]
    return mean, weights, variance
    
# sorr on the basis of w/varianve, all the weights

#%% 

def normalize_weight(weight):
    return weight/weight.sum()
    
#%%


def strauffer_grimson(image):
    global mean,variance, weights,alpha
    match = 1
    background_image = np.random.normal(0,1,size= (240,352))

    # match with the best 
    for i,row in enumerate(image):
        for j,value in enumerate(row):
           #print(i,j,value)
            # sorting the mean , weight and variance on the basis of the weight/variance
            weights[i][j] = normalize_weight(weights[i][j])                
            mean[i][j], weights[i][j] ,variance[i][j] =  sort_index(mean[i][j], weights[i][j], variance[i][j])
            # we have for each pixel total 5 different gaussians which are sorted according to importance
            
            #find selected guassian mixture model
            #for each pixel we need to select one of the guassians
            distance_gaussian=[abs((value-mean[i][j][k])/variance[i][j][k]) for k in range(0,5)]

            #print("True", max([abs((value-mean[i][j][k])/variance[i][j][k]) for k in range(0,5)]))
            #gaussian_value = [ gaussian(value, mean[i][j][k], variance[i][j][k]) for k in range(0,5)]            
            #max_value = max(gaussian_value)
            #max_index = gaussian_value.index(max_value) 
            
            #no match found, replace the last one
            if(min(distance_gaussian)>2.5):
                mean[i][j][-1] = value 
                variance[i][j][-1] = 1 # check for variance
                weights[i][j][-1]=0.1
            else:
                min_value = min(distance_gaussian)
                min_index = distance_gaussian.index(min_value) 
                #print(distance_gaussian,distance_gaussian[min_index])
                rho = calculate_rpo(alpha, value, mean[i][j][min_index ], variance[i][j][min_index ])
                mean[i][j][min_index ] = update_mean(rho,value,mean[i][j][min_index ])
                variance[i][j][min_index ] = update_sd(rho,value,mean[i][j][min_index ],variance[i][j][min_index ])
                weights[i][j] = [  update_weight(alpha,weights[i][j][k],1) if k ==min_index else  update_weight(alpha,weights[i][j][k],0) for k in range(0,5)]
            
            weights[i][j] = normalize_weight(weights[i][j])                
            mean[i][j], weights[i][j] ,variance[i][j] =  sort_weight(mean[i][j], weights[i][j], variance[i][j])
            #print(weights[i][j])
            sum_weight=0
            sum_mean=0
            for k in range(0,5):
                sum_weight+=weights[i][j][k]
                sum_mean+=mean[i][j][k]*weights[i][j][k]
                if(sum_weight>threshold):
                    background_image[i][j]=sum_mean/(k+1)                    
                    break
                
    return background_image
                    
    
#%% 
# open and view the video
np.random.seed(10)
mean = np.random.normal(0,1,size= (240,352,5))
variance = np.random.normal(0,1,size= (240,352,5))
weights = np.random.normal(0,1,size= (240,352,5))

mean = (mean - np.amin(mean) )/ ( np.amax(mean)- np.amin(mean))
variance = (variance - np.amin(variance) )/ ( np.amax(variance)- np.amin(variance))
weights = (weights - np.amin(weights) )/ ( np.amax(weights)- np.amin(weights))
 
#weight = [normalize_weight(weight[i][j]) for j,_ in enumerate(i) for i,_ in enumerate(weight)]

alpha = 0.3
threshold = 0.5
size = (240,352)
cap = cv2.VideoCapture('umcp.mpg')
result_background = cv2.VideoWriter('background.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
if (cap.isOpened()== False):
    print("Error opening video stream or file")
# Read until video is completed
    
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        norm_gray = (gray - np.amin(gray) )/ ( np.amax(gray)- np.amin(gray))
        # Display the resulting frame
        cv2.imshow('Frame',gray)
        background = strauffer_grimson(norm_gray)
        normal_background= (background - np.amin(background) )/ ( np.amax(background)- np.amin(background))
        normal_background = 255 * normal_background
        normal_background = normal_background.astype(np.uint8)
        cv2.imshow('background_background ',normal_background)
        cv2.imshow('fore_background ',normal_background-gray)
        
        result_background.write(normal_background) 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# When everything done, release the video capture object
cap.release()
result_background.release()

# Closes all the frames
cv2.destroyAllWindows()
