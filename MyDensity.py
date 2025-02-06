#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:08:32 2022

@author: laurahenry
"""



import pylab
from pylab import *
ion()

import glob 
import os 
import numpy as np
import sys
import time
import pickle #
import scipy
from scipy import interpolate
from scipy.optimize import leastsq
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import leastsq
from scipy.ndimage import label
from matplotlib.path import Path # for masking EQ image
import pandas as pd
import csv
import tkinter
from tkinter import *
import matplotlib.pyplot as plt
import time
import itertools
import protocol
import json

try:
    import PyTango
except:
    print(" ")
    import colorama
    from colorama import Fore
    
try:
    from spyc.core import DeviceManager
    dm = DeviceManager()
except:
    print(" ")
import os
import pylab
pylab.ion()
try:
    import CaesarDataMultiChannel
    mc = CaesarDataMultiChannel.CaesarData()
except:
    mc = 0


class Density():
    ccup = "\x1b\x5b\x41"

    #init
    def __init__(self):
      # self
      #print("This is the current version for acquisition and data analysis (March 2024)... Welcome !")
      self.parameters = False
      


    def setParameters(self, filename = None, rsam = None, renv = None, energy = None, delta_E = None, mu_pho_abs = None, mu_pho_env =None, allChannel = True):
        print( '###############################################################')
        print( "DEFINE SETUP PARAMETERS FOR ACQUISITION AND ANALYSIS")
        print( '###############################################################')
        self.filename = filename
        if self.filename == None:
            user_input = input("Define the filename : ")
            self.filename = str(user_input)
        self.data_filename = self.filename +'.pickle'

        self.rsam = rsam
        if self.rsam == None:        
            user_input = float(input("What is the sample diameter (in mm) : "))
            self.rsam = user_input
        self.rsam /=20.

        self.renv = renv
        if self.renv == None:
            user_input = float(input("What is the capsule diameter (in mm) : "))
            self.renv = user_input
        self.renv/=20.
 
        self.energy = energy
        if self.energy == None:
                user_input = float(input("Energy used for data analysis (in keV) : "))
                self.energy = user_input
        self.delta_E = delta_E
        if self.delta_E == None:
            user_input = float(input("Delta_E for data analysis (in keV) : "))
            self.delta_E = user_input
        
        self.mu_pho_abs = mu_pho_abs
        if self.mu_pho_abs == None:
            user_input = float(input('Initial guess for SAMPLE \u03BC.\u03C1 (g/cm^2): '))
            self.mu_pho_abs = user_input
        
        self.mu_pho_env = mu_pho_env
        if self.mu_pho_env == None:        
            user_input = float(input('Initial guess for capsule \u03BC.\u03C1 (g/cm2): '))
            self.mu_pho_env = user_input
        self.r_sam = self.rsam
        self.r_env = self.renv
        self.cst=1
        self.r0= self.r_env
        self.corrlin = False
        self.zstart = 0   #later we'll implement the angle scan and data treatment
        self.zend = 0    #later we'll implement the angle scan and data treatment
        self.allChannel = allChannel
        if self.allChannel == True:
            self.ch00= True
            self.ch01= True
            self.ch02= True
            self.ch03= True
            self.ch04= True
            self.ch05= True
            self.ch06= True
        else:
            print('***************************************************************')
            print('LIMITING Ge ELEMENTS IN THE DATA ANALYSIS DUE TO THE ANVIL GAP ')
            print('***************************************************************')
            user_input = str(input('counting with ch00 [y]/n ? '))
            if user_input =='y' or user_input =='yes' or user_input =='Y' or user_input =='':
                self.ch00= True #
            else:
                self.ch00 = False
            user_input = str(input('counting with ch01 [y]/n ? '))
            if user_input =='y' or user_input =='yes' or user_input =='Y' or user_input =='':
                self.ch01= True 
            else:
                self.ch01 = False
            user_input = str(input('counting with ch02 [y]/n ? '))
            if user_input =='y' or user_input =='yes' or user_input =='Y' or user_input =='':
                self.ch02= True 
            else:
                self.ch02 = False
            user_input = str(input('counting with ch03 [y]/n ? '))
            if user_input =='y' or user_input =='yes' or user_input =='Y' or user_input =='':
                self.ch03= True 
            else:
                self.ch03 = False
            user_input = str(input('counting with ch04 [y]/n ? '))
            if user_input =='y' or user_input =='yes' or user_input =='Y' or user_input =='':
                self.ch04= True 
            else:
                self.ch04 = False
            user_input = str(input('counting with ch05 [y]/n ? '))
            if user_input =='y' or user_input =='yes' or user_input =='Y' or user_input =='':
                self.ch05= True 
            else:
                self.ch05 = False
            user_input = str(input('counting with ch06 [y]/n ? '))
            if user_input =='y' or user_input =='yes' or user_input =='Y' or user_input =='':
                self.ch06= True 
            else:
                self.ch06 = False
        self.CHs = [self.ch00, self.ch01, self.ch02, self.ch03, self.ch04, self.ch05, self.ch06]
        self.parameters = True

    def setExpDir(self, filepath = None):
        self.filepath = filepath
        if self.filepath == None:
            self.filepath = str(input("Please enter the full path /nfs/ruche/psiche-soleil/com-psiche/proposal/2023/.../ :"))
        else:
            self.filepath = filepath
        
        #print("This has not been properly integrated yet, need to copy Andrew's code from CaesarData")
       
    def findParameters(self, filename = None):
        print('###############################################################')
        print( "POST-TREATMENT : LOOKING FOR THE BEST ENERGY")
        print('###############################################################')
        self.filename = filename
        if self.filename == None:
            user_input = input("Define the filename : ")
            self.filename = str(user_input)
        self.data_filename = self.filename +'.pickle'
        self.data = pd.read_pickle(self.filepath + '/' +self.data_filename) 
        self.xrd_raw = self.data['data']
        
        self.energy = 0.0495*np.arange(1,2049) #approximative
        plt.close("Looking for parameters")
        fig = plt.figure(figsize = (10,5), num = "Looking for parameters")

        gs = fig.add_gridspec(2,3)
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax2 = fig.add_subplot(gs[1, 0:2])
        ax3 = fig.add_subplot(gs[:, 2])
        ax3.set_title('raw absorption profile')

        self.center_scan = int(self.xrd_raw.shape[0]/2)
        ax1.plot(self.energy , self.xrd_raw[0,:,:].mean(1), 'k-', label = 'raw_XRD (outside)')
        ax1.plot(self.energy , self.xrd_raw[self.center_scan,:,:].mean(1), 'b-', label = 'raw_XRD (center)')
        ax1.set_title('click to display abs. profile')
        ax1.set_xlabel('energy (keV)')
        ax1.set_ylabel('intensity (a.u.)')
        ax1.axis('tight')
        ax1.legend()
        self.detector_image = np.zeros((7,2048))
        for i in range(7):
            self.detector_image[i,:]=self.xrd_raw[0,:,i]
        ax2.imshow(self.detector_image, cmap = 'cool', extent = [self.energy[0], self.energy[-1], 0, 7])
        ax2.axis('tight')
        ax2.set_xlabel('energy (keV)')
        ax2.set_ylabel('detector channel')
        ax2.axis('tight')

        #ax[1].set_title('detector image for channel selection')
        num = []
        def onclick(event):
            cmap = get_cmap('Dark2')
            ix = event.xdata
            print ('Energy = ' + str('%.2f' %ix) + 'keV')  
            ax1.vlines(ix, 0, np.max( self.xrd_raw[0,:,:].mean(1)), colors = cmap(len(num)) , linestyles='solid', label= str('%.1f' %ix) + 'keV')
            ax1.legend()
            plt.show()
            self.Chan1tmp = int((ix*1000 - .5*1000)/50)
            self.Chan2tmp = int((ix*1000 + .5)*1000/50)
            self.d_tmp = self.xrd_raw[:,self.Chan1tmp:self.Chan2tmp,:].mean(1).mean(1)            
            ax3.plot(self.d_tmp, marker='o', markerfacecolor = cmap(len(num)),label= str('%.1f' %ix) + ' (±.5) keV')
            ax3.axis('tight')
            ax3.legend(loc = 'lower right')
            num.append(ix)

        cid = fig.canvas.mpl_connect('button_press_event', onclick)


    def doDensity(self, mc = mc, xstart=-1, xend=1, xstep=0.02, ctime=10, motor = 'ptomorefx'):
        
        if motor =='ptomorefx' or 'ptx':
            pos_init = dm.actors[motor].value * 1.
        else:    
            pos_init = 'toto'
                
        if self.parameters == True:
            savename = self.filepath+ '/' +self.filename +'.pickle'
        
            while(os.path.exists(savename)):
                print('filename already exists!')
                self.filename = input("Choose a new name : ")
                savename = self.filepath +self.filename +'.pickle'
        
            machinecurrent = PyTango.AttributeProxy('ans/ca/machinestatus/current')
            diode_Izero = PyTango.AttributeProxy('i03-c-c00/ca/sai.1/averagechannel1')
        
            xpos = np.arange(xstart, xend, xstep)
        
            mc._CaesarData__DP.presetValue = ctime
            mc._CaesarData__DP.presetType = 'FIXED_REAL'
        
            data = {'xstart':xstart, 'xend':xend, 'xstep':xstep, 'ctime':ctime}
        
            output = np.zeros((len(xpos), 2048, 7))
            livetime = np.zeros((len(xpos),7))
            deadtime = np.zeros((len(xpos),7))
            diode = np.zeros((len(xpos),2))
            beamcurrent = np.zeros((len(xpos),2))
            #get_ipython().magic('feopen')
            get_ipython().magic('fwbsopen')
            time.sleep(2)
        
            try:
                for ii,xx in enumerate(xpos):
                    diode[ii,0] = diode_Izero.read().value * 1.
                    beamcurrent[ii,0] = machinecurrent.read().value * 1.
        
                    get_ipython().magic('amove %s %f' %(motor, xx))  #to be check with Andy
                    mc._CaesarData__DP.snap()
                    time.sleep(ctime)
                    while mc._CaesarData__DP.State() != PyTango._PyTango.DevState.STANDBY:
                        time.sleep(0.5)
                    time.sleep(0.5)
        
                    diode[ii,1] = diode_Izero.read().value * 1.
                    beamcurrent[ii,1] = machinecurrent.read().value * 1.
        
        
                    output[ii, :, 0] = mc._CaesarData__DP.channel00 * 1
                    livetime[ii, 0] = mc._CaesarData__DP.livetime00 * 1.
                    deadtime[ii,0] = mc._CaesarData__DP.deadtime00 * 1.
        
                    output[ii, :,1] = mc._CaesarData__DP.channel01 * 1
                    livetime[ii,1] = mc._CaesarData__DP.livetime01 * 1.
                    deadtime[ii,1] = mc._CaesarData__DP.deadtime01 * 1.
        
                    output[ii, :,2] = mc._CaesarData__DP.channel02 * 1
                    livetime[ii,2] = mc._CaesarData__DP.livetime02 * 1.
                    deadtime[ii,2] = mc._CaesarData__DP.deadtime02 * 1.
        
                    output[ii, :,3] = mc._CaesarData__DP.channel03 * 1
                    livetime[ii,3] = mc._CaesarData__DP.livetime03 * 1.
                    deadtime[ii,3] = mc._CaesarData__DP.deadtime03 * 1.
        
                    output[ii, :,4] = mc._CaesarData__DP.channel04 * 1
                    livetime[ii,4] = mc._CaesarData__DP.livetime04 * 1.
                    deadtime[ii,4] = mc._CaesarData__DP.deadtime04 * 1.
        
                    output[ii, :,5] = mc._CaesarData__DP.channel05 * 1
                    livetime[ii,5] = mc._CaesarData__DP.livetime05 * 1.
                    deadtime[ii,5] = mc._CaesarData__DP.deadtime05 * 1.
        
                    output[ii, :,6] = mc._CaesarData__DP.channel06 * 1
                    livetime[ii,6] = mc._CaesarData__DP.livetime06 * 1.
                    deadtime[ii,6] = mc._CaesarData__DP.deadtime06 * 1.
            except:
                print("There was a problem...")
            finally:
                data.update({'data':output})
                data.update({'livetime':livetime})
                data.update({'deadtime':deadtime})
                data.update({'diode':diode})
                data.update({'beamcurrent':beamcurrent})
                pickle.dump(data, open(savename, 'wb'))
                print("saved file %s" % savename)
                
                print("This is closing the front end ")
        
                pylab.figure()
                pylab.imshow(output[:,:,3])
                pylab.axis('tight')
        else:
            print('You need to run setParameters() first')

        if pos_init == 'toto':
            print("not moving any motor")
        else: 
            print("Moving back " + motor + " to original position : "+ str(pos_init))
            get_ipython().magic('amove %s %f' %(motor, pos_init))      


    def doDensityAngle(self, mc = mc, xstart=-1, xend=1, zstart = 1, zend = -1, nb_pts = 100, ctime=10, motor = 'ptomorefx'):
        
        if motor =='ptomorefx' or 'ptx':
            pos_initX = dm.actors[motor].value * 1.
        else:    
            pos_initX = 'toto'
        
        pos_initZ = dm.actors['ptz'].value * 1.
                
        if self.parameters == True:
            savename = self.filepath+ '/' +self.filename +'.pickle'
        
            while(os.path.exists(savename)):
                print('filename already exists!')
                self.filename = input("Choose a new name : ")
                savename = self.filepath +self.filename +'.pickle'
        
            machinecurrent = PyTango.AttributeProxy('ans/ca/machinestatus/current')
            diode_Izero = PyTango.AttributeProxy('i03-c-c00/ca/sai.1/averagechannel1')
        
            xpos = np.linspace(xstart, xend, nb_pts)
            zpos = np.linspace(zstart, zend, nb_pts) # linspace here
            xstep = math.sqrt((zend - zstart)**2+(xend-xstart)**2)/nb_pts
            mc._CaesarData__DP.presetValue = ctime
            mc._CaesarData__DP.presetType = 'FIXED_REAL'
        
            data = {'xstart':xstart, 'xend':xend, 'zstart':zstart, 'zend':zend,'xstep':xstep, 'ctime':ctime}
        
            output = np.zeros((len(xpos), 2048, 7))
            livetime = np.zeros((len(xpos),7))
            deadtime = np.zeros((len(xpos),7))
            diode = np.zeros((len(xpos),2))
            beamcurrent = np.zeros((len(xpos),2))
            #get_ipython().magic('feopen')
            get_ipython().magic('fwbsopen')
            time.sleep(2)
        
            try:
                for ii,xx in enumerate(xpos):
                    zz = zpos[ii]
                    diode[ii,0] = diode_Izero.read().value * 1.
                    beamcurrent[ii,0] = machinecurrent.read().value * 1.
        
                    get_ipython().magic('amove %s %f %s %f' %(motor, xx, 'ptz', zz))  
                    mc._CaesarData__DP.snap()
                    time.sleep(ctime)
                    while mc._CaesarData__DP.State() != PyTango._PyTango.DevState.STANDBY:
                        time.sleep(0.5)
                    time.sleep(0.5)
        
                    diode[ii,1] = diode_Izero.read().value * 1.
                    beamcurrent[ii,1] = machinecurrent.read().value * 1.
        
        
                    output[ii, :, 0] = mc._CaesarData__DP.channel00 * 1
                    livetime[ii, 0] = mc._CaesarData__DP.livetime00 * 1.
                    deadtime[ii,0] = mc._CaesarData__DP.deadtime00 * 1.
        
                    output[ii, :,1] = mc._CaesarData__DP.channel01 * 1
                    livetime[ii,1] = mc._CaesarData__DP.livetime01 * 1.
                    deadtime[ii,1] = mc._CaesarData__DP.deadtime01 * 1.
        
                    output[ii, :,2] = mc._CaesarData__DP.channel02 * 1
                    livetime[ii,2] = mc._CaesarData__DP.livetime02 * 1.
                    deadtime[ii,2] = mc._CaesarData__DP.deadtime02 * 1.
        
                    output[ii, :,3] = mc._CaesarData__DP.channel03 * 1
                    livetime[ii,3] = mc._CaesarData__DP.livetime03 * 1.
                    deadtime[ii,3] = mc._CaesarData__DP.deadtime03 * 1.
        
                    output[ii, :,4] = mc._CaesarData__DP.channel04 * 1
                    livetime[ii,4] = mc._CaesarData__DP.livetime04 * 1.
                    deadtime[ii,4] = mc._CaesarData__DP.deadtime04 * 1.
        
                    output[ii, :,5] = mc._CaesarData__DP.channel05 * 1
                    livetime[ii,5] = mc._CaesarData__DP.livetime05 * 1.
                    deadtime[ii,5] = mc._CaesarData__DP.deadtime05 * 1.
        
                    output[ii, :,6] = mc._CaesarData__DP.channel06 * 1
                    livetime[ii,6] = mc._CaesarData__DP.livetime06 * 1.
                    deadtime[ii,6] = mc._CaesarData__DP.deadtime06 * 1.
            except:
                print("There was a problem...")
            finally:
                data.update({'data':output})
                data.update({'livetime':livetime})
                data.update({'deadtime':deadtime})
                data.update({'diode':diode})
                data.update({'beamcurrent':beamcurrent})
                pickle.dump(data, open(savename, 'wb'))
                print("saved file %s" % savename)
                
                print("This is closing the front end ")
                get_ipython().magic('feclose')
        
                pylab.figure()
                pylab.imshow(output[:,:,3])
                pylab.axis('tight')
        else:
            print('You need to run setParameters() first')

        if pos_init == 'toto':
            print("not moving any motor")
        else: 
            print("Moving back " + motor + " to original position : "+ str(pos_initX))
            get_ipython().magic('amove %s %f' %(motor, pos_initX))  
        print("Moving back ptz to original position : "+ str(pos_initZ))    
        get_ipython().magic('amove %s %f' %(ptz, pos_initZ))      



    def loadData(self):
        print ('Preprocess data...')
        print('Reading ' + self.data_filename)
        self.data = pd.read_pickle(self.filepath + '/' +self.data_filename) 
        datay=self.data['data'][:,0,3]#modif
        self.data_cEP=np.zeros((len(datay),2048,7))
        channels=np.arange(1,2049)
        self.x1 = 0
        self.y1 = self.data['xend']-self.data['xstart']
        self.xstep =self.data['xstep']
        self.step=len(self.data['data'])
        step = self.step
        self.x1*=.1
        self.y1*=.1
        self.detectors=np.arange(0,7)
        for i in range(len(self.detectors)):
            # The CaesarData class:
            class CaesarData:
            
                def __init__(self, detector="i03-C-CX1/dt/dtc-mca_xmap.1", dataname="channel03", experimentdir=""):
                    self.energies = np.zeros(2048)
                    # add some empty attributes for the caesar dataset
                    self.caesar_image = np.zeros((step,2048)) #Rajout du step
                    self.caesar_image_original =self.caesar_image * 1.
                    
                def __getitem__(self,datay,data): 
                    for ii in range (len(datay)):
                        self.caesar_image[ii, :] = data['data'][ii,:,i]
                        ii+=1
                
            
                def BuildCaesar(self,channels):
                    
                    self.energies = 0.013 + 0.04976*channels 
                    self.energies[np.nonzero(self.energies<0.0000001)]=0.0000001  
                    self.caesar_image_original = self.caesar_image * 1.
                    #print(self.energies)
            
                def correctEscapePeaks(self):
                    '''Correct escape peaks using function derived in Clark_0416
                        probably works best for amorphous things becaues it doesn't 
                        treat peak widths'''
                    imE = self.caesar_image * 1.
                    escapef = 0.41501256 * np.exp(self.energies * -0.08536899)
                    escapef = np.tile(escapef, (imE.shape[0], 1))
                    ndx = np.nonzero(self.energies>20)[0][0]
                    dE = self.energies[ndx+1] - self.energies[ndx] # channel width at 20 keV
                    dchan = -int(np.round(9.876 / dE))
                    tmp = imE*escapef # escape intensity
                    tmp2 = np.roll(tmp, dchan, 1) # observed escape peaks
                    imEcor = imE + tmp - tmp2 # add the lost intensity, remove the escape peaks
                    self.caesar_image_original = self.caesar_image * 1.
                    self.caesar_image = imEcor
                    return imEcor
            montest=CaesarData()
            montest.__getitem__(datay,self.data)
            montest.BuildCaesar(channels)
            self.data_cEP[:,:,i]=montest.correctEscapePeaks()
        self.Chan1 = int((self.energy*1000 - self.delta_E*1000)/50)
        self.Chan2 = int((self.energy + self.delta_E)*1000/50)
        self.d=[] #d=np.zeros(step)
        for ii, jj in enumerate((self.detectors)):
            
            if self.CHs[jj] is True:
                self.d.append((self.data_cEP[:,self.Chan1:self.Chan2,jj].mean(1))/(self.data['livetime'][:,jj]))
            else :
                ii+=1
        self.xaxis = np.linspace(self.x1,self.y1, self.step)
        self.d=np.asarray(self.d)
        self.d=self.d.transpose()
        self.d=self.d[:,:].sum(1)
        plt.close('Data')
        plt.figure('Data')
        plt.plot(self.xaxis, self.d,'ko', label = str(self.data_filename[:-7]))
        plt.legend()

    def removeBackground(self, bg_filename, scale = 1):
        self.scale = scale
        print (Fore.RED, 'Loading and removing background...')
        self.bg_data_filename = str(bg_filename)+ '.pickle'
        print(Fore.WHITE, 'Reading ' + self.bg_data_filename )
        self.bg_data = pd.read_pickle(r''+self.filepath+str(self.bg_data_filename)) 
        bg_datay=self.bg_data['data'][:,0,3]#modif
        self.bg_data_cEP=np.zeros((len(bg_datay),2048,7))
        channels=np.arange(1,2049)
        self.bg_x1 = 0
        self.bg_y1 = sqrt((self.bg_data['xend']-self.bg_data['xstart'])**2+(self.zend-self.zstart)**2)
        self.bg_xstep =self.bg_data['xstep']
        self.bg_step=len(self.bg_data['data'])
        step = self.bg_step
        self.bg_x1*=.1
        self.bg_y1*=.1
        self.detectors=np.arange(0,7)
        for i in range(len(self.detectors)):
            # The CaesarData class:
            class CaesarData:
            
                def __init__(self, detector="i03-C-CX1/dt/dtc-mca_xmap.1", dataname="channel03", experimentdir=""):
                    self.energies = np.zeros(2048)
                    # add some empty attributes for the caesar dataset
                    self.caesar_image = np.zeros((step,2048)) #Rajout du step
                    self.caesar_image_original =self.caesar_image * 1.
                    
                def __getitem__(self,datay,data): 
                    for ii in range (len(datay)):
                        self.caesar_image[ii, :] = data['data'][ii,:,i]
                        ii+=1
                
            
                def BuildCaesar(self,channels):
                    
                    self.energies = 0.013 + 0.04976*channels 
                    self.energies[np.nonzero(self.energies<0.0000001)]=0.0000001  
                    self.caesar_image_original = self.caesar_image * 1.
                    #print(self.energies)
            
                def correctEscapePeaks(self):
                    '''Correct escape peaks using function derived in Clark_0416
                        probably works best for amorphous things becaues it doesn't 
                        treat peak widths'''
                    imE = self.caesar_image * 1.
                    escapef = 0.41501256 * np.exp(self.energies * -0.08536899)
                    escapef = np.tile(escapef, (imE.shape[0], 1))
                    ndx = np.nonzero(self.energies>20)[0][0]
                    dE = self.energies[ndx+1] - self.energies[ndx] # channel width at 20 keV
                    dchan = -int(np.round(9.876 / dE))
                    tmp = imE*escapef # escape intensity
                    tmp2 = np.roll(tmp, dchan, 1) # observed escape peaks
                    imEcor = imE + tmp - tmp2 # add the lost intensity, remove the escape peaks
                    self.caesar_image_original = self.caesar_image * 1.
                    self.caesar_image = imEcor
                    return imEcor
            montest=CaesarData()
            montest.__getitem__(bg_datay,self.bg_data)
            montest.BuildCaesar(channels)
            self.bg_data_cEP[:,:,i]=montest.correctEscapePeaks()
        # self.Chan1 = int((self.energy*1000 - self.delta_E*1000)/50)
        # self.Chan2 = int((self.energy + self.delta_E)*1000/50)
        self.bg_d=[] #d=np.zeros(step)
        for ii, jj in enumerate((self.detectors)):
            
            if self.CHs[jj] is True:
                self.bg_d.append((self.bg_data_cEP[:,self.Chan1:self.Chan2,jj].mean(1))/(self.data['livetime'][:,jj]))
            else :
                ii+=1
        self.bg_xaxis = np.linspace(self.bg_x1,self.bg_y1, self.bg_step)
        self.bg_d=np.asarray(self.bg_d)
        self.bg_d=self.bg_d.transpose()
        self.bg_d=self.bg_d[:,:].sum(1)
        plt.close('Background removal')
        plt.figure('Background removal')
        scan_data=interpolate.interp1d(self.bg_xaxis, self.bg_d, kind='cubic',bounds_error=False, fill_value="extrapolate")
        self.bg_d = scan_data(self.xaxis)
        plt.plot(self.xaxis, self.d,'o',label = str(self.data_filename))
        plt.plot(self.xaxis, self.bg_d*self.scale,'x',label = str(self.bg_data_filename))
        plt.plot(self.xaxis, self.d/(self.scale*self.bg_d),'x',label = "difference")
        plt.legend()
        self.d = self.d/(self.scale*self.bg_d)

    def removeBackground_click(self, bg_filename):
        print (Fore.RED, 'Loading and removing background...')
        self.bg_data_filename = str(bg_filename)+ '.pickle'
        print(Fore.WHITE, 'Reading ' + self.bg_data_filename )
        self.bg_data = pd.read_pickle(r''+self.filepath+str(self.bg_data_filename)) 
        bg_datay=self.bg_data['data'][:,0,3]#modif
        self.bg_data_cEP=np.zeros((len(bg_datay),2048,7))
        channels=np.arange(1,2049)
        self.bg_x1 = 0
        self.bg_y1 = sqrt((self.bg_data['xend']-self.bg_data['xstart'])**2+(self.zend-self.zstart)**2)
        self.bg_xstep =self.bg_data['xstep']
        self.bg_step=len(self.bg_data['data'])
        step = self.bg_step
        self.bg_x1*=.1
        self.bg_y1*=.1
        self.detectors=np.arange(0,7)
        for i in range(len(self.detectors)):
            # The CaesarData class:
            class CaesarData:
            
                def __init__(self, detector="i03-C-CX1/dt/dtc-mca_xmap.1", dataname="channel03", experimentdir=""):
                    self.energies = np.zeros(2048)
                    # add some empty attributes for the caesar dataset
                    self.caesar_image = np.zeros((step,2048)) #Rajout du step
                    self.caesar_image_original =self.caesar_image * 1.
                    
                def __getitem__(self,datay,data): 
                    for ii in range (len(datay)):
                        self.caesar_image[ii, :] = data['data'][ii,:,i]
                        ii+=1
                
            
                def BuildCaesar(self,channels):
                    
                    self.energies = 0.013 + 0.04976*channels 
                    self.energies[np.nonzero(self.energies<0.0000001)]=0.0000001  
                    self.caesar_image_original = self.caesar_image * 1.
                    #print(self.energies)
            
                def correctEscapePeaks(self):
                    '''Correct escape peaks using function derived in Clark_0416
                        probably works best for amorphous things becaues it doesn't 
                        treat peak widths'''
                    imE = self.caesar_image * 1.
                    escapef = 0.41501256 * np.exp(self.energies * -0.08536899)
                    escapef = np.tile(escapef, (imE.shape[0], 1))
                    ndx = np.nonzero(self.energies>20)[0][0]
                    dE = self.energies[ndx+1] - self.energies[ndx] # channel width at 20 keV
                    dchan = -int(np.round(9.876 / dE))
                    tmp = imE*escapef # escape intensity
                    tmp2 = np.roll(tmp, dchan, 1) # observed escape peaks
                    imEcor = imE + tmp - tmp2 # add the lost intensity, remove the escape peaks
                    self.caesar_image_original = self.caesar_image * 1.
                    self.caesar_image = imEcor
                    return imEcor
            montest=CaesarData()
            montest.__getitem__(bg_datay,self.bg_data)
            montest.BuildCaesar(channels)
            self.bg_data_cEP[:,:,i]=montest.correctEscapePeaks()
        # self.Chan1 = int((self.energy*1000 - self.delta_E*1000)/50)
        # self.Chan2 = int((self.energy + self.delta_E)*1000/50)
        self.bg_d=[] #d=np.zeros(step)
        for ii, jj in enumerate((self.detectors)):
            
            if self.CHs[jj] is True:
                self.bg_d.append((self.bg_data_cEP[:,self.Chan1:self.Chan2,jj].mean(1))/(self.data['livetime'][:,jj]))
            else :
                ii+=1
        self.bg_xaxis = np.linspace(self.bg_x1,self.bg_y1, self.bg_step)
        self.bg_d=np.asarray(self.bg_d)
        self.bg_d=self.bg_d.transpose()
        self.bg_d=self.bg_d[:,:].sum(1)
        
        global cid, fig, coords, nclicks
        coords = []
        plt.close('Background removal')
        fig = plt.figure('Background removal')
        gs = fig.add_gridspec(1,2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        scan_data=interpolate.interp1d(self.bg_xaxis, self.bg_d, kind='cubic',bounds_error=False, fill_value="extrapolate")
        self.bg_d = scan_data(self.xaxis)
        ax1.plot(self.xaxis, self.d, marker = 'o', color = 'k', mfc='none', label = str(self.data_filename[:-7]))
        ax1.plot(self.xaxis, self.bg_d,'rx',label = str(self.bg_data_filename[:-7]))
        #ax1.plot(self.xaxis, self.d/(self.scale*self.bg_d),'gx',label = "difference")
        ax1.legend()
        nclicks = 4

        def onclick2(event):
            ix, iy = event.xdata, event.ydata
            if len(coords)!=0:
                sys.stdout.write("\b"*3)
                sys.stdout.flush()
            sys.stdout.write("(point %d)" % (len(coords)+1)+'\n')
            sys.stdout.flush()
            coords.append((ix, iy))
            ax1.scatter(ix, iy, s=40, c='b', marker='o')
            plt.show()
            if len(coords)==nclicks:
                fig.canvas.mpl_disconnect(cid)
                self.bg_linCorr = np.ones(len(self.bg_d))
                # Correcting data and background by linear correction on click
                a_linCorr = (coords[1][1]-coords[0][1])/(coords[1][0]-coords[0][0])
                b_linCorr = coords[0][1] - a_linCorr*coords[0][0]
                a_datalinCorr = (coords[3][1]-coords[2][1])/(coords[3][0]-coords[2][0])
                b_datalinCorr = coords[2][1] - a_datalinCorr*coords[2][0]
                self.bg_linCorr = self.xaxis*a_linCorr + b_linCorr
                self.d_linCorr = self.xaxis*a_datalinCorr + b_datalinCorr

                ax1.plot(self.xaxis, self.d_linCorr,'k-',label = str(self.data_filename[:-7])+"_corr")
                ax1.plot(self.xaxis, self.bg_linCorr,'r-',label = str(self.bg_data_filename[:-7])+"_corr")

                ax2.plot(self.xaxis, self.d/self.d_linCorr,'k-',label = str(self.data_filename[:-7])+"_corr")
                ax2.plot(self.xaxis, self.bg_d/self.bg_linCorr,'r-',label = str(self.bg_data_filename[:-7])+"_corr")

                self.d = (self.d/self.d_linCorr)/(self.bg_d/self.bg_linCorr)
                ax2.plot(self.xaxis, self.d,marker = 'o', color = 'b', mfc='none',label = "Corrected from bkg")
                ax2.legend()
                return

        cid = fig.canvas.mpl_connect('button_press_event', onclick2)


    def nextPickle(self, new_filename):
        print ('Keeping parameters constant...')
        self.data_filename = str(new_filename)+ '.pickle'
        print('Reading ' + self.data_filename )
        self.data = pd.read_pickle(r''+self.filepath + '/' + str(self.data_filename)) 
        datay=self.data['data'][:,0,3]#modif
        self.data_cEP=np.zeros((len(datay),2048,7))
        channels=np.arange(1,2049)
        self.x1 = 0
        self.y1 = (self.data['xend']-self.data['xstart'])
        self.xstep =self.data['xstep']
        self.step=len(self.data['data'])
        step = self.step
        self.x1*=.1
        self.y1*=.1
        self.detectors=np.arange(0,7)
        for i in range(len(self.detectors)):
            # The CaesarData class:
            class CaesarData:
            
                def __init__(self, detector="i03-C-CX1/dt/dtc-mca_xmap.1", dataname="channel03", experimentdir=""):
                    self.energies = np.zeros(2048)
                    # add some empty attributes for the caesar dataset
                    self.caesar_image = np.zeros((step,2048)) #Rajout du step
                    self.caesar_image_original =self.caesar_image * 1.
                    
                def __getitem__(self,datay,data): 
                    for ii in range (len(datay)):
                        self.caesar_image[ii, :] = data['data'][ii,:,i]
                        ii+=1
                
            
                def BuildCaesar(self,channels):
                    
                    self.energies = 0.013 + 0.04976*channels 
                    self.energies[np.nonzero(self.energies<0.0000001)]=0.0000001  
                    self.caesar_image_original = self.caesar_image * 1.
                    #print(self.energies)
            
                def correctEscapePeaks(self):
                    '''Correct escape peaks using function derived in Clark_0416
                        probably works best for amorphous things becaues it doesn't 
                        treat peak widths'''
                    imE = self.caesar_image * 1.
                    escapef = 0.41501256 * np.exp(self.energies * -0.08536899)
                    escapef = np.tile(escapef, (imE.shape[0], 1))
                    ndx = np.nonzero(self.energies>20)[0][0]
                    dE = self.energies[ndx+1] - self.energies[ndx] # channel width at 20 keV
                    dchan = -int(np.round(9.876 / dE))
                    tmp = imE*escapef # escape intensity
                    tmp2 = np.roll(tmp, dchan, 1) # observed escape peaks
                    imEcor = imE + tmp - tmp2 # add the lost intensity, remove the escape peaks
                    self.caesar_image_original = self.caesar_image * 1.
                    self.caesar_image = imEcor
                    return imEcor
            montest=CaesarData()
            montest.__getitem__(datay,self.data)
            montest.BuildCaesar(channels)
            self.data_cEP[:,:,i]=montest.correctEscapePeaks()
        # self.Chan1 = int((self.energy*1000 - self.delta_E*1000)/50)
        # self.Chan2 = int((self.energy + self.delta_E)*1000/50)
        self.d=[] #d=np.zeros(step)
        for ii, jj in enumerate((self.detectors)):
            
            if self.CHs[jj] is True:
                self.d.append((self.data_cEP[:,self.Chan1:self.Chan2,jj].mean(1))/(self.data['livetime'][:,jj]))
            else :
                ii+=1
        self.xaxis = np.linspace(self.x1,self.y1, self.step)
        self.d=np.asarray(self.d)
        self.d=self.d.transpose()
        self.d=self.d[:,:].sum(1)
        #plt.close('Data')
        plt.figure('Data')
        plt.plot(self.xaxis, self.d,'o',label = str(self.data_filename))
        plt.legend()

    def loadData_xy(self):
        self.data_filename = self.filename
        print ('Preprocess data...')
        print('Reading ' + self.data_filename)
        self.data = np.loadtxt(self.filepath + self.data_filename+ '.dat' )
        self.c = self.data[:,0]
        self.d = self.data[:,1]
        self.y1 = (self.c[-1] - self.c[0])*.1
        self.x1 = 0
        self.xstep = (self.c[1] - self.c[0])  #### need to check that
        self.step = len(self.data)
        self.xaxis = np.linspace(self.x1,self.y1, self.step)

        plt.figure('Data')
        plt.plot(self.xaxis, self.d,'o',label = str(self.data_filename))
        plt.legend()
        
    def crop(self,left = 2, right = 2):
        plt.close('Cropping')
        print ('Cropping...')
        self.left = left
        self.right = right
        if self.right == 0:
            self.d = self.d[self.left:]
            self.xaxis = self.xaxis[self.left:]
        else:
            self.d = self.d[self.left:-self.right]
            self.xaxis = self.xaxis[self.left:-self.right]             
        plt.figure('Cropping')
        plt.plot(self.xaxis,self.d,'o')
        self.step = len(self.d)
        self.x1 = self.xaxis[0]
        self.y1 = self.xaxis[-1]

    def mirror(self, reverse = False , overlap= 70):
        print('Mirroring...')
        self.overlap = overlap
        self.reverse = reverse
        plt.close('mirror')
        if reverse == True:
            self.d = flip(self.d)
        else:
            self.d
        
        fit_parameter = int(self.overlap)
        self.dif2 = []
        def opt_mirror(fit_parameter, d):
            self.d = d
            self.overlap = fit_parameter

            self.overlap = int(self.overlap)
            
            self.d2 = flip(self.d[:self.overlap+1])
            self.d_combined = np.concatenate((self.d,self.d2),axis=None)

            return self.d[-1]-self.d2[0]
        for i in range(1, len(self.d)-5):
            self.dif2.append(opt_mirror(i,self.d))
            
        self.dif2 = list(np.abs(self.dif2))
        fit_parameter = self.dif2.index(min((self.dif2)))+2
        self.dif3 = opt_mirror(fit_parameter,self.d)
        self.mfit=scipy.optimize.leastsq(opt_mirror, fit_parameter, args =(self.d))[0]

        self.d = self.d_combined
        
        plt.figure('mirror')
        self.xaxis = np.linspace(0, (len(self.d))*self.xstep*.1, len(self.d))
        plt.plot(self.xaxis,self.d,'bo')

        self.x1= self.xaxis[0]
        self.y1= self.xaxis[-1]
        self.step = len(self.xaxis)

    def correctBaseline(self, Norm_point):
        print ('Correcting baseline with linear function...')

        self.Norm_point = Norm_point
        
        I_0=np.zeros(len(self.d))
        
        for xi in range(len(self.d)):
            if xi<=self.Norm_point:
                I_0[xi]=self.d[xi]
            elif xi>=(len(self.d)-self.Norm_point):
                I_0[xi]=self.d[xi]
            else:
                I_0[xi]=NaN
        # remove nans
        c_mod = self.xaxis[np.nonzero(~np.isnan(I_0))]
        I_0 = I_0[np.nonzero(~np.isnan(I_0))]
        #estimation of linear fit
        a1=(self.d[-1]-self.d[0])/self.xaxis[-1]
        b1=self.d[0]
        linear_parameters=[a1,b1]
        def linearfit(linear_parameters,c_mod,I_0):
            a1, b1 = linear_parameters
            linearfit=a1*c_mod+b1
            diff=linearfit-I_0
            return diff
        linfit=scipy.optimize.leastsq(linearfit, linear_parameters, args =(c_mod, I_0))[0]
        diff = linearfit (linfit, self.xaxis, self.d)
        linearfit=diff+self.d
        
        plt.figure("Baseline correction")
        plt.plot(self.xaxis,linearfit,'m-', label=('linerar fit'))
        plt.plot(self.xaxis,self.d,label=('data'))
        plt.plot(c_mod,I_0,'m*', label=('Baseline')) 
        plt.xlabel=('Position (cm)')
        plt.ylabel=('Intensity profile (a.u.)')
        leg=legend(loc='lower right', frameon=True); 

        for xx in range (len(self.d)):
            self.d[xx]=self.d[xx]/linearfit[xx]
            xx+=1
        plt.figure('Data')
        plt.plot(self.xaxis,self.d*b1,'mo', markerfacecolor='none', label=('data corrected from baseline'))
        plt.legend()
     
        
    def function_fit(self, normalization = 'max', Norm_point = 0, double_capsule = False):
        """ Normalization can help you deal with I0 issue 
        'max' is default and divides by the maximum value of the profile
        'mean' makes an average of few points outside the capsule, it will ask you how many points on the left and right you want to average (~5)
        'first' or 'last' will ask you how many point on the left or right you want to include in the normalization. 
        'linear' corrects by a linear function given by the xx and yy number of points outside of the profile absorption.
        """
        # plt.close('Fitting')
        print ('Fitting...')
        self.normalization = normalization
        self.c=np.arange(0,(self.y1-self.x1),self.xstep*0.1) 
        self.c1=np.arange(0,(self.y1-self.x1),0.00005)


        ### input for Boron epoxy / hBN / graphite contributions in case of double capsule fitting ###
        self.mu_pho_env2 = 0.001
        self.r_env2 = self.c1[-1]/2
        self.Chan = int((self.energy*1000)/50)
        self.r0_env = self.y1/2
        if len(self.c)>len(self.d):
            self.c=np.arange(0,(self.y1-self.x1)-self.xstep*0.1,self.xstep*0.1)
            self.c1=np.arange(0,(self.y1-self.x1)-self.xstep*0.1,0.00005)
        if len(self.c)<len(self.d):
            self.c=np.arange(0,(self.y1-self.x1)+self.xstep*0.1,self.xstep*0.1)
            self.c1=np.arange(0,(self.y1-self.x1)+self.xstep*0.1,0.00005)
        else:
             self.c
             self.c1
        #self.d = self.d/max(self.d)
        self.l=np.zeros(len(self.c1))
        self.t=np.zeros(len(self.c1))
        self.delta=np.zeros(len(self.c1))
        
        if self.normalization == 'mean':
            if Norm_point ==0:
                Norm_point = int(input("How many points on the left and right to normalize ? "))
            else:
                Norm_point = Norm_point               
            Norm_value = (self.d[0:Norm_point].mean() + self.d[-Norm_point:].mean())/2
        elif self.normalization == 'last':
            if Norm_point ==0:
                Norm_point = int(input("How many points on the left to normalize ? "))
            else:
                Norm_point = Norm_point
            Norm_value = self.d[-Norm_point:].mean()
        elif self.normalization == 'first':
            if Norm_point ==0:
                Norm_point = int(input("How many points on the right to normalize ? "))
            else:
                Norm_point = Norm_point

            Norm_value = self.d[:Norm_point].mean()
        elif self.normalization == 'linear':
            if Norm_point ==0:
                Norm_point = int(input("How many points to fit the linear function ? "))
            else:
                Norm_point = Norm_point
            
            self.correctBaseline(Norm_point)
            Norm_value = 1
            
        else:
            Norm_value = max(self.d)   
        
        self.d = self.d/Norm_value     
        scan_abs=interpolate.interp1d(self.c, self.d, kind='cubic',bounds_error=False, fill_value="extrapolate")
        self.d1=scan_abs(self.c1)
        self.slitprofile = np.zeros(len(self.c1))
        self.slitprofile[((int(len(self.c1)/2))-50):((int(len(self.c1)/2))+50)]=1
        self.myx = np.arange(len(self.c1))
        # print(len(self.c1))
        self.sourceprofile = np.zeros(len(self.c1))
        self.sourceprofile = np.exp(-((self.myx-(len(self.c1)/2))**2)/(2*(9.5549**2)))
        self.beam=np.convolve(self.sourceprofile, self.slitprofile, mode='same')
        self.l_convolve=np.zeros(len(self.c1))
        self.t_convolve=np.zeros(len(self.c1))
        self.t_env=np.zeros(len(self.c1))
        self.t_env2=np.zeros(len(self.c1))
        self.t_sam=np.zeros(len(self.c1))
        if double_capsule is True:
            self.fit_parameters=[self.mu_pho_abs, self.mu_pho_env, self.r_sam, self.r0, self.r_env, self.mu_pho_env2, self.r_env2 ] 
            def optfit(fit_parameters, c1, d1):
                self.mu_pho_abs, self.mu_pho_env, self.r_sam, self.r0, self.r_env, self.mu_pho_env2, self.r_env2 = fit_parameters
                self.r0_env=self.r0
                
                for x in range(len(c1)):
                    
                    self.t[x]=np.abs((c1[x]-self.r0_env))
                    self.delta[x] = np.abs(c1[x] - self.r0_env)
                    if (self.t[x]<self.r_sam) & (self.delta[x]<=self.r_sam):
                        self.l[x]=2*np.sqrt(self.r_sam**2-(self.t[x])**2)
                        self.t_env[x] = 2*np.sqrt(self.r_env**2-self.delta[x]**2)-self.l[x]
                        self.t_env2[x] = 2*np.sqrt(self.r_env2**2-self.delta[x]**2)-self.t_env[x]-self.l[x]
                    elif (self.t[x]>=self.r_sam) & (self.delta[x]<=self.r_env) & (self.delta[x]>=self.r_sam):
                        self.l[x]=0
                        self.t_env[x]=2*np.sqrt(self.r_env**2-self.delta[x]**2)
                        self.t_env2[x] = 2*np.sqrt(self.r_env2**2-self.delta[x]**2)-self.t_env[x]

                    elif (self.r_env2**2-self.delta[x]**2)>=0:
                        self.l[x]=0
                        self.t_env[x]=0
                        self.t_env2[x] = 2*np.sqrt(self.r_env2**2-self.delta[x]**2)
                    else:
                        self.l[x]=0
                        self.t_env[x]=0
                        self.t_env2[x] = 0
                            
                self.delta_s=np.abs(c1 - self.r0)
                self.t_sam[np.nonzero(self.delta_s>self.r_sam)]=0
                for y in range(len(c1)):
                    self.delta_s[y] = np.abs(c1[y] - self.r0)
                    if (self.delta_s[y]<=self.r_sam):
                        self.t_sam[y]=2*np.sqrt(self.r_sam**2-(self.delta_s[y])**2)
                    else:
                        self.t_sam[y]=0
                        
                self.l_convolve=(np.convolve(self.beam/self.beam.sum(),self.t_sam, mode='same'))
                self.t_convolve=(np.convolve(self.beam/self.beam.sum(), self.t_env, mode='same'))
                self.t2_convolve=(np.convolve(self.beam/self.beam.sum(), self.t_env2, mode='same'))
                self.fit=self.cst*np.exp(-self.mu_pho_abs*self.l_convolve-self.mu_pho_env*self.t_convolve - self.mu_pho_env2*self.t2_convolve)
                self.dif=self.fit-d1
                return self.dif
        else:
            self.fit_parameters=[self.mu_pho_abs, self.mu_pho_env, self.r_sam, self.r0, self.r_env] 

            def optfit(fit_parameters, c1, d1):
                self.mu_pho_abs, self.mu_pho_env, self.r_sam, self.r0, self.r_env = fit_parameters
                self.r0_env=self.r0
                for x in range(len(c1)):
                    self.t[x]=np.abs((c1[x]-self.r0_env))
                    self.delta[x] = np.abs(c1[x] - self.r0_env)
                    if (self.t[x]<self.r_sam) & (self.delta[x]<=self.r_sam):
                        self.l[x]=2*np.sqrt(self.r_sam**2-(self.t[x])**2)
                        self.t_env[x] = 2*np.sqrt(self.r_env**2-self.delta[x]**2)-self.l[x]
                    elif (self.t[x]>self.r_sam) & (self.delta[x]<=self.r_env) & (self.delta[x]>=self.r_sam):
                        self.l[x]=0
                        self.t_env[x]=2*np.sqrt(self.r_env**2-self.delta[x]**2)
                    else:
                        self.l[x]=0
                        self.t_env[x]=0
    
                self.delta_s=np.abs(c1 - self.r0)
                self.t_sam[np.nonzero(self.delta_s>self.r_sam)]=0
                for y in range(len(c1)):
                    self.delta_s[y] = np.abs(c1[y] - self.r0)
                    if (self.delta_s[y]<=self.r_sam):
                        self.t_sam[y]=2*np.sqrt(self.r_sam**2-(self.delta_s[y])**2)
                    else:
                        self.t_sam[y]=0
                self.l_convolve=(np.convolve(self.beam/self.beam.sum(),self.t_sam, mode='same'))
                self.t_convolve=(np.convolve(self.beam/self.beam.sum(), self.t_env, mode='same'))
                self.fit=self.cst*np.exp(-self.mu_pho_abs*self.l_convolve-self.mu_pho_env*self.t_convolve)
                self.dif=self.fit-d1
                return self.dif
            
        self.dif = optfit(self.fit_parameters, self.c1, self.d1)
        self.fit=self.dif+self.d1
        self.xfit=scipy.optimize.leastsq(optfit, self.fit_parameters, args =(self.c1, self.d1))[0]
        self.dif = optfit (self.xfit, self.c1, self.d1)
        self.fit=self.dif+self.d1
        print('Standard deviation of residual is : ', '%.3f'%(100*self.dif.std()), chr(37))
        print("mu*pho(sample) = ", '%.3f'%(self.xfit[0]),"\n", "mu*pho(capsule) = " , '%.3f'%(self.xfit[1]),"\n", "sample radius(microns) = " , round(1e4*(self.xfit[2])),"\n", "capsule radius (microns) = ", round(1e4*self.r_env))
        plt.figure('Fitting')
        plt.plot(self.c,self.d,'ko')
        plt.plot(self.c1, self.fit,'r-')
        plt.plot(self.c1, self.dif, 'k-')
        plt.gca().set_xlabel('Scan position (cm)')
        plt.gca().set_ylabel('Intensity profile (a.u.)')
        plt.ylim(-0.02, None)
        
    
    def function_fit_noCapsule(self, normalization = 'max', Norm_point = 3):
        """ Normalization can help you deal with I0 issue 
        'max' is default and divides by the maximum value of the profile
        'mean' makes an average of few points outside the capsule, it will ask you how many points on the left and right you want to average (~5)
        'first' or 'last' will ask you how many point on the left or right you want to include in the normalization. 
        """
        plt.close('Fitting')
        print ('Fitting...')
        self.normalization = normalization
        self.c=np.arange(0,(self.y1-self.x1),self.xstep*0.1) 
        self.c1=np.arange(0,(self.y1-self.x1),0.00005)
        self.l=np.zeros(len(self.c1))
        self.t=np.zeros(len(self.c1))
        self.delta=np.zeros(len(self.c1))
        self.fit_parameters=[self.mu_pho_abs,self.r_sam, self.r0]  #extra important
        # fit_parameters=[mu_pho_abs,r0,r_sam,cst]  #extra important
        self.Chan = int((self.energy*1000)/50)
        
        if len(self.c)!=len(self.d):
            self.c=np.arange(0,(self.y1-self.x1)-self.xstep*0.1,self.xstep*0.1)
            self.c1=np.arange(0,(self.y1-self.x1)-self.xstep*0.1,0.00005)
        else:
             self.c
             self.c1
             
        if self.normalization == 'mean':
            if Norm_point == None:
                Norm_point = int(input("How many points on the left and right to normalize ? "))
            Norm_value = (self.d[0:Norm_point].mean() + self.d[-Norm_point:].mean())/2
        elif self.normalization == 'last':
            if Norm_point == None:
                Norm_point = int(input("How many points on the right to normalize ? "))
            Norm_value = self.d[-Norm_point:].mean()
        elif self.normalization == 'first':
            if Norm_point == None:
                Norm_point = int(input("How many points on the left and right to normalize ? "))
            Norm_value = self.d[:Norm_point].mean()
        else:
            Norm_value = max(self.d)   

        self.d = self.d/Norm_value
        scan_abs=interpolate.interp1d(self.c, self.d, kind='linear',bounds_error=False, fill_value="extrapolate")
        self.d1=scan_abs(self.c1)
        self.slitprofile = np.zeros(len(self.c1))
        self.slitprofile[((int(len(self.c1)/2))-25):((int(len(self.c1)/2))+25)]=1
        self.myx = np.arange(len(self.c1))
        # print(len(self.c1))
        self.sourceprofile = np.zeros(len(self.c1))
        self.sourceprofile = np.exp(-((self.myx-(len(self.c1)/2))**2)/(2*(9.5549**2)))
        self.beam=np.convolve(self.sourceprofile, self.slitprofile, mode='same')
        self.l_convolve=np.zeros(len(self.c1))
        self.t_convolve=np.zeros(len(self.c1))
        self.t_env=np.zeros(len(self.c1))
        self.t_sam=np.zeros(len(self.c1))
        def optfit(fit_parameters, c1, d1):
            self.mu_pho_abs, self.r_sam, self.r0= fit_parameters
            self.r0_env=self.r0
            for x in range(len(c1)):
                self.t[x]=np.abs((c1[x]-self.r0_env))
                self.delta[x] = np.abs(c1[x] - self.r0_env)
                if (self.t[x]<self.r_sam) & (self.delta[x]<=self.r_sam):
                    self.l[x]=2*np.sqrt(self.r_sam**2-(self.t[x])**2)
                    self.t_env[x] = 0
                elif (self.t[x]>self.r_sam) & (self.delta[x]<=self.r_env) & (self.delta[x]>=self.r_sam):
                    self.l[x]=0
                    self.t_env[x]=0
                else:
                    self.l[x]=0
                    self.t_env[x]=0

            self.delta_s=np.abs(c1 - self.r0)
            self.t_sam[np.nonzero(self.delta_s>self.r_sam)]=0
            for y in range(len(c1)):
                self.delta_s[y] = np.abs(c1[y] - self.r0)
                if (self.delta_s[y]<=self.r_sam):
                    self.t_sam[y]=2*np.sqrt(self.r_sam**2-(self.delta_s[y])**2)
                else:
                    self.t_sam[y]=0
                    
            self.mu_pho_env = 0
            self.l_convolve=(np.convolve(self.beam/self.beam.sum(),self.t_sam, mode='same'))
            self.fit=self.cst*np.exp(-self.mu_pho_abs*self.l_convolve)
            self.dif=self.fit-d1
            return self.dif
        self.dif = optfit(self.fit_parameters, self.c1, self.d1)
        self.fit=self.dif+self.d1
        self.xfit=scipy.optimize.leastsq(optfit, self.fit_parameters, args =(self.c1, self.d1))[0]
        self.dif = optfit (self.xfit, self.c1, self.d1)
        self.fit=self.dif+self.d1
        print('Standard deviation of residual is : ', '%.3f'%(100*self.dif.std()), chr(37))
        print("mu*pho(sample) = ", '%.3f'%(self.xfit[0]),"\n", "mu*pho(capsule) = " , "No capsule" ,"\n", "sample radius(microns) = " , '%.3f'%(self.xfit[1]*10000),"\n", "capsule radius (microns) = ", "No capsule")
        plt.figure('Fitting')
        plt.plot(self.c,self.d,'ko')
        plt.plot(self.c1, self.fit,'r-')
        plt.plot(self.c1, self.dif, 'k-')
        plt.xlabel('Scan position (cm)')
        plt.ylabel('Intensity profile (a.u.)')
        plt.ylim(-0.02, None)
    
    def function_fit_sample(self):
        #plt.close('Fitting')
        print ('Fitting sample...')
        print ('This function fits only the sample region :\n run md.crop() to remove signal from the capsule...')
        print ('The input value of mu_pho_env must be known beforehand and input in md.setParameters()...')
        

        self.c=np.arange(0,(self.y1-self.x1),self.xstep*0.1) 
        self.c1=np.arange(0,(self.y1-self.x1),0.00005)
        self.l=np.zeros(len(self.c1))
        self.t=np.zeros(len(self.c1))
        self.delta=np.zeros(len(self.c1))
        self.fit_parameters=[self.mu_pho_abs, self.r0, self.cst]  #extra important
        # fit_parameters=[mu_pho_abs,r0,r_sam,cst]  #extra important
        self.Chan = int((self.energy*1000)/50)
        
        if len(self.c)!=len(self.d):
            self.c=np.arange(0,(self.y1-self.x1)-self.xstep*0.1,self.xstep*0.1)
            self.c1=np.arange(0,(self.y1-self.x1)-self.xstep*0.1,0.00005)
        else:
             self.c
             self.c1
        self.d = self.d/max(self.d)
        self.d = self.d*(np.exp(-self.mu_pho_env*(2*self.r_env* math.sin(math.acos(self.r_sam/self.r_env)))))

        scan_abs=interpolate.interp1d(self.c, self.d, kind='linear',bounds_error=False, fill_value="extrapolate")
        self.d1=scan_abs(self.c1)
        self.slitprofile = np.zeros(len(self.c1))
        self.slitprofile[((int(len(self.c1)/2))-50):((int(len(self.c1)/2))+50)]=1
        self.myx = np.arange(len(self.c1))
        # print(len(self.c1))
        self.sourceprofile = np.zeros(len(self.c1))
        self.sourceprofile = np.exp(-((self.myx-(len(self.c1)/2))**2)/(2*(9.5549**2)))
        self.beam=np.convolve(self.sourceprofile, self.slitprofile, mode='same')
        self.l_convolve=np.zeros(len(self.c1))
        self.t_convolve=np.zeros(len(self.c1))
        self.t_env=np.zeros(len(self.c1))
        self.t_sam=np.zeros(len(self.c1))
        def optfit2(fit_parameters, c1, d1):
            self.mu_pho_abs,  self.r0, self.cst= fit_parameters
            self.r0_env=self.r0
            for x in range(len(c1)):
                self.t[x]=np.abs((c1[x]-self.r0_env))
                self.delta[x] = np.abs(c1[x] - self.r0_env)
                if (self.delta[x]<=self.r_env):
                    self.l[x]=2*np.sqrt(np.abs(self.r_sam**2-(self.t[x])**2))
                    self.t_env[x] = 2*np.sqrt(self.r_env**2-self.delta[x]**2)-self.l[x]
                else:
                    self.t_env[x]=0
                

            self.delta_s=np.abs(c1 - self.r0)
            self.t_sam[np.nonzero(self.delta_s>self.r_sam)]=0
            for y in range(len(c1)):
                self.delta_s[y] = np.abs(c1[y] - self.r0)
                if (self.delta_s[y]<=self.r_sam):
                    self.t_sam[y]=2*np.sqrt(self.r_sam**2-(self.delta_s[y])**2)
                else:
                    self.t_sam[y]=0
            
            self.l_convolve=(np.convolve(self.beam/self.beam.sum(),self.t_sam, mode='same'))
            self.t_convolve=(np.convolve(self.beam/self.beam.sum(), self.t_env, mode='same'))
            self.fit=self.cst*np.exp(-self.mu_pho_abs*self.l_convolve-self.mu_pho_env*self.t_convolve)
            self.dif=self.fit-d1
            return self.dif
        self.dif = optfit2(self.fit_parameters, self.c1, self.d1)

        self.fit=self.dif+self.d1
        self.xfit=scipy.optimize.leastsq(optfit2, self.fit_parameters, args =(self.c1, self.d1))[0]
        self.dif = optfit2 (self.xfit, self.c1, self.d1)
        self.fit=self.dif+self.d1
        print('Standard deviation of residual is : ' '%.3f'%(100*self.dif.std()), chr(37))
        print("mu*pho(sample) = ", '%.3f'%(self.xfit[0]),"\n", "mu*pho(capsule) = " , '%.3f'%(self.mu_pho_env),"\n", "sample radius(microns) = " , round(1e4*self.r_sam),"\n", "capsule radius (microns) = ", round(1e4*self.r_env))
        plt.figure('Fitting')
        plt.plot(self.c1,self.d1,'o', label = self.data_filename[7:-7])
        plt.plot(self.c1, self.fit,'r-')
        plt.plot(self.c1, self.dif, 'k-')
        plt.xlabel('Scan position (cm)')
        plt.ylabel('Intensity profile (a.u.)')
        plt.ylim(-0.02, None)
        
    def fit_trueGeometry(self, tomoFile = '',extension = '.vol', tomoFilePath = '', dimZYX =[512,2048,2048], binning = 2, threshold_S = 10, pixelSize = 1.3, height = 0.1, capsule = True):
        plt.close('Fitting')
        plt.close('Tomo')  
        self.capsule = capsule
        if self.capsule is True:
            Density.function_fit(self, normalization = 'mean', Norm_point = 1)
        else:
            Density.function_fit_noCapsule(self, normalization = 'mean', Norm_point = 1)
        self.fileNameInput = tomoFile
        self.tmp = tomoFilePath
        self.ratio = binning
        self.threshold_S = threshold_S
        self.pixelSize = pixelSize
        self.pixelSize*=self.ratio
        self.ext = extension
        self.height = height
        self.dimXOut= dimZYX[2]
        self.dimYOut= dimZYX[1]
        self.dimZOut= dimZYX[0]
        self.sliceNb = int(self.height*1000/self.pixelSize)/2

        if os.path.exists(self.tmp+self.fileNameInput+self.ext) is True:

            print ('Now processing : ', self.fileNameInput)
        
            fileInput = open(self.tmp + self.fileNameInput + self.ext, "r")
            floatNum = np.fromfile(fileInput, dtype=np.float32, count=self.dimXOut*self.dimYOut*self.dimZOut)
            floatNum = np.reshape(floatNum, (self.dimZOut, self.dimYOut, self.dimXOut))

            center_z = int(self.dimZOut/2)
            
            print("Building the average slice")
            
            # Crop the data (Z, Y X)
            floatNumCrop = floatNum[center_z-int(self.sliceNb):center_z+int(self.sliceNb), :, :]
            # Bin to Z
            floatNumBin = floatNumCrop[:,:,:].sum(0)
            dirY = np.arange(0,floatNumBin.shape[0])
            dirX = np.arange(0,floatNumBin.shape[1])
            self.thld_line = np.copy(dirX)*0+self.threshold_S
            f2, bxs = plt.subplots(2,2, figsize = (10,8), num = "Tomo")
            bxs[0,0].imshow(floatNumBin[:,:], cmap = 'gray')
            bxs[0,0].set_title('Mean slice over '+str(self.height)+' mm in height')
            bxs[0,1].plot(floatNumBin[int(self.dimYOut//2),:], 'k-', label = 'data')
            bxs[0,1].plot(dirX,self.thld_line, 'r-.', label = 'Current threshold')
            bxs[0,1].set_title('Radial profile')
            bxs[0,1].legend(loc = 'upper right')
            bxs[0,1].set_xlabel('x-axis (pixel)')
            bxs[0,1].set_ylabel('grey level (a.u.)')
            print("Starting segmentation... with sample threshold = ", str(self.threshold_S))
            floatNumSeg = np.zeros(floatNumBin.shape)


            for j,k in itertools.product(dirY, dirX):
                if floatNumBin[j,k]>self.threshold_S:
                    floatNumSeg[j,k]=1
                else:
                    floatNumSeg[j,k]=0


            print( "Building absorption profile with current threshold")

            
            
            bxs[1,0].imshow(floatNumSeg[:,:], cmap = 'gray')
            bxs[1,0].set_title('Segmentation with current threshold')
            bxs[1,1].plot(dirX*self.pixelSize/1e4,(floatNumSeg[:,:].sum(0)*self.pixelSize/1e4), 'g-')
            bxs[1,1].set_xlabel('x-axis(cm)')
            bxs[1,1].set_ylabel('sample thickness (cm)')

            self.t_sam =flip(floatNumSeg[:,:].sum(0)*self.pixelSize/1e4)
            self.axis = dirX*self.pixelSize/1e4
           
            #Estimation of r1 (this is the worst - Mickael Scott)
            tata = list(self.t_sam)
            self.r1 =self.r0 - self.axis[tata.index(np.max(tata))]           
            
            ####################################
            self.fit_parameters=[self.mu_pho_abs,self.r0]  #extra important
            ####################################            

            scan_abs=interpolate.interp1d(self.c, self.d, kind='cubic',bounds_error=False, fill_value="extrapolate")
            self.d1=scan_abs(self.c1)
            scan_abs_2=interpolate.interp1d(self.axis+self.r1, self.t_sam, kind='cubic',bounds_error=False, fill_value="extrapolate")
            self.sam_t=scan_abs_2(self.c1)

            #optfit
            def optfit(fit_parameters, c1, d1):
                self.mu_pho_abs, self.r0 = fit_parameters
                tata = list(self.t_sam)
                self.r1 =self.r0 - self.axis[tata.index(np.max(tata))]                
                self.t_sam = flip(floatNumSeg[:,:].sum(0)*self.pixelSize/1e4)
                scan_abs_2=interpolate.interp1d(self.axis+self.r1, self.t_sam, kind='cubic',bounds_error=False, fill_value="extrapolate")
                self.sam_t=scan_abs_2(self.c1)
                self.l_convolve=(np.convolve(self.beam/self.beam.sum(),self.sam_t, mode='same'))
                self.fit=self.cst*np.exp(-self.mu_pho_abs*self.l_convolve-self.mu_pho_env*self.t_convolve)
                self.dif=self.fit-d1
                return self.dif

            self.dif = optfit(self.fit_parameters, self.c1, self.d1)
            self.fit=self.dif+self.d1
            self.xfit=scipy.optimize.leastsq(optfit, self.fit_parameters, args =(self.c1, self.d1))[0]
            self.dif = optfit (self.xfit, self.c1, self.d1)
            self.fit=self.dif+self.d1
            print( 'Standard deviation of residual is : '+ '%.3f'%(100*self.dif.std()), chr(37))
            print( "mu*pho(sample) = ", '%.3f'%(self.xfit[0]),"\n", "mu*pho(capsule) = " , '%.3f'%(self.mu_pho_env),"\n", "offset r0 (cm) = " , '%.3f'%(self.r1),"\n", "capsule radius (microns) = ", round(1e4*self.r_env))
            plt.figure('Fitting')


            plt.plot(self.c1,self.d1,'ko')
            plt.plot(self.c1, self.fit,'g-')
            plt.plot(self.c1, self.dif, 'k-')
            plt.xlabel('Scan position (cm)')
            plt.ylabel('Intensity profile (a.u.)')
            plt.ylim(-0.02, None)
    
        else:
            print( "The tomography :", self.tmp+self.fileNameInput+self.ext, "is not found") 

    def batchProcessing(self, md, filelist = None, tomolist = None, firstTime = False, filetype = '.pickle'):
        plt.close('all')
        self.filetype = filetype
        self.extension = len(filetype)
        def writeList(filename, filelist):
           np.savetxt(filename, filelist, delimiter=" ", fmt = '%s')
        
        if firstTime == True:           
            print("Run once md.batchProcessing(md = md), by modifying the protocol.py file provided along this code")
            print("If you don't have a protocol.py file, please contact me laura.henry@synchrotron-soleil.fr")
            print("Then md.batchProcessing(filelist = 'filelist.txt')")
        else:
            if filelist == None:
                print("creating the filelist from all file in the current directory -- Feel free to modify it")
                filelist = glob.glob(md.filepath + '*'+self.filetype)
                filelist = [elem.split(os.sep)[-1] for elem in filelist]
                writeList('fileslist.txt', filelist)
                
            else:
                mylist  = np.loadtxt(filelist, str)
                Density = []
                Density_err = []
                Density_env = []
                SampleRadius =[]
                CapsuleRadius = []
                
                for i, filename in enumerate(mylist):
                    density, density_env, density_err, sampleRadius, capsuleRadius = protocol.protocol(filename[:-self.extension], len(mylist))
                    Density.append(density)
                    Density_env.append(density_env)
                    SampleRadius.append(sampleRadius)
                    Density_err.append(density_err)
                    CapsuleRadius.append(capsuleRadius)
                    plt.figure('Fitting')
                    plt.savefig(self.filepath+str(filename[:-self.extension])+'_fit.png')
                    plt.close('Fitting')
                    print("finished with " + filename)

                f2, bxs = plt.subplots(1,3, figsize = (12,6), num = str(filelist[:-4]))
                bxs[0].plot(Density,'bo')
                bxs[0].plot(Density,'b-')
                bxs[0].set_title('Mu*pho (sample)')
                bxs[1].plot(Density_env,'ro')
                bxs[1].plot(Density_env,'r-')
                bxs[1].set_title('Mu*pho (capsule)')
                bxs[2].plot(SampleRadius,'go')
                bxs[2].plot(SampleRadius,'g-')
                bxs[2].set_title('Sample radius (microns)')
                with open(str(filelist[:-4])+'_output.txt', 'w') as out:
                    writer = csv.writer(out, delimiter = ',')
                    writer.writerow(['filename', 'Density Sample (cm-1)', 'Density std (%)', 'Density Capsule (cm-1)', 'Sample radius (micron)', 'Capsule radius (microns)'])
                    for line in range(len(mylist)):
                        writer.writerow([mylist[line], Density[line], Density_err[line], Density_env[line], SampleRadius[line], CapsuleRadius[line]])


        
      
    def saveData(self):
        with open(self.filepath + '/' + str(self.filename)[:] +'_'+ str(round(self.energy))+'keV_data'+ '.csv', 'w') as output:
            writer = csv.writer(output, delimiter = ',')
            writer.writerow(['Raw_data_x', 'Raw_data_y'])
            for line in range(len(self.d)):
                writer.writerow([self.c[line], self.d[line]])
                
        with open(self.filepath + '/' + str(self.filename)[:] +'_'+ str(round(self.energy))+'keV_fit'+ '.csv', 'w') as output:
           writer = csv.writer(output, delimiter = ',')
           writer.writerow(['fit_x', 'fit_y', 'dif_y'])
           for line in range(len(self.fit)):
               writer.writerow([self.c1[line],self.fit[line], self.dif[line]])
        return self.data_filename, self.mu_pho_abs



