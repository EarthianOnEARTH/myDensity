# -*- coding: utf-8 -*-
'''
OK:  Make a class that handles caesar data
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# CaesarData2 - development version

# to go further - add export of ADX patterns with rebinning
# set the rebinning range - auto or specified
# clean up so that away from soleil it still works!
# try / except around pyTango bits?

# extract a gsas file - in energy or angle
# background corrections based on caesar data

# note that in Anaconda on windows, the tkFileDialog GUIs seem to work.
## So this could be the standard option.

# for the moment we read caesar data from text files extracted from nxs
# but could get data directly from nxs.
# should be able to open a single spectrum from a text file

# tool for converting all txt to gsas...  in here, or separate?

# more comments for svn...
# more comments for svn...

# clean up and make one standard version...

# THIS IS NOW THE PRODUCTION VERSION



from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import input
from builtins import range
from builtins import object
from past.utils import old_div
try:
    import PyTango
    SOLEIL = True
    # get the theta device for scanning emergencies
    DPtheta = PyTango.DeviceProxy('i03-c-c05/ex/caesar')
except:
    print("Failed to import PyTango - assume not at SOLEIL!")
    SOLEIL = False

import glob # for listing files in a directory
import os # for sep, ...
import numpy as np
import sys
import time
import pylab
pylab.ion() # switch on interactive figure mode?
import pickle # first, easy saving option
from scipy.optimize import leastsq
from scipy import interpolate
from . import fitting
# for GUI selecting files and folders
if SOLEIL:
    from . import GUI_tools
else:
    import tkinter.filedialog

# get the Spyc DeviceManager for reading values
if SOLEIL:
    try:
        from spyc.core import DeviceManager
        dm = DeviceManager()
    except:
        print("Failed to get Spyc DeviceManager - assume outside Spyc")

# Emiliano's findNextFileName
def findNextFileName(prefix,ext,file_index=1):
    #
    #Prepare correct filename to avoid overwriting
    #
    psep=prefix.rfind(os.sep)
    if(psep!=-1):
        fdir=prefix[:psep]
    else:
        fdir="."
    if(psep!=-1): prefix=prefix[psep+1:]
    if ext!="":
        fname=prefix+"_"+"%04i"%(file_index)+"."+ext
    else:
        fname=prefix+"_"+"%04i"%(file_index)
    _dir=os.listdir(fdir)
    while(fname in _dir):
        file_index+=1
        if ext!="":
            fname=prefix+"_"+"%04i"%(file_index)+"."+ext
        else:
            fname=prefix+"_"+"%04i"%(file_index)
    fname=fdir+os.sep+fname
    return fname


def simple_medfilt(im, filt, spacing=[1, 1]):
    # filt is [v,h]
    vr = old_div((filt[0]-1),2)
    vrange = list(range(-vr, vr+1, spacing[0]))
    hr = old_div((filt[1]-1),2)
    hrange = list(range(-hr, hr+1, spacing[1]))
    stack=np.zeros((im.shape[0], im.shape[1], len(vrange)*len(hrange)))
    ndx=0
    for vv in vrange:
        for hh in hrange:
            stack[:,:,ndx] = np.roll(np.roll(im, hh, 1), vv, 0)
            ndx+=1
    imfilt = np.median(stack, 2)
    return imfilt


def make_bkg(im, step=1):
    # make a 2D background from the two summed profiles
    nrows, ncols = im.shape
    bkg_e = old_div(im.sum(0),nrows)
    # smooth this out with a spline? - reducing to 20 steps
    tck = interpolate.splrep(np.arange(ncols), bkg_e, w=1./np.sqrt(bkg_e-bkg_e.min()+1), s=ncols, t=np.arange(2,ncols-2,step))
    bkg_e_s = interpolate.splev(np.arange(ncols), tck, der=0)
    # can also get a profile of scattered intensity as f(angle)
    bkg_a = old_div(im.sum(1),ncols)
    step_a = old_div(nrows,20)
    tck = interpolate.splrep(np.arange(nrows), bkg_a, w=1./np.sqrt(bkg_a-bkg_a.min()+1), s=nrows, t=np.arange(step_a,nrows-step_a,step_a))
    bkg_a_s = interpolate.splev(np.arange(nrows), tck, der=0)
    # so this is a simple guess to a 2D background
    bkg_area = np.multiply(np.reshape(bkg_e_s, (1, ncols)), np.reshape(bkg_a_s, (nrows, 1)))
    # return the 2D and 1D backgrounds
    return bkg_area


# The CaesarData class:
class CaesarData(object):

    def __init__(self, detector="i03-C-CX1/dt/dtc-mca_xmap.1", dataname="channel00", experimentdir=""):
        # set the detector device, if specified
        self.setDetector(detector, dataname)
        # set the experiment directory, if specified
        self.setExperimentdir(experimentdir)
        self.datadir = ""
        # get the Caesar device
        if SOLEIL:
            self.__APcaesar = PyTango.AttributeProxy("I03-C-C05/EX/CAESAR/theta")
            # for GUI menu things
            self.__gui = GUI_tools.GUIListBox()
        # empty attributes
        self.energy = "" # energy for caesar - can be binned
        self.energy_unbinned = "" # energy for spectrum - always 2048
        # add some empty attributes for the caesar dataset
        self.caesar_image = ""
        self.caesar_image_original = ""
        self.caesar_image_background = ""
        self.caesar_tth_nominal = ""
        self.caesar_tth = ""
        self.caesar_image_dsp = ""
        self.caesar_image_dspx = ""
        self.caesar_image_dsp_profile = ""
        self.caesar_image_dsp_profile_cor = ""
        self.caesar_image_Q = ""
        self.caesar_image_Qx = ""
        self.caesar_image_Q_profile = ""
        self.caesar_image_Q_profile_cor = ""
        self.caesar_rebin_range = "" # in keV (?)
        # scan parameters = myscan parameters, or loaded from pickle
        self.caesar_livetime = ""
        self.caesar_realtime = ""
        self.caesar_w1 = "" # usually constant
        self.caesar_w2 = "" # usually constant
        self.caesar_s1vg = "" # usually constant
        self.caesar_s1hg = "" # varied during scan
        self.caesar_cs1vg = "" # usually constant
        self.caesar_cs2vg = "" # usually constant
        self.caesar_counttime = "" # usually constant
        self.caesar_integral = "" # calculated integral for correcting afterwards
        # add some empty attributes for the spectrum
        self.spectrum=""
        self.spectrum_tth_nominal = ""
        self.spectrum_tth = ""
        self.spectrum_dspy = ""
        self.spectrum_dspx = ""
        self.spectrum_qy = ""
        self.spectrum_qx = ""
        # use the standard calibration
        self.angle_calibration= ""
        self.setAngleCalibration()
        self.energy_calibration = ""
        self.setEnergyCalibration()
        # default rebin range
        self.setRebinRange()
        self.caesar_rebin_nbins = 2048
        self.caesar_rebin_method = "rebin"
        # for plotting figures
        self.logscale=True
        self.plotdsp=True # True for dsp, False for Q
        # instrument geometry for calculating slit gaps during the scan
        self.geo_sample_diameter = ""
        self.geo_s1_distance = ""
        self.geo_R = ""
        self.geo_deltaR = ""


    ##### DRIVING FUNCTIONS - THINGS THAT THE USER CALLS #######

    def toggle_dsp_q(self):
        '''Change from Q to dsp, replot figures'''
        self.plotdsp = not self.plotdsp
        self.__showSpectrum()
        self.__showImage()

    def updateSpectrum(self):
        ''' read, convert, plot '''
        self.__getSpectrum()
        self.__convertSpectrum()
        self.__showSpectrum()

    def doTimeScan(self, npoints=5, timestep=5, scanname='timescan_'):
        '''Do a timescan of npoints of timestep seconds, and save
          data in folder with name scanname
          doTimeScan(npoints=5, timestep=5, scanname='timescan_')
        '''
        # make a new directory
        updatename = True
        count = 1
        datadir = os.path.join(self.experimentdir, scanname)
        origscanname = scanname # to start with
        while updatename:
            if not os.path.isdir(datadir):
                os.mkdir(datadir)
                updatename = False
            else:
                # folder exists already! complain!
                scanname = '%s_%d' % (origscanname, count)
                datadir = os.path.join(self.experimentdir, scanname)
                count += 1
        print("scan data will go to %s" % datadir)
        self.setDatadir(datadir)
        # call getSpectrum to make sure spectrum is up to data, and recalculate energy
        self.__getSpectrum()
        # apply the angle_calibration to the tth_angles
        self.__applyAngleCalibration()
        # make sure detector is stopped
        self.__DP.Stop()
        # do the timescan
        for ii in range(npoints):
            # Start acquiring
            self.__DP.Snap()
            # wait time
            time.sleep(timestep)
            # stop detector
            self.__DP.Stop()
            # call updateSpectrum to get and show spectrum
            self.updateSpectrum()
            # save the data
            self.exportGSAS_EDX(filename="timestep_%d" % ii, directory=datadir)
            # update stuff?
            pylab.draw()
            sys.stdout.flush()




    def doCaesarScan(self, scanname, checkpars=True, startfrompointN=0):
        '''Do a CAESAR scan
        with fancy live slit adjustments
        call calculateCaesarScan() method to prepare the scan
        doCaesarScan(scanname)
        '''
        # if required, calculate the scan trajectory - these values will be added to self
        if self.caesar_tth_nominal == "":
            print("No CAESAR scan defined. Calling calculateCaesarScan() to set parameters")
            self.calculateCaesarScan()

        if (self.caesar_image != "") and checkpars:
            print("We already have CAESAR data.  Scanning will overwrite the data in myCaesar object")
            print("Data are saved to disk at the end of each scan so nothing is lost")
            userinput = input("Proceed with scan? [y]/n : ")
            test = True if userinput.lower() in ['', 'y', 'yes', 'true'] else False
            if test == False:
                print("Aborting scan...")
                return

        if checkpars:
            # check current plan is good
            # may need to deal with single valued gaps... ie len(caesar_w1) = 1
            print("Will CAESAR scan from angle = %0.1f to angle %0.1f in steps of %0.2f" % (self.caesar_tth_nominal[0],self.caesar_tth_nominal[-1],(self.caesar_tth_nominal[1]-self.caesar_tth_nominal[0])))
            print("s1hg will move from %0.3f to %0.3f" % (self.caesar_s1hg[0], self.caesar_s1hg[-1]))
            print("cs1hg will move from %0.3f to %0.3f" % (self.caesar_w1[0], self.caesar_w1[-1]))
            print("cs2hg will move from %0.3f to %0.3f" % (self.caesar_w2[0], self.caesar_w2[-1]))

            if len(self.caesar_counttime)==1:
                print("counting time %0.1f per point" % self.caesar_counttime)
            else:
                print("initial counting time %0.1f per point" % self.caesar_counttime[0])
                print("final counting time %0.1f per point" % self.caesar_counttime[-1])

            userinput = input("Is this OK [y]/n : ")
            test = True if userinput.lower() in ['', 'y', 'yes', 'true'] else False
            if test==False:
                "Quitting - run calculateCaesarScan() to change parameters"
                return

        ## make sure that the beamstop is in!
        #get_ipython().magic(u'amove tomotz -1')
        #print "Moving in beamstop for Clark_0416 !!!"

        # vertical slits that don't move during the scan
        self.caesar_cs1vg = dm.actors['cs1vg'].value * 1. # gap cs1
        self.caesar_s1vg = dm.actors['s1vg'].value * 1. # gap cs2
        cs2vgflag=False
        if not hasattr(self, 'caesar_cs2vg'):
            self.caesar_cs2vg = dm.actors['cs2vg'].value * 1. # gap cs2
            print("constant cs2vg = %0.2f" % self.caesar_cs2vg)
        elif isinstance(self.caesar_cs2vg, (float, int)): # constant value
            print("constant cs2vg = %0.2f" % self.caesar_cs2vg)
        elif self.caesar_cs2vg == "": # as initialised, no value
            self.caesar_cs2vg = dm.actors['cs2vg'].value * 1. # gap cs2
            print("constant cs2vg = %0.2f" % self.caesar_cs2vg)
        elif len(self.caesar_cs2vg) == len(self.caesar_tth_nominal):
            print("Will open cs1vg, cs2vg from %0.2f to %0.2f" % (self.caesar_cs2vg[0], self.caesar_cs2vg[-1]))
            cs2vgflag = True
        else:
            print("DANGER - DIDNT UNDERSTAND caesar_cs2vg !!! ")

        # apply the angle_calibration to the tth_angles
        self.__applyAngleCalibration()

        # make a new directory
        datadir = os.path.join(self.experimentdir, scanname)
        if not os.path.isdir(datadir):
            os.mkdir(datadir)
        self.setDatadir(datadir)
        # call getSpectrum to make sure spectrum is up to data, and recalculate energy
        self.__getSpectrum()
        # prepare an array to put the data in
        self.caesar_image = np.zeros([len(self.caesar_tth_nominal), len(self.spectrum)])
        self.caesar_image_original = ""
        # reset the mask
        self.maskCaesar()
        # apply the angle_calibration to the tth_angles
        self.__applyAngleCalibration()
        # make sure detector is stopped
        self.__DP.Stop()
        # prepare the realtime and livetime arrays
        time.sleep(1)
        try:
            self.caesar_realtime = np.zeros(self.caesar_tth_nominal.shape)
            self.caesar_livetime = np.zeros(self.caesar_tth_nominal.shape)
        except:
            print("failed the first time, sleep 5 seconds and try again...")
            time.sleep(5)
            self.caesar_realtime = np.zeros(self.caesar_tth_nominal.shape)
            self.caesar_livetime = np.zeros(self.caesar_tth_nominal.shape)

        # make sure the preset value is greater than the longest counttime
        self.__DP.presetType = 'FIXED_REAL'
        self.__DP.presetValue = self.caesar_counttime.max()
        # perform the scan - try / except / finally to save data
        try:
            # open the shutters
            get_ipython().magic(u'feopen')
            get_ipython().magic(u'fwbsopen')
            for ii in range(startfrompointN, len(self.caesar_tth_nominal)):
                # is CAESAR happy?
                if dm.actors['theta'].state == PyTango._PyTango.DevState.FAULT:
                    print("CAESAR is not happy - try to fix by setOrigin !!")
                    DPtheta.setOrigin()
                    time.sleep(0.5)

                # move to the position
                if cs2vgflag:
                    get_ipython().magic(u'amove theta %f s1hg %f cs1hg %f cs2hg %f' % (self.caesar_tth_nominal[ii], self.caesar_s1hg[ii], self.caesar_w1[ii], self.caesar_w2[ii]))
                    get_ipython().magic(u'amove cs1vg %f cs2vg %f' % (self.caesar_cs2vg[ii], self.caesar_cs2vg[ii]))
                else:
                    get_ipython().magic(u'amove theta %f s1hg %f cs1hg %f cs2hg %f' % (self.caesar_tth_nominal[ii], self.caesar_s1hg[ii], self.caesar_w1[ii], self.caesar_w2[ii]))

                # start acquisition
                self.__DP.Snap()
                if not type(self.caesar_counttime)==np.ndarray:
                    time.sleep(self.caesar_counttime)
                else:
                    time.sleep(self.caesar_counttime[ii])
                self.__DP.Stop()
                # read the livetime and the realtime
                self.caesar_realtime[ii] = getattr(self.__DP, ('realtime%02d' % self.__datachannel))
                self.caesar_livetime[ii] = getattr(self.__DP, ('livetime%02d' % self.__datachannel))
                spectrum = getattr(self.__DP, self.__dataname)
                # add to the image
                self.caesar_image[ii, :] = spectrum
        except:
            print("Something went wrong ! Save data taken thus far anyway")
        finally:
            print("return to original s1hg, cs2hg: %f" % self.caesar_s1hg[0])
            get_ipython().magic(u'amove s1hg %f cs2hg %f' % (self.caesar_s1hg[0], self.caesar_w2[0]))
            if cs2vgflag:
                print("return cs1vg, cs2vg to original position")
                get_ipython().magic(u'amove cs1vg %f cs2vg %f' % (self.caesar_cs2vg[0], self.caesar_cs2vg[0]))
            # close the shutters
            get_ipython().magic(u'feclose')
            get_ipython().magic(u'fwbsclose')
            # save the data
            myfilename = findNextFileName(os.path.join(datadir, "caesar"), "pickle")
            # make a dictionary with everything in it:
            myscandata = {'caesar_image':self.caesar_image,
                        'caesar_tth_nominal':self.caesar_tth_nominal,
                        'caesar_tth':self.caesar_tth,
                        'energy':self.energy,
                        's1hg':self.caesar_s1hg,
                        'realtime':self.caesar_realtime,
                        'livetime':self.caesar_livetime,
                        'w1':self.caesar_w1,
                        'w2':self.caesar_w2,
                        'cs1vg':self.caesar_cs1vg,
                        'cs2vg':self.caesar_cs2vg,
                        's1vg':self.caesar_s1vg,
                        'counttime':self.caesar_counttime,
                        'sample_diameter':self.geo_sample_diameter,
                        's1_distance':self.geo_s1_distance,
                        'R':self.geo_R,
                        'deltaR':self.geo_deltaR,
                        'integral':self.caesar_integral}
            pickle.dump(myscandata, open(myfilename, 'wb'))
            print("scan data dumped in %s" % myfilename)

        # after the scan - convert and display
        self.__convertImage()
        #self.__correctCaesarSpectrum()
        self.__showImage()
        # export the summed profile
        self.__exportGSAS_combined()


    def calculateCaesarScan(self):
        '''Based on the instrument geometry, calculate the gauge volume as a function of angle'''
        # update this to prioritise opening the detector slits - i.e. use more flux, hopefully less W fluo
        # also much more effective to open the w1/w2 gaps togethercompared to what I was doing before
        # get the geometry variables
        # get the values
        # based on Alisha's previous experiment, with boost - so nominally constant integrated intensity -
        # deadtime peaks at 6 degrees, and then decays.  The boost (x2 at 20, x2 again at 25)has about the right effect - the
        # deadtime at 25 is similar to the peak
        # could have a boost factor of 4 at 28 degrees / 1 at 10 degrees / 2 at 17 degrees
        # this is a relatively straight line
        # rewrite to use self.calculateGaugeVolumes to avoid duplicating code
        # should keep the calculatio values in self.myscan_XXX rather than self.caesar_XXX ?
        # add an argument to calculateGaugeVolume to use the right one
        # check we have geometry
        if self.geo_R == "":
            print("Need to set the geometry")
            self.setGeometry()

        if self.caesar_image != "":
            print("We already have CAESAR data.  Defining a new scan will reset data in myCaesar")
            print("Data are saved to disk at the end of each scan so nothing is lost")
            userinput = input("Proceed with calculation? [y]/n : ")
            test = True if userinput.lower() in ['', 'y', 'yes', 'true'] else False
            if test == False:
                print("Aborting calculation...")
                return

        # fixed step in angle:
        userinput = input("start angle [2.5] :  ")
        startangle = 2.5 if userinput == '' else float(userinput)
        userinput = input("end angle [25] :  ")
        endangle = 25. if userinput == '' else float(userinput)
        userinput = input("Variable step size?  y/[n] :  ")
        varistep = True if userinput.lower() in ['y', 'yes', 'true'] else False
        # account for two theta angle
        if varistep:
            userinput = input("estimate useful energy minimum (keV) [25] : ")
            emin = 25. if userinput == '' else float(userinput)
            userinput = input("estimate useful energy maximum (keV) [50] : ")
            emax = 50. if userinput == '' else float(userinput)
            userinput = input("overlap in Q (percent) [75] : ")
            overlap = 75. if userinput == '' else float(userinput)
            tth_nom = [startangle]
            nextangle = 0
            k = 4*np.pi/12.398
            th0rad = np.deg2rad(old_div(startangle,2))
            while nextangle < endangle:
                Qmin0 = emin*k*np.sin(th0rad)
                Qmax0 = emax*k*np.sin(th0rad)
                Qmin1 = Qmin0 + (old_div((100-overlap),100))*(Qmax0-Qmin0)
                th0rad = np.arcsin(old_div(Qmin1,(k*emin))) # next theta in radians
                nextangle = 2*np.rad2deg(np.arcsin(old_div(Qmin1,(k*emin))))
                tth_nom.append(nextangle)
            tth_nom[-1] = endangle
            self.caesar_tth_nominal = np.array(tth_nom)
            print("will do %0.2f to %0.2f in %d steps..." % (startangle, endangle, len(tth_nom)))
        else:
            userinput = input("angle increment [0.2] :  ")
            stepsizeangle = 0.2 if userinput == '' else float(userinput)
            self.caesar_tth_nominal = np.arange(startangle, endangle+stepsizeangle, stepsizeangle)

        # detector slits
        w1_min_default=0.035
        w2_max_default=2.0
        userinput = input("Minimum value for cs1hg (w1) [suggest %0.3f]:  " % w1_min_default)
        w1_min = w1_min_default*1. if userinput == '' else float(userinput)
        userinput = input("Maximum value for cs2hg (w2) [suggest %0.3f]:  " % w2_max_default)
        w2_max = w2_max_default*1. if userinput == '' else float(userinput)
        # complementary values
        w1_max = old_div(w2_max * self.geo_R, (self.geo_R + self.geo_deltaR))
        w2_min = old_div(w1_min * (self.geo_R + self.geo_deltaR), self.geo_R)
        # boost intensity at high angles
        userinput = input("Boost the intensity at high angles by opening the detector slits? [y]/n : ")
        boost = True if userinput in ['', 'y', 'yes'] else False
        # incident slits
        userinput = input("s1hg minimum value [0.025]:  ")
        s1hg_min = 0.025 if userinput == '' else float(userinput)
        userinput = input("s1hg maximum value [0.25]:  ")
        s1hg_max = 0.25 if userinput == '' else float(userinput)
        # count time
        userinput = input("count time (s) for theta < 12 deg [25]:  ")
        counttimeA = 25. if userinput == '' else float(userinput)
        userinput = input("count time (s) for 12 < theta < 20 [50]:  ")
        counttimeB = 50. if userinput == '' else float(userinput)
        userinput = input("count time (s) for theta > 20 [100]:  ")
        counttimeC = 100. if userinput == '' else float(userinput)
        # geometry - already defined in self

        # deal with opening cs2vg for SAXS
        userinput = input("Open cs2vg progressively? y/[n] : ")
        if userinput in ['y', 'yes']:
            userinput = input("Starting value cs2vg  [0.2] : ")
            cs2vg_min = 0.2 if userinput == '' else float(userinput)
            userinput = input("Max value cs2vg  [10] : ")
            cs2vg_max = 10. if userinput == '' else float(userinput)
            azi_radius = (self.geo_R + self.geo_deltaR) * self.caesar_tth_nominal
            azi_ang = old_div(cs2vg_min,azi_radius[0])
            self.caesar_cs2vg = azi_ang * azi_radius
            self.caesar_cs2vg[np.nonzero(self.caesar_cs2vg > cs2vg_max)] = cs2vg_max
        else:
            print("Keep cs2vg constant")


        # apply the angle calibration to these nominal values
        self.__applyAngleCalibration()
        # prepare the variables for optimising - start closed
        self.caesar_s1hg = np.ones(self.caesar_tth.shape) * s1hg_min
        self.caesar_w1 = np.ones(self.caesar_tth.shape) * w1_min
        self.caesar_w2 = np.ones(self.caesar_tth.shape) * w2_min
        # times
        counttime = np.ones(self.caesar_tth.shape) * counttimeB
        counttime[np.nonzero(self.caesar_tth<12)] = counttimeA
        counttime[np.nonzero(self.caesar_tth>20)] = counttimeC
        self.caesar_counttime = counttime
        print("total counting time will be %0.1f hours" % (counttime.sum()/3600.))

        # do 5 iterations to refine slit gaps
        print("Calculate slit gaps to maintain nominally constant integrated signal")
        for jj in range(5):
            # call calculateGaugeVolumes
            self.calculateGaugeVolumes()
            # iterative correction to w1, w2, and s1hg? - this looks a good approach
            # note might be weighting more the detector slits that the incident slits with the exponents...?
            # open detector side only - change exponent from 0.33 on three slits to 0.5 on w1, w2
            self.caesar_w1 = old_div(self.caesar_w1, (self.caesar_integral**0.5)) # to seek constant integral, integral ~ w1**2
            self.caesar_w1 = old_div(self.caesar_w1 * w1_min, self.caesar_w1.min())
            self.caesar_w1[np.nonzero(self.caesar_w1 > w1_max)] = w1_max
            self.caesar_w2 = old_div(self.caesar_w1 * (self.geo_R + self.geo_deltaR), self.geo_R) # optimum relationship
            #self.caesar_s1hg = self.caesar_s1hg / (self.caesar_integral**0.33) # to seek constant integral
            #self.caesar_s1hg = self.caesar_s1hg * s1hg_min / self.caesar_s1hg.min()
            #self.caesar_s1hg[np.nonzero(self.caesar_s1hg > s1hg_max)] = s1hg_max
            print("done one iteration")

        pylab.figure(500)
        print("Clearing figure 500")
        pylab.clf()
        pylab.subplot(3,1,1)
        pylab.title('slit gaps')
        pylab.plot(self.caesar_tth, self.caesar_s1hg, '-.b', label='s1hg without boost')
        pylab.plot(self.caesar_tth, self.caesar_w1, '-.g', label='cs1hg without boost')
        pylab.plot(self.caesar_tth, self.caesar_w2, '-.r', label='cs2hg without boost')
        pylab.subplot(3,1,2)
        pylab.title('gauge length')
        pylab.plot([startangle, endangle], [self.geo_sample_diameter, self.geo_sample_diameter], '-k', label='sample diameter')
        pylab.plot(self.caesar_tth, self.caesar_gaugelen, '-.b', label='without boost')
        pylab.subplot(3,1,3)
        pylab.title('nominal intensity')
        pylab.plot(self.caesar_tth, self.caesar_integral, '-.b', label='without boost')

        # apply the boost to the detector slits
        if boost:
            print("Adding the boost at high angles")
            # this should be continuous
            # factor is (angle - 10)*0.167
            # slits must open by the square root of this
            # this also seems to keep the gauge length fairly constant after 10 degrees
            factor = 1. + ((self.caesar_tth - 10)*0.15) # first scan was 0.167
            factor[np.nonzero(factor<1)] = 1
            factor = np.sqrt(factor)
            self.caesar_w1 = self.caesar_w1 * factor
            self.caesar_w2 = self.caesar_w2 * factor
            # calculate again
            self.calculateGaugeVolumes()
            pylab.subplot(3,1,1)
            pylab.title('slit gaps')
            pylab.plot(self.caesar_tth, self.caesar_s1hg, '-b', label='s1hg with boost')
            pylab.plot(self.caesar_tth, self.caesar_w1, '-g', label='cs1hg with boost')
            pylab.plot(self.caesar_tth, self.caesar_w2, '-r', label='cs2hg with boost')
            pylab.subplot(3,1,2)
            pylab.title('gauge length')
            pylab.plot(self.caesar_tth, self.caesar_gaugelen, '-b', label='with boost')
            pylab.subplot(3,1,3)
            pylab.title('nominal intensity')
            pylab.plot(self.caesar_tth, self.caesar_integral, '-b', label='with boost')
        else:
            print("No boost at high angles")
        for ii in range(1,4):
          pylab.subplot(3,1,ii)
          pylab.legend()




######  CALIBRATION FUNCTIONS #######

    def setEnergyCalibration(self, calibration_string="0.0058079, 0.049974, 0"):
        ''' Set the energy calibration as a comma separated list
        setEnergyCalibration(calibration_string="A, B, C")
        energy(keV) = A  +  B x chan  +  C x chan**2
        defaults (0.0058079, 0.049974, 0)
        '''
        calibration_string = calibration_string.split(',')
        # convert each bit to float
        calibration_string[:] = [np.float(elem) for elem in calibration_string]
        self.energy_calibration = np.array(calibration_string)
        print("energy calibration is : %s" % self.energy_calibration)
        print("Apply calibration to current data")
        self.__applyEnergyCalibration()
        # update figures to take account of the new energies
        self.__updateFigures()

    def setAngleCalibration(self, calibration_string="0.0005, 0.0094"):
        ''' Set the Caesar angle calibration as a comma separated list
            offset = m*(nominal angle) + constant
            myCaesar.setAngleCalibration(\"m, c\")'''
        calibration_string = calibration_string.split(',')
        # convert each bit to float
        calibration_string[:] = [np.float(elem) for elem in calibration_string]
        self.angle_calibration = np.array(calibration_string)
        print("angle calibration is : %s" % self.angle_calibration)
        print("Apply caibration to current data")
        self.__applyAngleCalibration()
        # update figures to take account of the new energies
        self.__updateFigures()


    def __applyEnergyCalibration(self):
        '''Apply the current energy calibration to the current data
        simple ensures that the energy values are correct - does not convert data or show data
        '''
        # is there data?
        binning = 1 # by default
        nchan_unbinned = 2048
        if len(self.caesar_image) != 0:
            nchan = self.caesar_image.shape[1]
            # if we have binned caesar data
            binning = int(old_div(nchan_unbinned,nchan))
        # unbinned energy
        channels = np.arange(nchan_unbinned)
        self.energy_unbinned = self.energy_calibration[0] + self.energy_calibration[1]*channels + self.energy_calibration[2]*channels**2
        # to avoid divide by zero error, fake the zero/negative values in energy
        self.energy_unbinned[np.nonzero(self.energy<0.0000001)]=0.0000001
        self.energy = self.energy_unbinned.reshape(((old_div(nchan_unbinned,binning)), binning)).mean(1)


    def __applyAngleCalibration(self):
        '''Apply the current angle calibration to the current nominal angles
        simple ensures that the tth values are correct - does not convert data or show data
        '''
        if type(self.caesar_tth_nominal)==np.ndarray:
            tth_offset = (self.caesar_tth_nominal*self.angle_calibration[0]) + self.angle_calibration[1]
            self.caesar_tth = self.caesar_tth_nominal + tth_offset
        else:
            print("No caesar angles to convert")
        if type(self.spectrum_tth_nominal)==float:
            tth_offset = (self.spectrum_tth_nominal*self.angle_calibration[0]) + self.angle_calibration[1]
            self.spectrum_tth = self.spectrum_tth_nominal + tth_offset
        else:
            print("No spectrum angle to convert")

    def setGeometry(self):
        '''Set the instrument geometry for Caesar scan intensity corrections'''
        tmp = input('Enter the sample diameter in millimeters [2] : ')
        tmp = 2.0 if tmp == '' else float(tmp)
        self.geo_sample_diameter = tmp
        tmp = input('Enter the slit s1 -> sample distance in millimeters [500] : ')
        tmp = 500.0 if tmp == '' else float(tmp)
        self.geo_s1_distance = tmp
        tmp = input('Enter the sample -> detector slit 1 distance in millimeters [250] : ')
        tmp = 250.0 if tmp == '' else float(tmp)
        self.geo_R = tmp
        tmp = input('Enter the detector slit 1 -> detector slit 2 distance in millimeters [1000] : ')
        tmp = 1000.0 if tmp == '' else float(tmp)
        self.geo_deltaR = tmp

    def showGeometry(self):
        '''Display the current geometry'''
        print("#### CAESAR GEOMETRY: ####")
        print("sample diameter: %0.2f" % self.geo_sample_diameter)
        print("slit s1 -> sample: %0.2f" % self.geo_s1_distance)
        print("sample -> detector slit 1: %0.2f" % self.geo_R)
        print("detector slit 1 -> detector slit 2" % self.geo_deltaR)
        print("#### Use setGeometry to modify values ####")

    def setDetector(self, detector="", dataname=""):
        '''
            If at Soleil, set the detector
            setDetector(detector="i03-C-CX1/dt/dtc-mca_xmap.1", dataname="channel00")
        '''
        if SOLEIL and detector!="":
            self.__DP = PyTango.DeviceProxy(detector)
            self.__dataname = dataname
            self.__datachannel = int(dataname[-2:])
            print("detector is : %s, dataname is %s" % (detector, dataname))
        else:
            self.__DP = ""
            self.__dataname = ""
            self.__datachannel = ""
            print("no detector set")
        # in both cases, reinitialise the spectrum
        self.spectrum = ""


    def setExperimentdir(self, experimentdir=""):
        ''' set the data directory '''
        if experimentdir!="":
            if experimentdir[-1] == os.sep:
                experimentdir = experimentdir[0:-1] # remove a trailing /
            if os.path.isdir(experimentdir):
                self.experimentdir = experimentdir
                print("experiment directory is %s" % experimentdir)
            else:
                print("%s is not a valid directory - try again please...")
                self.experimentdir = ""
        elif not(SOLEIL):
            # not at SOLEIL, so can use GUI
            print("select experiment directory - you may have to search for the dialog window!")
            sys.stdout.flush()
            self.experimentdir = tkinter.filedialog.askdirectory()
            self.experimentdir = os.path.normpath(self.experimentdir)
            print("experiment directory is %s" % self.experimentdir)
        else:
            self.experimentdir = ""
            print("no experiment directory - set using setExperimentdir()")
        # in both cases, reinitialise the image
        self.caesar_image = ""


    def setDatadir(self, datadir=""):
        '''Set or select a data directory'''
        if datadir == "":
            if SOLEIL:
                if self.experimentdir!="":
                    # list the experiment directory
                    ldir = os.listdir(self.experimentdir)
                    dlist = [] # list of subdirectories - Caesar scans
                    for x in ldir:
                        if os.path.isdir(self.experimentdir + os.sep + x):
                            dlist.append(x)
                    dlist.sort(reverse=True)
                    # close all matplotlib images - does this help?
                    pylab.close("all")
                    # open the GUI list
                    self.__gui.buildGUI(dlist)
                    # after this is closed, get the selection
                    selection = self.__gui.get_selection()
                    self.datadir = self.experimentdir + os.sep + selection
                    print("data directory is %s" % self.datadir)
                else:
                    print("You must first set the experiment directory using setExperimentdir()")
            else:
                # not at SOLEIL - therefore can use tkFileDialog
                print("select data directory - you may have to search for the dialog window!")
                sys.stdout.flush()
                self.datadir = tkinter.filedialog.askdirectory(initialdir=self.experimentdir)
                self.datadir = os.path.normpath(self.datadir)
                psep = self.datadir.rfind(os.sep)
                # update the experiment dir as the parent dir of the datadir
                self.experimentdir = self.datadir[0:psep]
                print("data directory is %s" % self.datadir)
        else:
            if os.path.isdir(datadir):
                self.datadir = datadir
                print("data directory is %s" % self.datadir)
            else:
                print("%s is not a valid directory - try again please...")
                self.datadir = ""


    def loadBackgroundPickle(self, filename=None):
        '''Load a pickle with a background scan - empty cell or air.  Intensity will be subtracted when you load new data'''
        # remove any existing background
        self.caesar_image_background = ""
        # read new background without binning
        self.loadPickle(binning=0)
        # save this as a background
        self.caesar_image_background = self.caesar_image * 1.
        print("WARNING - it is up to the user to make sure that the background scan uses the same conditions as the data!")



    def loadPickle(self, filename=None, binning=None):
        '''Load data from a Python .pickle file (used for the home-made Caesar)'''
        if (filename!=None) and (os.path.isfile(filename)):
            print('loading %s' % filename)
            datadir = filename[0:filename.rfind(os.sep)]
            self.setDatadir(datadir)
        else:
            self.setExperimentdir(self.experimentdir)
            if self.experimentdir!="":
                # select a data directory
                self.setDatadir()
                if self.datadir!="":
                    if SOLEIL:
                        # choose a text file in this directory
                        # list the data directory
                        ldir = os.listdir(self.datadir)
                        flist = [] # list of text files
                        for x in ldir:
                            if x.endswith(".pickle"):
                                flist.append(x)
                        flist.sort(reverse=True)
                        # close all matplotlib images - does this help?
                        pylab.close("all")
                        # open the GUI list
                        self.__gui.buildGUI(flist)
                        # after this is closed, get the selection
                        selection = self.__gui.get_selection()
                        filename = self.datadir + os.sep + selection
                    else:
                        filename = tkinter.filedialog.askopenfilename(initialdir=self.datadir)
                    # now open the file
        try:
            fh = open(filename, 'rb')
            tmp = pickle.load(fh)
            fh.close()
            if type(tmp)==dict:
                print("new format pickle file containing a dictionary")
                # unpack the dictionary
                print("loading data, and updating sample geometry")
                self.caesar_image = tmp['caesar_image']
                self.caesar_tth_nominal = tmp['caesar_tth_nominal']
                self.caesar_tth = tmp['caesar_tth']
                self.energy = tmp['energy']
                self.caesar_s1hg = tmp['s1hg']
                self.caesar_realtime = tmp['realtime']
                self.caesar_livetime = tmp['livetime']
                if len([tmp['w1']])==1:
                    self.caesar_w1 = tmp['w1'] * np.ones(len(self.caesar_tth))
                else:
                    self.caesar_w1 = tmp['w1']
                if len([tmp['w2']])==1:
                    self.caesar_w2 = tmp['w2'] * np.ones(len(self.caesar_tth))
                else:
                    self.caesar_w2 = tmp['w2']
                self.caesar_cs1vg = tmp['cs1vg']
                self.caesar_cs2vg = tmp['cs2vg']
                self.caesar_s1vg = tmp['s1vg']
                self.caesar_counttime = tmp['counttime']
                self.caesar_integral = tmp['integral']
                self.geo_sample_diameter = tmp['sample_diameter']
                self.geo_s1_distance = tmp['s1_distance']
                self.geo_R = tmp['R']
                self.geo_deltaR = tmp['deltaR']
                # make an empty mask
                self.maskCaesar()
                # if it looks like this, we can pre-normalise the data
                self.prenormaliseCaesar()
            elif len(tmp)==5:
                # version 0
                [self.caesar_image, self.caesar_tth, self.energy, scanangles, scangaps] = tmp
                print("warning - this is an original pickle file, and has some missing info")
                print("trying to guess the underlying nominal caesar angles")
                start=np.round(self.caesar_tth[0])
                stop=np.round(self.caesar_tth[-1])
                steps = self.caesar_image.shape[0]
                print("looks like from %d to %d degrees in %d steps" % (start,stop,steps))
                self.caesar_tth_nominal = np.linspace(start, stop, steps)
            elif len(tmp)>=6:
                print("this looks like a new pickle file")
                # this has at least 6 entries
                [self.caesar_image, self.caesar_tth_nominal, self.caesar_tth, self.energy, self.caesar_tth_nominal, self.caesar_s1hg] = tmp[0:6]
                # but it may have some others
                if len(tmp)==10:
                    self.caesar_realtime = tmp[6]
                    self.caesar_livetime = tmp[7]
                    self.caesar_w1 = tmp[8]
                    self.caesar_w2 = tmp[9]
            else:
                print("this is another format, to be defined - not loading")
                return

            # before we bin, can subtract a background if we have one
            if len(self.caesar_image_background) != 0:
                if self.caesar_image_background.shape == self.caesar_image.shape:
                    print("Subtracting the background")
                    self.caesar_image = self.caesar_image - self.caesar_image_background
                else:
                    print("Cannot subtract background because the shape is wrong!")

            # bin noisy data?
            if binning==None:
                print("for broad peaks, can bin in energy to reduce noise")
                binning = str(input("bin data by factor X to reduce noise? X/[n or 0]  : "))
                binning = 0 if binning == "" else int(binning)

            ysize, xsize = self.caesar_image.shape
            if binning != 0:
                if np.mod(xsize, binning) == 0:
                    self.caesar_image = self.caesar_image.reshape((ysize, (old_div(xsize,binning)), binning)).mean(2)
                    self.energy = self.energy_unbinned.reshape(((old_div(xsize,binning)), binning)).mean(1)
                else:
                    print("binning %d channels by %d not possible" % (len(self.energy), binning))
            else:
                print("not binning data")
            # make an empty mask
            self.maskCaesar()
            # convert and display
            self.__convertImage()
            #self.__correctCaesarSpectrum()
            self.__showImage()
            # export the summed profile
            #self.__exportGSAS_combined()
        except:
            print("problem reading file %s" % filename)


    def prenormaliseCaesar(self):
        '''Correct a Caesar loaded from a pickle for the counting times, slit gaps etc'''
        imE = self.caesar_image
        # first, simplest - counting times - included in live time
        live = np.reshape(np.tile(self.caesar_livetime, (1, imE.shape[1])), imE.T.shape).T
        imE2 = old_div(imE, live) # counts per second
        if self.experimentdir=='/nfs/ruche-psiche/psiche-soleil/com-psiche/Clark_0416':
            # Apply a correction to cs2hg - assuming backlash
            print("Assuming backlash problem on first cs2hg slit gap - adding 0.054... ")
            ndx = np.nonzero(self.caesar_w2 == self.caesar_w2[0])
            self.caesar_w2[ndx] = self.caesar_w2[ndx] + 0.054
        else:
            print("assuming no backlash on cs2hg...")
        # Now the geometry corrections
        self.calculateGaugeVolumes()
        # normalise by the calculated integral
        integral = np.reshape(np.tile(self.caesar_integral, (1, imE.shape[1])), imE.T.shape).T
        imE3 = old_div(imE2, integral)
        # reinject this into self
        self.caesar_image_original = self.caesar_image * 1.
        self.caesar_image = imE3
        # convert and display
        self.__convertImage()
        #self.__correctCaesarSpectrum()
        self.__showImage()

    def removeLine(self, linendx):
        '''remove a bad line from a caesar dataset'''
        goodlines = np.arange(len(self.caesar_tth))
        goodlines = np.nonzero(goodlines!=linendx)[0]
        self.caesar_image = self.caesar_image[goodlines, :]
        self.caesar_tth = self.caesar_tth[goodlines]
        self.caesar_tth_nominal = self.caesar_tth_nominal[goodlines]
        print('remove line %d from the dataset' % linendx)
        # convert and display
        self.__convertImage()
        #self.__correctCaesarSpectrum()
        self.__showImage()

    def normaliseCaesar(self):
        '''Normalise a Caesar with the simplest idea of the incident beam'''
        imE = self.caesar_image
        tmp = imE.mean(0)
        tmp = np.tile(tmp, (imE.shape[0], 1))
        imEnorm = old_div(imE, tmp)
        # reinject this into self
        self.caesar_image_original = self.caesar_image * 1.
        self.caesar_image = imEnorm
        # convert and display
        self.__convertImage()
        #self.__correctCaesarSpectrum()
        self.__showImage()

    def correctEscapePeaks(self):
        '''Correct escape peaks using function derived in Clark_0416
            probably works best for amorphous things becaues it doesn't
            treat peak widths'''
        imE = self.caesar_image * 1.
        escapef = 0.41501256 * np.exp(self.energy * -0.08536899)
        escapef = np.tile(escapef, (imE.shape[0], 1))
        ndx = np.nonzero(self.energy>20)[0][0]
        dE = self.energy[ndx+1] - self.energy[ndx] # channel width at 20 keV
        dchan = -int(np.round(9.876 / dE))
        tmp = imE*escapef # escape intensity
        tmp2 = np.roll(tmp, dchan, 1) # observed escape peaks
        imEcor = imE + tmp - tmp2 # add the lost intensity, remove the escape peaks
        # reinject this into self
        self.caesar_image_original = self.caesar_image * 1.
        self.caesar_image = imEcor
        # convert and display
        self.__convertImage()
        #self.__correctCaesarSpectrum()
        self.__showImage()


    def calculateGaugeVolumes(self):
        '''Given the slit gaps in self.caesar_X, and the sample geo in self.geo_X, calculate volumes'''
        # get the geometry
        R = self.geo_R
        deltaR = self.geo_deltaR
        sample_d = self.geo_sample_diameter
        s1_distance = self.geo_s1_distance
        source_sigma = old_div((1/2.355) * (s1_distance * 0.900), (23500 - s1_distance))
        scanangles = self.caesar_tth
        tth = old_div(scanangles * np.pi, 180)
        # caesar slits
        s1hg = self.caesar_s1hg
        w1 = self.caesar_w1 * np.ones(scanangles.shape)
        # are we varying w2 ?
        if len(self.caesar_w2)==1:
            w2 = self.caesar_w2 * np.ones(scanangles.shape)
        else:
            w2 = self.caesar_w2
        # detector delta theta - which slit is limiting factor?
        dtheta = np.min([old_div(w1,R), old_div(w2,(R+deltaR))], 0)
        dtheta = old_div(dtheta, dtheta.min())
        # different contributions to the gauge
        A = old_div(w1 * (R + deltaR), deltaR) # projection caesar slit 1
        B = old_div(w2 * R, deltaR) # projection caesar slit 2
        # effect of the angle
        AA = old_div(A, np.sin(tth))
        BB = old_div(B, np.sin(tth))
        CC = old_div(s1hg, np.tan(tth))
        DD = old_div(source_sigma, np.tan(tth))
        # calculate the gauge
        spos = np.linspace(-sample_d, sample_d, 2000*sample_d) # twice the sample diameter in micron steps
        # sample limits for integrating
        ndxS1 = np.nonzero(spos>(-sample_d/2.))[0][0]
        ndxS2 = np.nonzero(spos<(sample_d/2.))[0][-1]
        integral = np.zeros(len(tth))
        gaugelen = np.zeros(len(tth))
        # work through all angles
        for ii in range(len(tth)):
            # for each angle, calculate the gauge
            pA = np.zeros(spos.shape)
            ndxA1 = np.nonzero(spos>(-AA[ii]/2.))[0][0]
            ndxA2 = np.nonzero(spos<(AA[ii]/2.))[0][-1]
            pA[ndxA1:ndxA2] = 1
            pB = np.zeros(spos.shape)
            ndxB1 = np.nonzero(spos>(-BB[ii]/2.))[0][0]
            ndxB2 = np.nonzero(spos<(BB[ii]/2.))[0][-1]
            pB[ndxB1:ndxB2] = 1
            pC = np.zeros(spos.shape)
            ndxC1 = np.nonzero(spos>(-CC[ii]/2.))[0][0]
            ndxC2 = np.nonzero(spos<(CC[ii]/2.))[0][-1]
            pC[ndxC1:ndxC2] = 1
            # gaussian source size
            pD = np.zeros(spos.shape)
            pD = np.exp(old_div(-(spos**2), (2 * (DD[ii]**2))))
            # detector side - normalise so max = dtheta
            pAB = np.convolve(pA, pB, 'same')
            pAB = old_div(pAB, pAB.max())
            pAB = pAB * dtheta[ii]
            # incident side - normalise so sum = s1hg
            pCD = np.convolve(pC, pD, 'same')
            pCD = old_div(pCD * s1hg[ii], pCD.sum())
            # total gauge length
            pABCD = np.convolve(pAB, pCD, 'same')
            # integrate over the sample diameter
            integral[ii] = pABCD[ndxS1:ndxS2].sum()
            tmp = pABCD.max()/50.
            gaugelen[ii] = (np.nonzero(pABCD>tmp)[0][-1] - np.nonzero(pABCD>tmp)[0][0]) * 0.001

        # add these to self...  may already have integral but this might be improved
        self.caesar_gaugelen = gaugelen
        self.caesar_integral = integral
        return self


    def loadCaesar(self):
        '''select a data directory, load, convert and show'''
        self.setExperimentdir(self.experimentdir)
        self.setDatadir()
        if self.datadir != "":
            # if at Soleil, redraw the figures which are closed for the GUI
            if SOLEIL:
                self.__showSpectrum()
            print("loading and processing scan")
            self.__getImage()
            self.__convertImage()
            #self.__correctCaesarSpectrum()
            self.__showImage()
            # export the summed profile
            self.__exportGSAS_combined()
        else:
            print("no experiment directory set - use setExperimentdir()")



    def __getSpectrum(self):
        ''' get the latest spectrum from the detector '''
        if self.__dataname!="" and self.__DP!="":
            print("read the current spectrum, correct Caesar angle")
            self.spectrum = getattr(self.__DP, self.__dataname)
            tth = self.__APcaesar.read().value
            self.spectrum_tth_nominal = tth
            # if the spectrum doesn't have the same size as the caesar image, wipe the caesar
            if len(self.caesar_image) != 0:
                ncaesar = self.caesar_image.shape[1]
                nchan = len(self.spectrum)
                if ncaesar != nchan:
                    print("New spectrum has %d channels, Caesar has %d. Reset Caesar to avoid problems" % (ncaesar, nchan))
                    self.caesar_image = ""
                    self.caesar_image_original = ""
            # calculate the true angle
            self.__applyAngleCalibration()
            # recalculate the channel energies
            self.__applyEnergyCalibration()
        else:
            print("No detector / data defined !")
            self.spectrum = ""
            self.spectrum_tth = ""
        # reset the converted spectrum
        self.spectrum_dspy = ""
        self.spectrum_dspx = ""

    def loadSpectrum(self):
        '''open a spectrum from a txt file in data directory'''
        if SOLEIL:
            if self.experimentdir!="":
                # select a data directory
                self.setDatadir()
                if self.datadir!="":
                    # choose a text file in this directory
                    # list the data directory
                    ldir = os.listdir(self.datadir)
                    flist = [] # list of text files
                    for x in ldir:
                        if x.endswith(".txt"):
                            flist.append(x)
                    flist.sort(reverse=True)
                    # close all matplotlib images - does this help?
                    pylab.close("all")
                    # open the GUI list
                    self.__gui.buildGUI(flist)
                    # after this is closed, get the selection
                    selection = self.__gui.get_selection()
                    filename = self.datadir + os.sep + selection
            else:
                self.datadir = ""
                print("no experiment directory set - use setExperimentdir()")

        else:
            # not at SOLEIL - therefore can use tkFileDialog
            print("select spectrum .txt datafile - you may have to search for the dialog window!")
            sys.stdout.flush()
            filename = tkinter.filedialog.askopenfilename(initialdir=self.experimentdir, filetypes=[("text files", ".txt")])
            filename = os.path.normpath(filename)
        try:
            # Now read the spectrum - same at SOLEILor at home
            self.spectrum = np.loadtxt(filename)
            # read the angle
            tthvalue = filename[-10:-4]
            if tthvalue[0]=='_':
                tthvalue = tthvalue[1::]
            # this is the nominal value - convert it
            self.spectrum_tth_nominal = float(tthvalue)
            self.__applyAngleCalibration()
            self.__applyEnergyCalibration()
            self.__convertSpectrum()
            self.__showSpectrum()
        except:
            print("Failed to read %s" % filename)


    def __getImage(self):
        ''' read in an image from the Caesar acquistion text files'''
        if self.datadir!="":
            # get the name of the scan:
            parts = self.datadir.split(os.sep)
            # strip out the zero length parts
            parts[:] = [elem for elem in parts if len(elem)!=0]
            dataname = parts[-1]
            # get the length of the name
            namelength = len(dataname)

            # get a list of all the text files
            filelist = glob.glob(self.datadir+os.sep+dataname+"*txt")
            # need to go through this list, understanding the filenames
            fnames = [elem.split(os.sep)[-1] for elem in filelist]
            # filename format: dirname/dirname_%04d-[_%0.3f.txt / %0.3.txt]
            # values are the step and the actuator position
            #print filelist[0]
            # read a first spectrum
            a = np.loadtxt(filelist[0])
            # define the image
            image = np.zeros([len(fnames), len(a)])
            # will record the nominal tth angles
            tth_nom = np.zeros(len(fnames))

            # read all the steps
            step = np.zeros(len(fnames))
            sys.stdout.write("reading files")
            for ii in range(len(fnames)):
                # which step is this?
                step[ii] = int(fnames[ii][(namelength+1):(namelength+5)])
                # read the spectrum
                image[step[ii], :] = np.loadtxt(filelist[ii])
                # read the angle from the filename
                tthvalue = fnames[ii][-10:-4]
                if tthvalue[0]=='_':
                    tthvalue = tthvalue[1::]
                tth_nom[step[ii]] = float(tthvalue)
                # display something
                sys.stdout.write(".")
                sys.stdout.flush()
                if np.mod(ii, 5)==4:
                    sys.stdout.write("\b\b\b\b\b")
            sys.stdout.write("\nfinished reading\n")
            # return the image
            self.caesar_tth_nominal = tth_nom
            self.caesar_image = image
            self.caesar_image_original = ""
            # apply the angle_calibration to the tth_angles
            self.__applyAngleCalibration()
            # make sure the energy calibration is up to date
            self.__applyEnergyCalibration()
            # avoid strange image behaviour in timescan case (all tth values the same)
            if self.caesar_tth[-1] == self.caesar_tth[0]:
                print("### TWEAK CONSTANT ANGLES TO DISPLAY DATA CORRECTLY ###")
                self.caesar_tth[-1] = self.caesar_tth[-1] + 0.00001
            # apply the gauge volume correction
            print("If sample is larger than gauge volume, should correct for gauge volume as f(angle)")
            check = str(input("Correct for gauge volume? y/[n]  : "))
            if check == 'y' or check == 'yes':
                self.__gaugeVolumeCorrection()
            else:
                print("not correcting gauge volume")
            # bin noisy data?
            print("for broad peaks, can bin in energy to reduce noise")
            check = str(input("bin data x2 to reduce noise? y/[n]  : "))
            if check == 'y' or check == 'yes':
                self.caesar_image = self.caesar_image[:, 0::2] + self.caesar_image[:, 1::2]
                self.energy = old_div((self.energy[0::2]+self.energy[1::2]),2)
            else:
                print("not binning data")
            print("Can try to correct the background -- EXPERIMENTAL!!! --")
            check = str(input("correct background? y/[n]  : "))
            if check == 'y' or check == 'yes':
                self.__backgroundCorrection()
            else:
                print("not correcting background - you can use the correctBackground() method later")

        else:
            print("No data directory defined !")
            self.caesar_image=""
            self.caesar_tth=""

##### DATA PROCESSING FUNCTIONS #########

    def __gaugeVolumeCorrection(self):
        '''correct for gauge volume proportional to 1/sin(tth)'''
        for ii in range(len(self.caesar_tth)):
            self.caesar_image[ii, :] = self.caesar_image[ii, :] * np.sin(old_div(self.caesar_tth[ii]*np.pi,180))
        print("Applied sin(two theta) correction for gauge volume as f(angle)")

    def correctBackground(self, filtsize=[3,25], step=100, negweight=3):
        '''Try correcting the background of a Caesar acquistion'''
        if self.caesar_image != "":
            print("correcting the current Caesar data")
            self.__backgroundCorrection(filtsize, step, negweight)
            self.__convertImage()
            #self.__correctCaesarSpectrum()
            self.__showImage()
        else:
            print("Load Caesar data first!")

    def undoCorrection(self):
        '''Revert to the previous veresion of the caesar image'''
        if self.caesar_image_original != "":
            print("Undo the background correction")
            self.caesar_image = self.caesar_image_original
            self.__convertImage()
            self.__showImage()
        else:
            print("No original data found!")

    def __backgroundCorrection(self, filtsize=[3,25], step=100, negweight=3):
        '''A background correction that make an approximate background using profiles from
        summing the caesar image in the two directions with smoothing.  The approximate background
        is then fitted to the each energy profile, with a term that allows some distortion of the shape.
        The least squares fitting is modifiedto allow peaks above the data, but to penalise negative values
        (i.e. background value > data)
        The fitting results are smoothed, and used to produce a background function which is subtracted.
        Arguments are the median filter size for smoothing, and a step size (in detector channels)
        to smooth out stripes if diffraction peaks move by a step greater than the peak width,
        and the extra weight given to negative values when fitting the background
        It seems to work pretty well with the defaults'''

        # what ratio to use?
        im = self.caesar_image*1.
        nrows, ncols = self.caesar_image.shape

        # filter to reduce noise - removes spikes from liquid data - removes peaks?
        # in case the filter is large, we can use a step size > 1
        filtstep = np.ceil(np.array(filtsize)/10.)
        filtstep = [int(ff) for ff in filtstep]
        im2 = simple_medfilt(im, filtsize, filtstep)
        bkg_area = make_bkg(im2, step)

        # the bkg_area is a good guess...
        # can it be tweaked to fit better?
        out = np.zeros((nrows, 2))
        bkg_improved = np.zeros(bkg_area.shape)
        for ii in range(nrows):
            prof = im2[ii, :]
            bkg = bkg_area[ii, :]

            def optfunc(x):
                # function for fitting, including the extra weight for negative values
                bkg_mod = bkg*(x[0] + np.arange(ncols)*x[1])
                # dif = background - data
                dif = prof - bkg_mod
                # modify the differences - want a large penalty for negatives, and small (-> 0) for large values
                pos = np.nonzero(dif>1)
                dif[pos] = 1 + np.log(dif[pos])
                neg = np.nonzero(dif<0)
                dif[neg] = dif[neg]*negweight
                return dif

            xfit = leastsq(optfunc, [1, 0])[0]
            out[ii, :] = xfit
        # smooth the fit results
        out_s = np.zeros(out.shape)
        tck = interpolate.splrep(np.arange(nrows), out[:,0], s=nrows)
        out_s[:,0] = interpolate.splev(np.arange(nrows), tck, der=0)
        tck = interpolate.splrep(np.arange(nrows), out[:,1], s=nrows)
        out_s[:,1] = interpolate.splev(np.arange(nrows), tck, der=0)
        # build the final background
        for ii in range(nrows):
            xfit = out_s[ii, :]
            bkg_improved[ii, :] = bkg_area[ii, :]*(xfit[0] + np.arange(ncols)*xfit[1])
        # apply this to the original, unsmoothed image
        imnew = im-bkg_improved
        # get rid of stupid tiny values - does silly things to log()
        imnew[np.nonzero(abs(imnew)<0.0000001)]=0
        # return
        self.background = bkg
        self.caesar_image_original = self.caesar_image
        self.caesar_image = imnew

    def amorphousDataTreatment(self):
        '''Try applying Alisha data treatment Caesar acquistion'''
        if self.caesar_image != "":
            print("correcting the current Caesar data")
            self.__amorphousTreatment()
            self.__showImage()
        else:
            print("Load Caesar data first!")

    def maskCaesar(self, mask=None):
        '''Mask out a region
        If not argument given, reset the mask
        Otherwise, add a region to mask, defined as [tth start, tth end, energy start, energy end] '''
        print(mask)
        if mask == None:
            print("reset the mask")
            self.maskE = np.ones(self.caesar_image.shape)
            #tmp = raw_input('reset the current mask? [y]/n : ')
            #tmp = 'y' if tmp == '' else tmp
            #if tmp =='y':
            #    print "reset the mask"
            #    self.maskE = np.ones(self.caesar_image.shape)
            #else:
            #    print "Not changing mask"
        else:
            tth_ndx_start = np.nonzero(self.caesar_tth>mask[0])[0][0]
            tth_ndx_end = np.nonzero(self.caesar_tth<mask[1])[0][-1]+1
            energy_ndx_start = np.nonzero(self.energy>mask[2])[0][0]
            energy_ndx_end = np.nonzero(self.energy<mask[3])[0][-1]+1
            self.maskE[tth_ndx_start:tth_ndx_end, energy_ndx_start:energy_ndx_end] = 0

    def __amorphousTreatment(self, lowE=None, highE=None, weighting=None, smooth=None):
        '''Alisha data treatment to produce a spectrum'''
        # work on current caesar scan
        imE = self.caesar_image * 1.0
        tth = self.caesar_tth * 1.0
        energy = self.energy * 1.0
        # work on prenormalised data
        print("Fit using a reduced energy range - ")
        if lowE == None:
            lowE = input("Starting energy [default 20] :")
            lowE = 20.0 if lowE == '' else float(lowE)
        ndx1 = np.nonzero(energy>lowE)[0][0]
        if highE == None:
            highE = input("Final energy [default 60] :")
            highE = 60.0 if highE == '' else float(highE)
        ndx2 = np.nonzero(energy<highE)[0][-1]
        if weighting == None:
            check = input("In fitting, weight difference by 1 / mean(intensity) ? [y]/n :")
            check = 'y' if check == '' else str(check)
            weighting=True if check == 'y' else False
        if smooth == None:
            print("Selected range is %d channels of the energy spectrum" % (ndx2-ndx1))
            check = input("Fit with smoothing (60 points) [y - smoothing]/n - all channels independant :")
            smooth = 'y' if check == '' else str(check)
        # guess the spectrum?
        imEsm = simple_medfilt(imE, [3,3])
        tmp = imEsm.sum(0)
        if smooth=='y':
            xpoints = np.int16(np.linspace(ndx1, ndx2, 60)) # hard code 60 for the moment
        else:
            xpoints = np.arange(ndx1, ndx2)
        x0 = tmp[xpoints]
        x0 = old_div(x0, x0.max())
        # apply the mask before fitting
        imE[np.nonzero(self.maskE==0)]=np.nan
        imEsm[np.nonzero(self.maskE==0)]=np.nan
        # fitting step - fit using smoothed imE
        print("fitting the effective beam spectrum")
        xfit = leastsq(optfunc, x0, args=(imEsm, tth, energy, ndx1, ndx2, weighting, smooth))[0]
        xfit = old_div(xfit, xfit.max())
        pylab.figure(103)
        pylab.plot(energy[xpoints], x0, label = 'first guess')
        pylab.plot(energy[xpoints], xfit, label = 'refined...')
        # use these coeffs to calculate the corrected image and correction array
        # use the unsmooth image here
        dif,imEnorm,imQ,Qgrid,spectrum = optfunc(xfit, imE, tth, energy, ndx1, ndx2, weighting, smooth, for_optimisation=False)
        dif,imEnormsm,imQsm,Qgrid,spectrum = optfunc(xfit, imEsm, tth, energy, ndx1, ndx2, weighting, smooth, for_optimisation=False)
        self.fitted_spectrum = spectrum * 1.0
        self.fitted_spectrum_ndx = [ndx1, ndx2]
        # extract the profile from the Qimage - this now contains Nans
        sumQ = np.nansum(imQ, 0) # sum of not nans
        countQ = (np.logical_not(np.isnan(imQ))).sum(0) # number of not nans
        Qprofile = old_div(sumQ, countQ)
        #Qprofile = Qprofile / Qprofile.max()
        sumQsm = np.nansum(imQsm, 0) # sum of not nans
        countQsm = (np.logical_not(np.isnan(imQsm))).sum(0) # number of not nans
        Qprofilesm = old_div(sumQsm, countQsm)
        #Qprofilesm = Qprofilesm / Qprofilesm.max()
        pylab.figure()
        pylab.subplot(2,1,1)
        my_plot(imQ, Qgrid, tth)
        pylab.plot(Qgrid, Qprofile, 'k')
        pylab.subplot(2,1,2)
        my_plot(imQ, Qgrid, tth, 'c')
        pylab.plot(Qgrid, Qprofile, 'k')
        pylab.plot(Qgrid, Qprofilesm, 'r')
        # inject back into the figure
        self.caesar_image_original = self.caesar_image * 1.
        imnew = np.zeros(self.caesar_image.shape)
        imnew[:, ndx1:ndx2] = imEnorm
        self.caesar_image = imnew
        #self.__convertImage()
        # add our profile
        self.caesar_image_Q = imQ
        self.caesar_image_Qx = Qgrid
        self.caesar_image_Q_profile = Qprofile
        self.caesar_image_Q_profile_cor = Qprofilesm # use for smoothed...
        self.__showImage()

    def applyFittedSpectrum(self):
        '''This does not fit the spectrum - it takes the existing fitted spectrum and applies it to the current dataset'''
        # after fitting a spectrum, apply it to a new, different dataset
        imE = self.caesar_image*1.
        tth = self.caesar_tth*1.
        energy = self.energy*1.
        imEsm = simple_medfilt(imE, [3,3])
        spectrum = old_div(self.fitted_spectrum, self.fitted_spectrum.max())
        spectrum2D = np.tile(spectrum, (imE.shape[0], 1))
        # apply the mask
        imE[np.nonzero(self.maskE==0)]=np.nan
        imEsm[np.nonzero(self.maskE==0)]=np.nan
        # normalise using this
        imEnorm = old_div(imE[:, self.fitted_spectrum_ndx[0]:self.fitted_spectrum_ndx[1]], spectrum2D)
        imEnorm = old_div(imEnorm, imEnorm.max())
        imEnormsm = old_div(imEsm[:, self.fitted_spectrum_ndx[0]:self.fitted_spectrum_ndx[1]], spectrum2D)
        imEnormsm = old_div(imEnormsm, imEnormsm.max())
        # convert image to Q
        imQ, Qgrid = myConvertImage(imEnorm, tth, energy[self.fitted_spectrum_ndx[0]:self.fitted_spectrum_ndx[1]])
        imQsm, Qgrid = myConvertImage(imEnormsm, tth, energy[self.fitted_spectrum_ndx[0]:self.fitted_spectrum_ndx[1]])
        # extract the profile from the Qimage - this now contains Nans
        sumQ = np.nansum(imQ, 0) # sum of not nans
        countQ = (np.logical_not(np.isnan(imQ))).sum(0) # number of not nans
        Qprofile = old_div(sumQ, countQ)
        #Qprofile = Qprofile / Qprofile.max()
        sumQsm = np.nansum(imQsm, 0) # sum of not nans
        countQsm = (np.logical_not(np.isnan(imQsm))).sum(0) # number of not nans
        Qprofilesm = old_div(sumQsm, countQsm)
        #Qprofilesm = Qprofilesm / Qprofilesm.max()
        pylab.figure()
        pylab.subplot(2,1,1)
        my_plot(imQ, Qgrid, tth)
        pylab.plot(Qgrid, Qprofile, 'k')
        pylab.subplot(2,1,2)
        my_plot(imQ, Qgrid, tth, 'c')
        pylab.plot(Qgrid, Qprofile, 'k')
        pylab.plot(Qgrid, Qprofilesm, 'r')
        # inject back into the figure
        self.caesar_image_original = self.caesar_image * 1.
        imnew = np.zeros(self.caesar_image.shape)
        imnew[:, self.fitted_spectrum_ndx[0]:self.fitted_spectrum_ndx[1]] = imEnorm
        self.caesar_image = imnew
        #self.__convertImage()
        # add our profile
        self.caesar_image_Q = imQ
        self.caesar_image_Qx = Qgrid
        self.caesar_image_Q_profile = Qprofile
        self.caesar_image_Q_profile_cor = Qprofilesm # use for smoothed...
        self.__showImage()


    def removeSpikes(self, filt=[7,7]):
        '''Remove spikes from the caesar image'''
        imE = self.caesar_image * 1.0
        imE = old_div(imE, imE.max())
        imEsm = simple_medfilt(imE, filt)
        tmp = old_div((imE - imEsm),imEsm)
        mask = tmp > 0.5
        imE[np.nonzero(mask)] = imEsm[np.nonzero(mask)]
        self.caesar_image_original = self.caesar_image * 1.
        self.caesar_image = imE * 1.
        self.__convertImage()
        self.__showImage()

    def removeSpikes2(self):
        '''Remove spikes from the caesar image'''
        # this is good at identifying the spikes, but the correction is weak
        imE = self.caesar_image * 1.0
        imEsm = simple_medfilt(imE, [7,1]) # compare to other angles
        tmp = np.min(np.dstack((imE, imEsm)), 2) # remove peaks, but don't fill holes?
        # this might be better in some respects - preserves local minima.
        # better would be to have smooth transitions between the images
        imE = tmp # simply replace the image by the smoothed image
        self.caesar_image_original = self.caesar_image * 1.
        self.caesar_image = imE * 1.
        self.__convertImage()
        self.__showImage()

    def saveCurrentSpectrum(self, myname=None):
        # need to export this
        if (self.caesar_image_Q_profile=="") or (self.caesar_image_Q_profile_cor==""):
            print("run amorphousDataTreatment() first")
        if myname==None:
            myname = self.datadir.split(os.sep)[-1]
        # get the next file name
        myfilename = findNextFileName(self.datadir+os.sep+myname, "txt")
        #myfilename = myname + '.txt'
        # open the file
        myfile = open(myfilename, 'wt')
        # do not write a header of PowderCell
        myfile.write("Q, I\n")
        for ii in range(len(self.caesar_image_Qx)):
            myfile.write("%f    %f    %f\n" % (self.caesar_image_Qx[ii], self.caesar_image_Q_profile[ii], self.caesar_image_Q_profile_cor[ii]))
        myfile.close()
        print("txt file: \n  %s\nwritten" % myfilename)


    def __convertSpectrum(self):
        ''' convert data from energy to dspacing and Q'''
        # convert the spectrum, if available
        if len(self.spectrum) != 0:
            # current theta angle, radians
            theta = old_div((old_div(self.spectrum_tth,2))*np.pi,180)
            # current data in dsp and Q
            dsp = 12.398 / (2*np.sin(theta)*self.energy_unbinned)
            Q = old_div(2*np.pi,dsp)
            # limits for interpolation
            dmax = 12.398 / (2*np.sin(theta)*10) # current angle, lowest energy
            dmin = 12.398 / (2*np.sin(theta)*self.energy_unbinned[-1]) # current angle, highest energy
            dspgrid = np.linspace(dmin, dmax, 4096)
            Qmin = old_div(2*np.pi,dmax)
            Qmax = old_div(2*np.pi,dmin)
            Qgrid = np.linspace(Qmin, Qmax, 4096)
            # interpolate - invert dsp and data for ascending order
            self.spectrum_dspy = np.interp(dspgrid, dsp[::-1], self.spectrum[::-1], 0, 0)
            self.spectrum_dspx = dspgrid
            self.spectrum_qy = np.interp(Qgrid, Q, self.spectrum, 0, 0)
            self.spectrum_qx = Qgrid
        else:
            self.spectrum_dspy = ""
            self.spectrum_dspx = ""
            self.spectrum_qy = ""
            self.spectrum_qx = ""

    def __convertImage(self):
        ''' convert data from energy to dspacing and Q '''
        # convert the spectrum, if available
        if len(self.caesar_image) != 0:
                # determine the theta values
            theta = old_div(self.caesar_tth,2)
            # preallocate an output image
            image2 = np.zeros([self.caesar_image.shape[0], 4096])
            image3 = np.zeros([self.caesar_image.shape[0], 4096])
            # dspacing limits for the whole acquistion
            dmax = 12.398 / (2*np.sin(old_div(theta[0]*np.pi,180))*10) # lowest angle, lowest energy
            dmin = 12.398 / (2*np.sin(old_div(theta[-1]*np.pi,180))*self.energy[-1]) # highest angle, highest energy
            dspgrid = np.linspace(dmin, dmax, 4096)
            Qmin = old_div(2*np.pi,dmax)
            Qmax = old_div(2*np.pi,dmin)
            Qgrid = np.linspace(Qmin, Qmax, 4096)
            # convert line by line
            for ii in range(self.caesar_image.shape[0]):
                # energy profile
                prof = self.caesar_image[ii, :]
                prof = prof[::-1]
                # as d spacing
                dsp = 12.398 / (2*np.sin(old_div(theta[ii]*np.pi,180))*self.energy)
                dsp = dsp[::-1] # increasing
                image2[ii, :] = np.interp(dspgrid, dsp, prof, 0, 0)
                # as Q - flip profiles for increasing
                Q = old_div(2*np.pi,dsp)
                image3[ii, :] = np.interp(Qgrid, Q[::-1], prof[::-1], 0, 0)
            # build the (naive) combined dsp profile - only do this if we haven't done something else
            dspprofile = image2.sum(0)
            dspprofile_n = (image2!=0).sum(0)
            dspprofile = old_div(dspprofile,dspprofile_n)
            dspprofile[np.nonzero(np.isnan(dspprofile))] = 0 # remove nans
            print("correcting combined profile for number of datapoints")
            # build the (naive) combined Q profile
            Qprofile = image3.sum(0)
            Qprofile_n = (image3!=0).sum(0)
            Qprofile = old_div(Qprofile,Qprofile_n)
            Qprofile[np.nonzero(np.isnan(Qprofile))] = 0 # remove nans

            self.caesar_image_dsp = image2
            self.caesar_image_dspx = dspgrid
            self.caesar_image_dsp_profile = dspprofile
            self.caesar_image_dsp_profile_cor = ""
            self.caesar_image_Q = image3
            self.caesar_image_Qx = Qgrid
            self.caesar_image_Q_profile = Qprofile
            self.caesar_image_Q_profile_cor = ""
        else:
            self.caesar_image_dsp = ""
            self.caesar_image_dspx = ""
            self.caesar_image_dsp_profile = ""
            self.caesar_image_dsp_profile_cor = ""
            self.caesar_image_Q = ""
            self.caesar_image_Qx = ""
            self.caesar_image_Q_profile = ""
            self.caesar_image_Q_profile_cor = ""


####### FIGURE HANDLING FUNCTIONS ############

    def __updateFigures(self):
        '''Calculate and show results'''
        # update figures if needed
        if self.caesar_image != "":
            self.__convertImage()
            self.__showImage()
        if self.spectrum != "":
            self.__convertSpectrum()
            self.__showSpectrum()

    def __showSpectrum(self, clear_fig=True):
        ''' plot the spectrum if we have one '''
        fig = pylab.figure(100) # use figure 100 for the line profile
        fig.canvas.manager.set_window_title('Spectra')
        # position the figure top-right
        #pylab.get_current_fig_manager().window.wm_geometry("650x526+1000+20")
        if len(self.spectrum) != 0:
            if clear_fig:
                pylab.clf()
            # uppper figure - spectrum in energy
            pylab.subplot(2,1,1)
            pylab.plot(self.energy_unbinned, self.spectrum, label='Raw spectrum, %0.3f deg' % self.spectrum_tth)
            pylab.legend()
            pylab.xlabel('energy / keV')
            pylab.ylabel('intensity / counts')

        else:
            print("No spectrum data yet")

        if len(self.spectrum_dspx) != 0:
            # lower figure - spectrum in dspacing
            pylab.subplot(2,1,2)
            if self.plotdsp:
                pylab.plot(self.spectrum_dspx, self.spectrum_dspy, label='Converted spectrum, %0.3f deg' % self.spectrum_tth)
                pylab.legend()
                pylab.xlabel('d spacing / Angstroms')
                pylab.ylabel('intensity / AU')
            else:
                pylab.plot(self.spectrum_qx, self.spectrum_qy, label='Converted spectrum, %0.3f deg' % self.spectrum_tth)
                pylab.legend()
                pylab.xlabel('Q / Angstroms-1')
                pylab.ylabel('intensity / AU')
        else:
            print("No converted spectrum yet")

        # if possible, raise the window
        if callable(getattr(fig.canvas.manager.window, 'tkraise', None)):
            fig.canvas.manager.window.tkraise()
        # otherwise, can activateWindow, but lose the focus
        elif callable(getattr(fig.canvas.manager.window, 'activateWindow', None)):
            fig.canvas.manager.window.activateWindow()


    def __showImage(self):
        ''' show the image if we have one '''
        fig = pylab.figure(101) # use figure 101 for the line profile
        fig.canvas.manager.set_window_title('Caesar acquisition')
        # position the figure below top-right
        #tmp = pylab.get_current_fig_manager().window.wm_geometry("650x526+1000+450")
        # still need to add extents, axis labels, etc
        if len(self.caesar_image) != 0:
            pylab.clf()
            pylab.subplot(3,1,1)
            if self.logscale==True:
                pylab.imshow(np.log(self.caesar_image*self.maskE), extent=[self.energy[0], self.energy[-1], self.caesar_tth[-1], self.caesar_tth[0]])
            else:
                pylab.imshow(self.caesar_image*self.maskE, extent=[self.energy[0], self.energy[-1], self.caesar_tth[-1], self.caesar_tth[0]])
            pylab.xlabel('energy / keV')
            pylab.ylabel('two theta / degrees')
            pylab.axis("tight")
        else:
            print("No Caesar data yet")

        if len(self.caesar_image_dsp) != 0:
            pylab.subplot(3,1,2)
            if self.plotdsp:
                if self.logscale==True:
                    pylab.imshow(np.log(self.caesar_image_dsp), extent=[self.caesar_image_dspx[0], self.caesar_image_dspx[-1], self.caesar_tth[-1], self.caesar_tth[0]])
                else:
                    pylab.imshow(self.caesar_image_dsp, extent=[self.caesar_image_dspx[0], self.caesar_image_dspx[-1], self.caesar_tth[-1], self.caesar_tth[0]])
                pylab.xlabel('d spacing / Angstroms')
                pylab.ylabel('two theta / degrees')
            else:
                if self.logscale==True:
                    pylab.imshow(np.log(self.caesar_image_Q), extent=[self.caesar_image_Qx[0], self.caesar_image_Qx[-1], self.caesar_tth[-1], self.caesar_tth[0]])
                else:
                    pylab.imshow(self.caesar_image_Q, extent=[self.caesar_image_Qx[0], self.caesar_image_Qx[-1], self.caesar_tth[-1], self.caesar_tth[0]])
                pylab.xlabel('Q / Angstroms-1')
                pylab.ylabel('two theta / degrees')
            pylab.axis("tight")

            pylab.subplot(3,1,3)
            if self.plotdsp:
                pylab.plot(self.caesar_image_dspx, self.caesar_image_dsp_profile, label='Integrated profile')
                if self.caesar_image_dsp_profile_cor != "":
                    pylab.plot(self.caesar_image_dspx, self.caesar_image_dsp_profile_cor, 'r', label='Corrected intensities')
                pylab.legend()
                pylab.xlabel('d spacing / Angstroms')
                pylab.ylabel('Intensity')
            else:
                pylab.plot(self.caesar_image_Qx, self.caesar_image_Q_profile, label='Integrated profile')
                if self.caesar_image_Q_profile_cor != "":
                    pylab.plot(self.caesar_image_Qx, self.caesar_image_Q_profile_cor, 'r', label='Corrected intensities')
                pylab.legend()
                pylab.xlabel('Q / Angstroms-1')
                pylab.ylabel('Intensity')
            pylab.axis("tight")

        else:
            print("No converted Caesar data yet")
        # if possible, raise the window
        if callable(getattr(fig.canvas.manager.window, 'tkraise', None)):
            fig.canvas.manager.window.tkraise()
        # otherwise, can activateWindow, but lose the focus
        elif callable(getattr(fig.canvas.manager.window, 'activateWindow', None)):
            fig.canvas.manager.window.activateWindow()

######## CAESAR DATA FUNCTIONS  ###########
    # This is not useful if we use linear interpolation for rebinning

    def setRebinRange(self, rebin_range=0):
        '''set the energy range for rebinning Caesar data to ADX'''
        self.caesar_rebin_range = rebin_range
        print("Caesar rebin range is %0.3f keV" % self. caesar_rebin_range)
        print("Use setRebinRange() to modify")

    def showADXSpectrum(self, set_energy=40, clear_fig=True):
        ''' show an ADX spectrum extracted from the caesar data, with rebinning '''
        # this should do the actual Caesar rebinning of adjacent channels to improve the spectrum...
        if self.caesar_image!="":

            # find the channel closest to the set_energy
            tmp = abs(self.energy-set_energy)
            channel = np.nonzero(tmp==min(tmp))[0]
            true_energy = self.energy[channel]

            # get the simple profile
            ADX = self.caesar_image[:, channel].squeeze()
            tth_values = self.caesar_tth
            theta = old_div(self.caesar_tth*np.pi,360)

            # check the caesar_rebin_range
            dtheta = theta[1]-theta[0]
            # worst case required dE
            dE = old_div(set_energy*dtheta,theta[0])
            if dE > self.caesar_rebin_range:
                print("Using %0.2f keV rebin_range to ensure no gaps" % dE)
                rebin_range = dE
            else:
                print("Using specified %0.2f keV rebin_range" % self.caesar_rebin_range)
                rebin_range = self.caesar_rebin_range
            print("To modify this parameter, use setRebinRange method")

            # select all channels within the energy range specified
            tmp = abs(self.energy-(set_energy-(old_div(rebin_range,2))))
            channelA = np.nonzero(tmp==min(tmp))[0]-1
            tmp = abs(self.energy-(set_energy+(old_div(rebin_range,2))))
            channelB = np.nonzero(tmp==min(tmp))[0]+1

            # extract the relevent block of data
            blockint = self.caesar_image[:, channelA:channelB]
            blocklambda = 12.398/self.energy[channelA:channelB]

            # recalculate dspacings and equivilent thetas
            grid = np.meshgrid(np.r_[blocklambda], np.r_[theta])
            blockdsp = old_div(grid[0],(2*np.sin(grid[1])))
            newlambda = 12.398/set_energy
            newtheta = np.arcsin(old_div(newlambda,(2*blockdsp)))
            blocktth = old_div(newtheta*360,np.pi)
            # grid for rebinning data
            tth_grid = np.linspace(tth_values[0], tth_values[-1], self.caesar_rebin_nbins)
            ptth = blocktth.flatten(1)
            pint = blockint.flatten(1)
            ii = np.argsort(ptth)
            ptth = ptth[ii]
            pint = pint[ii]

            # rebin - simple linear interpolation?
            # may not be the best solution for noise
            if self.caesar_rebin_method=='linear':
                print("Resampling data using linear interpolation")
                print("Note for this case, no benefit to using a wider rebin range... ")
                ADX2 = np.round(np.interp(tth_grid, ptth, pint, 0, 0)) # round to whole numbers
            elif self.caesar_rebin_method=='rebin':
                print("Rebinning with range %0.1f keV and %d bins" % (self.caesar_rebin_range, self.caesar_rebin_nbins))
                ADX2 = np.zeros(tth_grid.shape)
                hbinw = old_div((tth_grid[1]-tth_grid[0]),2)
                for ii in range(len(tth_grid)):
                    ndx = np.nonzero(np.logical_and(ptth>(tth_grid[ii]-hbinw),  ptth<(tth_grid[ii]+hbinw)))
                    if len(ndx[0])!=0:
                        ADX2[ii] = pint[ndx].mean()
            else:
                print("rebin method [%s] not recognised!" % self.caesar_rebin_method)
                return

            # add the current profiles to self
            self.caesar_ADXx = tth_grid
            self.caesar_ADXy = ADX2
            self.caesar_ADXenergy = set_energy

            # automatically export the ADX spectrum as a GSAS file
            self.exportGSAS_ADX()

            fig = pylab.figure(102) # use figure 102 for the extracted profile
            fig.canvas.manager.set_window_title('Extracted ADX spectrum')
            if clear_fig:
                pylab.clf()
            pylab.plot(tth_values, ADX, 'o', label='single ADX spectrum at %0.1f keV' % true_energy)
            pylab.plot(tth_grid, ADX2, '-x', label='rebinned ADX spectrum at %0.1f keV' % set_energy)
            pylab.xlabel('two theta / deg')
            pylab.ylabel('intensity / AU')
            pylab.legend()
            # if possible, raise the window
            if callable(getattr(fig.canvas.manager.window, 'tkraise', None)):
                fig.canvas.manager.window.tkraise()
            # otherwise, can activateWindow, but lose the focus
            elif callable(getattr(fig.canvas.manager.window, 'activateWindow', None)):
                fig.canvas.manager.window.activateWindow()
        else:
            print("No Caesar data yet")


    def showEDXSpectrum(self, caesar_angle=0, clear_fig=True):
        ''' Pick an EDX spectrum from the current caesar'''
        if self.caesar_tth != "" and caesar_angle>=self.caesar_tth[0] and caesar_angle<=self.caesar_tth[-1]:
            # find the profile closest to the requested angle
            #angles = np.linspace(self.caesar_tth_range[0], self.caesar_tth_range[1], self.caesar_image.shape[0], True)
            angles = self.caesar_tth

            tmp = abs(angles - caesar_angle)
            step = np.nonzero(tmp==min(tmp))[0]
            self.spectrum = self.caesar_image[step[0], :]
            self.spectrum_tth = angles[step][0]

            self.__convertSpectrum()
            self.__showSpectrum(clear_fig)
        else:
            print("can't find this angle in the current Caesar - load data?")



    def exportGSAS_EDX(self, filename="psiche", directory="/nfs/ruche-psiche/psiche-soleil/tempdata/com-psiche/test_directory"):
        '''Export the current energy dispersive spectrum as a .gsas file
            Exports also a text file (.x_y) in two column format '''

        # get the next filename - if a name is supplied or at SOLEIL (no tk)
        if (filename!="psiche") or SOLEIL:
            # use the given arguments
            myfilename = findNextFileName(directory+os.sep+filename, "gsas")
        else: # not at SOLEIL, and no filename supplied, so ask user via tk
            # use the GUI dialog
            print("enter filename to export EDX spectrum - you may have to search for the dialog window!")
            sys.stdout.flush()
            myfilename = tkinter.filedialog.asksaveasfilename(initialdir=self.experimentdir, filetypes=[("gsas files", ".gsas")])
            if myfilename==():
                print("no file name given - can\'t save spectrum!")
            else:
                myfilename = os.path.normpath(myfilename)

        # here we will export the current spectrum to GSAS
        nchan = len(self.spectrum)
        lrec = 5 # data points per row for ESD
        nrec = np.ceil(old_div(nchan,lrec))
        calib = self.energy_calibration

        # write a gsas file in the correct format...
        # open the file
        myfile = open(myfilename, 'w')
        # looks like we want 80 characters, plus \r\n
        # value, err, value, err, etc
        # use the filename for the title
        myline = filename.ljust(80)+"\r\n"
        myfile.write(myline)
        # write the calibration line
        myline = "BANK 1 {} {} EDS {:0.3f} {:0.5f} {:0.12f} 0 ESD".format(nchan-1, nrec-1, calib[0], calib[1], calib[2])
        myline = myline.ljust(80)+"\r\n"
        myfile.write(myline)
        # write the data...
        for ii in range(int(nrec)):
            myline = ""
            for jj in range(lrec):
                ndx = (ii*lrec)+jj
                if ndx<nchan:
                    mynum = "{:d},".format(int(self.spectrum[ndx])).rjust(8)
                    myerr = "{:d},".format(int(np.sqrt(self.spectrum[ndx]))).rjust(8)
                    myline = myline + mynum + myerr
            myline = myline.ljust(80)+"\r\n"
            myfile.write(myline)
        myfile.close()
        print("gsas file: \n  %s\nwritten" % myfilename)

        # write a text file as well
        textfilename = myfilename[0:-5] + ".x_y"
        # open the file
        myfile = open(textfilename, 'w')
        # do not write a header of CowderCell
        #myfile.write(filename + "\r\n")
        for ii in range(nchan):
            myfile.write("{:.3f}".format(self.energy[ii]) + " " + "{:d}".format(int(self.spectrum[ii])) + "\r\n")
        myfile.close()
        print("txt file: \n  %s\nwritten" % textfilename)
        # write a pd text file as well
        pdtextfilename = myfilename[0:-5] + ".pd"
        # open the file
        myfile = open(pdtextfilename, 'w')
        # do not write a header of CowderCell
        #myfile.write(filename + "\r\n")
        for ii in range(nchan):
            myfile.write("{:d}".format(int(self.energy[ii]*1000)) + " " + "{:d}".format(int(self.spectrum[ii])) + "\r\n")
        myfile.close()
        print("pd txt file: \n  %s\nwritten" % pdtextfilename)

    def exportGSAS_ADX(self):
        '''Export the current energy dispersive spectrum as a .gsas file'''
        # export into the datadirectory with all the caesar text files
        # filename is "scan name" _ "set energy" keV_ "X".gsas, where X is a number to prevent overwriting

        # "scan name" is the name of the directory
        scanname = self.datadir.split(os.sep)[-1]
        filename = self.datadir + os.sep + scanname + "_" + str(self.caesar_ADXenergy) + "keV"
        myfilename = findNextFileName(filename, "gsas")

        # open the file
        myfile = open(myfilename, 'w')
        # write the title
        myline = myfilename.split(os.sep)[-1] + " (rebinned Caesar)"
        myline = myline[0:65] # crop to the 66 characters used by GSAS
        myline = myline.ljust(80)+"\r\n"
        myfile.write(myline)
        # write the calibration line
        nchan = len(self.caesar_ADXx)
        lrec = 10 # data points per row for STD
        nrec = np.ceil(float(nchan)/lrec)
        tth0 = self.caesar_ADXx[0]
        tthstep = old_div((self.caesar_ADXx[-1]-self.caesar_ADXx[0]),(nchan-1))
        calib = [tth0*100, tthstep*100] # for centidegrees
        myline = "BANK 1 {} {} CONST {:0.8f} {:0.8f} 0 0 STD".format(nchan-1, nrec-1, calib[0], calib[1])
        myline = myline.ljust(80)+"\r\n"
        myfile.write(myline)
        # write the data...
        for ii in range(int(nrec)):
            myline = ""
            for jj in range(lrec):
                ndx = (ii*lrec)+jj
                if ndx<nchan:
                    mynum = "{:d},".format(int(self.caesar_ADXy[ndx])).rjust(8)
                    myline = myline + mynum
            myline = myline.ljust(80)+"\r\n"
            myfile.write(myline)
        myfile.close()
        print("gsas file: \n  %s\nwritten" % myfilename)


    def exportAllToGSAS(self):
        '''Export ALL possible EDX text files in the experiment directory (and sub directories) to GSAS'''

        # check:
        print("Current experiment directory is %s" % self.experimentdir)
        print("This will export all EDX files in this directory and subdirectories to GSAS")
        check = input("Are you sure? yes/[no] : ")
        if check=="yes" or check=='y':

            # get all the files in this directory and sub directories
            # go through the list, converting valid files
            for root, dirs, files in os.walk(self.experimentdir):

                for myfile in files:

                    # is it a text file?
                    if myfile[-4::] == '.txt':

                        # build the full filename
                        myfilepath = root + os.sep + myfile
                        # read the file
                        spectrum = []
                        try:
                            spectrum = np.loadtxt(myfilepath)
                        except:
                            print("file: %s not suitable" % myfilepath)

                        # is the spectrum empty, does it have length 2048?
                        if len(spectrum)==2048 and not(spectrum==0).all():

                            # write a gsas file
                            # remove the .txt to leave the prefix
                            myfileprefix = myfile[0:-4]
                            # split into filename and directory for exportGSAS_EDX
                            print("export from file %s" % myfilepath)
                            self.spectrum = spectrum # to be visible in export method
                            self.exportGSAS_EDX(filename=myfileprefix, directory=root)
                        else:
                            print("file %s is the wrong length or only zeros" % myfile)


    def __exportGSAS_combined(self):
        '''export the single profile summary of a Caesar scan as a pseudo ADX spectra at 30 keV'''
        if self.caesar_image_dsp!="":
            print("Exporting single ADX profile summarising Caesar acquistion (Using corrected intensities): ")
            # convert to ADX profile
            pint = self.caesar_image_dsp_profile*1.
            pdsp = self.caesar_image_dspx*1.
            wlength = 12.398/30
            ptth = 2*np.arcsin(old_div(wlength,(2*pdsp)))
            # put in ascending order
            pint = pint[::-1]
            ptth = ptth[::-1]
            # interpolate to constant steps in tth
            ptthgrid = np.linspace(ptth[0], ptth[-1], 4096)
            pintgrid = np.interp(ptthgrid, ptth, pint, 0, 0)
            # convert to degrees
            ptthgrid = old_div(ptthgrid*180,np.pi)
            # limit max value
            if pintgrid.max() > 99999:
                fac = np.ceil(np.log10(old_div(pintgrid.max(),100000)))
                pintgrid = old_div(pintgrid, (10**fac))
            # write this as a GSAS file
            # "scan name" is the name of the directory
            scanname = self.datadir.split(os.sep)[-1]
            filename = self.datadir + os.sep + scanname + "_combined_30keV"
            myfilename = findNextFileName(filename, "gsas")
            # open the file
            myfile = open(myfilename, 'w')
            # write the title
            myline = myfilename.split(os.sep)[-1] + " (combined Caesar scan)"
            myline = myline[0:65] # crop to the 66 characters used by GSAS
            myline = myline.ljust(80)+"\r\n"
            myfile.write(myline)
            # write the calibration line
            nchan = len(ptthgrid)
            lrec = 10 # data points per row for STD
            nrec = np.ceil(float(nchan)/lrec)
            tth0 = ptthgrid[0]
            tthstep = old_div((ptthgrid[-1]-ptthgrid[0]),(nchan-1))
            calib = [tth0*100, tthstep*100] # for centidegrees
            myline = "BANK 1 {} {} CONST {:0.8f} {:0.8f} 0 0 STD".format(nchan-1, nrec-1, calib[0], calib[1])
            myline = myline.ljust(80)+"\r\n"
            myfile.write(myline)
            # write the data...
            for ii in range(int(nrec)):
                myline = ""
                for jj in range(lrec):
                    ndx = (ii*lrec)+jj
                    if ndx<nchan:
                        mynum = "{:d},".format(int(pintgrid[ndx])).rjust(8)
                        myline = myline + mynum
                myline = myline.ljust(80)+"\r\n"
                myfile.write(myline)
            myfile.close()
            print("gsas file: \n  %s\nwritten" % myfilename)

            # export two column text file too - Q
            if self.caesar_image_Q_profile != '':
                filenametxt = self.datadir + os.sep + scanname + "_Q"
                myfilenametxt = findNextFileName(filenametxt, "txt")
                myfiletxt = open(myfilenametxt, 'wt')
                myfiletxt.write('Q(A-1), intensity\n')
                for ii in range(len(self.caesar_image_Qx)):
                    myfiletxt.write('%f,%f\n' % (self.caesar_image_Qx[ii], self.caesar_image_Q_profile[ii]))
                myfiletxt.close()
                print("text file %s written and closed..." % myfilenametxt)
            # export two column text file too - dsp
            if self.caesar_image_dsp_profile != '':
                filenametxt = self.datadir + os.sep + scanname + "_dsp"
                myfilenametxt = findNextFileName(filenametxt, "txt")
                myfiletxt = open(myfilenametxt, 'wt')
                myfiletxt.write('d(A), intensity\n')
                for ii in range(len(self.caesar_image_dspx)):
                    myfiletxt.write('%0.8f,%0.8f\n' % (self.caesar_image_dspx[ii], self.caesar_image_dsp_profile[ii]))
                myfiletxt.close()
                print("text file %s written and closed..." % myfilenametxt)
            # export two column text file too - Q cor - smoothed
            if self.caesar_image_Q_profile_cor != '':
                filenametxt = self.datadir + os.sep + scanname + "_Q_cor"
                myfilenametxt = findNextFileName(filenametxt, "txt")
                myfiletxt = open(myfilenametxt, 'wt')
                myfiletxt.write('Q(A-1), intensity\n')
                for ii in range(len(self.caesar_image_Qx)):
                    myfiletxt.write('%f,%f\n' % (self.caesar_image_Qx[ii], self.caesar_image_Q_profile_cor[ii]))
                myfiletxt.close()
                print("text file %s written and closed..." % myfilenametxt)
            # export two column text file too - dsp cor - smoothed
            if self.caesar_image_dsp_profile_cor != '':
                filenametxt = self.datadir + os.sep + scanname + "_dsp_cor"
                myfilenametxt = findNextFileName(filenametxt, "txt")
                myfiletxt = open(myfilenametxt, 'wt')
                myfiletxt.write('d(A), intensity\n')
                for ii in range(len(self.caesar_image_dspx)):
                    myfiletxt.write('%0.8f,%0.8f\n' % (self.caesar_image_dspx[ii], self.caesar_image_dsp_profile_cor[ii]))
                myfiletxt.close()
                print("text file %s written and closed..." % myfilenametxt)



        else:
            print("No converted Caesar data found")

    def fitPeaksOn(self):
        if self.spectrum != "":
            # having displayed the figure, connect the button click thing
            fig = pylab.figure(100)
            self.cid = fig.canvas.mpl_connect('button_press_event', self.clickfit)
            print("Click on the peak to fit...")

    def fitPeaksOff(self):
        if self.spectrum != "":
            # having displayed the figure, connect the button click thing
            fig = pylab.figure(100)
            if self.cid != '':
                fig.canvas.mpl_disconnect(self.cid)
                print("Peak fitting off")
                self.cid = ""

    def clickfit(self, event):
        # get a x/y coordinate from the spectrum
        xx, yy = event.xdata, event.ydata
        #global coords
        coords = np.array([xx,yy])
        if self.plotdsp:
            xdata = self.spectrum_dspx
            ydata = self.spectrum_dspy
        else:
            xdata = self.spectrum_qx
            ydata = self.spectrum_qy
        xndx = np.nonzero(np.abs(xdata-coords[0])==np.abs(xdata-coords[0]).min())[0][0]
        # pick a better range to fit over
        test = True
        count = 6
        while test and (count<50):
            if ydata[xndx+count]<ydata[xndx+count-3]:
                count+=1
            else:
                test = False
        xndxB = xndx+count
        # and in the other direction
        test = True
        count = 6
        while test and (count<50):
            if ydata[xndx-count]<ydata[xndx-count+3]:
                count+=1
            else:
                test = False
        xndxA = xndx-count
        a,b = fitting.fit_gaussian_slope(xdata[xndxA:xndxB], ydata[xndxA:xndxB])
        #pylab.figure(100) # spectrum figure
        pylab.subplot(2,1,2)
        pylab.plot(xdata[xndxA:xndxB], b)
        if self.plotdsp:
            print("Peak center = %0.5f Angstrom" % a[1])
        else:
            print("Peak center = %0.5f Angstrom-1" % a[1])

###  fitting functions   ###





def optfunc(x, imE, tth, energy, ndx1, ndx2, weighting, smooth, for_optimisation=True):
    # don't need to shift images - use deviation from the mean at each Q
    # if xdata and xpoints are the same length no need to interpolate
    # x - these are points describing the incident spectrum in the range ndx1:ndx2
    xdata = np.arange(ndx1, ndx2)
    if smooth == 'y':
        xpoints = np.linspace(xdata[0], xdata[-1], len(x))
        spectrumf = interpolate.interp1d(xpoints, x, kind='linear', bounds_error=False, fill_value=1.)
        spectrum = spectrumf(xdata)
    else:
        spectrum = x * 1.
    spectrum = old_div(spectrum, spectrum.max())
    spectrum2D = np.tile(spectrum, (imE.shape[0], 1))

    # normalise using this
    imEnorm = old_div(imE[:, ndx1:ndx2], spectrum2D)
    #imEnorm = imEnorm / imEnorm.max()
    # avoid nans
    imEnorm = old_div(imEnorm, imEnorm[np.nonzero(np.logical_not(np.isnan(imEnorm)))].max())
    # convert image to Q
    imQ, Qgrid = myConvertImage(imEnorm, tth, energy[ndx1:ndx2])

    # mean along zero, ignoring nans
    sumQ = np.nansum(imQ, 0) # sum of not nans
    countQ = (np.logical_not(np.isnan(imQ))).sum(0) # number of not nans
    meanQ = old_div(sumQ, countQ)
    meanQ = np.tile(meanQ, (imQ.shape[0], 1))
    # difference
    difQ = imQ - meanQ
    if weighting:
        # divide dif by mean - thus more weight to low intensities
        difQ = old_div(difQ, meanQ)

    dif = difQ[np.nonzero(np.logical_not(np.isnan(imQ)))]

    if for_optimisation:
        return dif
    else:
        return dif, imEnorm, imQ, Qgrid, spectrum


def myConvertImage(im, tth, energy):
    ''' convert data from energy to dspacing and Q
        return Nan outside the energy range'''
    # convert the spectrum, if available
    theta = old_div(tth,2)
    # preallocate an output image
    im2 = np.zeros([im.shape[0], 1024])
    # dspacing limits for the whole acquistion
    dmax = 12.398 / (2*np.sin(old_div(theta[0]*np.pi,180))*energy[0]) # lowest angle, lowest energy
    dmin = 12.398 / (2*np.sin(old_div(theta[-1]*np.pi,180))*energy[-1]) # highest angle, highest energy
    Qmin = old_div(2*np.pi,dmax)
    Qmax = old_div(2*np.pi,dmin)
    Qgrid = np.linspace(Qmin, Qmax, 1024)
    # convert line by line
    for ii in range(im.shape[0]):
        # energy profile
        prof = im[ii, :]
        # as d spacing
        dsp = 12.398 / (2*np.sin(old_div(theta[ii]*np.pi,180))*energy)
        # as Q
        Q = old_div(2*np.pi,dsp)
        # as Q - flip profiles for increasing
        im2[ii, :] = np.interp(Qgrid, Q, prof, np.nan, np.nan)

    return im2, Qgrid


def my_plot(qimage, Qgrid, caesar_tth, colour=None, npoints=7):
    ndx = np.arange(0, qimage.shape[0], np.round(old_div(qimage.shape[0],npoints)))
    if colour != None:
        for ii in range(len(ndx)):
            pylab.plot(Qgrid, qimage[ndx[ii],:], colour, label=('two theta %0.1f' % caesar_tth[ndx[ii]]))
    else:
        for ii in range(len(ndx)):
            pylab.plot(Qgrid, qimage[ndx[ii],:], label=('two theta %0.1f' % caesar_tth[ndx[ii]]))
    pylab.legend()
