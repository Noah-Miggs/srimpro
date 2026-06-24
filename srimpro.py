# -*- coding: utf-8 -*-
"""
Created on Fri May 15 09:12:41 2026

@author: noahm
"""

import srim.output as srim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import os
import scipy as sci
import numpy.ma as ma
from matplotlib import ticker
import pandas as pd

#%% Interpolation Subroutines

# functions that performs gaussian elimination on a matrix equation Ax = b (where A is a matrix, x is the vector to be
# solved for, and b is a vector) to put it into upper diagonal form so that is can be solved by back-substitution
def gausselim2(A, b):
    # convert A and b to numpy arrays
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    # get size of A and b
    n = b.size
    # loop through columns
    for j in range(0,n-1):
        # find row with largest entry in current column and swap it with the pivot (diagonal) row
        k = np.argmax(A[j:,j]) # find pseudo-index of largest entry in column j starting from row j
        tempA = np.copy(A[j]) # copy row j so it can be swapped
        tempb = b[j] # copy corresponding entry of b so it can be swapped
        A[j] = np.copy(A[j+k]) # set row j to row k
        b[j] = b[j+k]
        A[j+k] = tempA # set row k to previous row j (now rows have been fully swapped)
        b[j+k] = tempb
        # go down current column to eliminate entries below the diagonal entry
        for i in range(j+1, n):
            # calculate "elimination factor"
            Cij = A[i,j] / A[j,j]
            # subtract Cij * row j from row i
            A[i,j:] = A[i,j:] - Cij * A[j,j:] # only subtract part of rows from diagonal on for efficiency
            b[i] = b[i] - Cij * b[j] # subtract Cij * b[j] from b[i]
    # return tuple of new A matrix and b vector
    return A, b


# function that performs back-substitution on a matrix equation of the form Ux = bs (where U is a matrix, x is the vector
# to be solved for, and bs is a vector) to solve for x, then returns x
def backsub2(U, bs):
    # convert U an bs to numpy arrays
    U = np.array(U)
    bs = np.array(bs)
    # get size of matrix/vector, and create x solution array
    n = bs.size
    xs = np.zeros(n)
    # calculate last index of solution
    xs[n-1] = bs[n-1]/U[n-1, n-1]
    # starting from the second last matrix row and increasing, calculate the index of the solution
    for i in range (n-2, -1, -1):
        # avoid looping through columns of matrix using matrix multiplication (vectorization)
        # multiply two arrays together: (1) part of current row of matrix starting at entry to the
        # right of the diagonal entry and going to the end of the row, (2) x solution array starting
        # the last index solved for and going to the end
        xs[i] = (bs[i] - U[i, i+1:] @ xs[i+1:]) / U[i,i]
    # return solution array (x vector)
    return xs


# function that calculates the estimated function output between the known data points using cubic spline
# interpolation. Arguments: xdata = known x values, ydata = known y values
# x = point to evaluate function at
def cubicSpline(xdata, ydata, x):
    # convert to numpy arrays
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    # check that desried x value to interpolate to is within range of given data
    if not (xdata[0] <= x <= xdata[-1]):
        return None
    # get number of data points
    n = xdata.size
    # create b vector
    b = 6*((ydata[2:] - ydata[1:-1])/(xdata[2:] - xdata[1:-1])-(ydata[1:-1] - ydata[:-2])/(xdata[1:-1] - xdata[:-2]))
    # create tridiagonal matrix
    A = np.zeros((n-2, n-2), dtype=float)
    np.fill_diagonal(A, 2*(xdata[2:] - xdata[:-2])) # fill main diagonal according to equation
    i = np.arange(n-3) # array to be used to allow access to the diagonals of A asjecent to its main diagonal
    j = np.arange(1, n-2)
    A[i,j] = xdata[2:-1] - xdata[1:-2] # fill diagonal right above main diagonal
    A[j,i] = xdata[2:-1] - xdata[1:-2] # fill diagonal right below main diagonal
    # solve for "c" coefficients
    Anew, bnew = gausselim2(A, b)
    cs = backsub2(Anew, bnew)
    # add zero element to front and back of cs array
    cs = np.append(cs, [0,0])
    cs = np.roll(cs, 1) # shift all elements of array down by 1 index, wraps last value to front
    # loop through all "panels" within the given data to find which one the x values lies in
    y = 0
    for i in range(1,n):
        if xdata[i-1] <= x <= xdata[i]: # when correct panel is found, use formula to calculate interpolation
            y = (ydata[i-1]*(xdata[i]-x)/(xdata[i]-xdata[i-1])+ydata[i]*(x-xdata[i-1])/(xdata[i]-xdata[i-1])
            -cs[i-1]/6*((xdata[i]-x)*(xdata[i]-xdata[i-1])-(xdata[i]-x)**3/(xdata[i]-xdata[i-1]))
            -cs[i]/6*((x-xdata[i-1])*(xdata[i]-xdata[i-1])-(x-xdata[i-1])**3/(xdata[i]-xdata[i-1])))
    # return calculated value
    return y

#%% Integration functions

# function that calculates the integral of a given set of data.
# inputs: xdata = array containing x coordinates, ydata = array containing y coordinates.
# assumptions: xdata and y data given as arrays, size of xdata and y data is the same.
def simpsons(xdata, ydata):
    # convert xdata and y data to numpy arrays
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    
    # calculate step size (h) and get number of data points (n)
    h = xdata[1] - xdata[0]
    n = xdata.size
    
    # create weight vector
    cs = np.zeros(n, dtype=float)
    cs[::2] = 2 / 3 # set every other index of array to 2/3, starting from index 0
    cs[1::2] = 4 / 3 # set every other index of array to 4/3, starting from index 1
    cs[0] = 1 / 3 # set first index to 1/3
    cs[-1] = 1 / 3 # set last index to 1/3
    cs = h * cs
    
    # compute integral by vector multiplication
    return cs @ ydata


# function that calculates the total number of vacancies and replacements (average number per ion) within a specified 
# depth range using the SRIM output files.
# returns number of vacancies and number of replacements in that order, rounded to nearest integer.
# input: path = file path to folder containing the SRIM output text files, x1 = minimum depth, x2 = maximum depth.
# assumption: path given as a string, displacement profile and depth values in SRIM in angstrom, x1 & x2 given in
# angstrom.
def totDisplacements(path, x1, x2):
    # get vacancy and replacment output files
    Vac = srim.Vacancy(path)
    NoVac = srim.NoVacancy(path)
    
    # access the required data within the output files
    vac = Vac.vacancies # total vacancy profile (all target elements)
    novac = NoVac.number # total replacement profile (all target elements)
    depth = Vac.depth # depth bins used in SRIM
    
    # check if selected values of x1 and x2 are valid
    if x1 < 0 or x2 < 0:
        print("x1 and x2 must be positive")
        return None
    elif x1 >= x2:
        print("x1 must be less than x2")
        return None
    
    # sum columns of vac (vacancies by each element) to get total vacancy production
    vac = np.sum(vac, axis=1)
    
    # restrict vac, novac, and depth to between x1 and x2
    subset = ((depth >= x1) & (depth <= x2))
    vac = vac[subset]
    novac = novac[subset]
    depth = depth[subset]
    
    # calculate total number of vacancies and replacments by integration (simpson's rule)
    vactot = round(simpsons(depth, vac))
    novactot = round(simpsons(depth, novac))
    
    return vactot, novactot


# function that automatically calculates the total energy deposition to various means (average per ion) within a 
# specified depth range using the SRIM output files.
# returns two arrays: array 1 contains the total energy deposited as ionization energy and the percent of energy
# deposited as ionization energy, array 2 contains the total energy deposited as damage energy and the percent of energy.
# deposited as damage energy. note that all returned energy values are in keV.
# input: path = file path to folder containing the SRIM output text files, x1 = minimum depth, x2 = maximum depth,
# method = indicator for what method to use for calculation of damage energy (either 'phonons_1' which calculates damage 
# energy by adding phonons by recoil and phonons by ion from PHONON.txt file, 'phonons_2' which calculates damage energy
# by using only phonons by recoil from PHONON.txt file, or 'cons' which calculates damage energy as the incident ion 
# energy deposited between x1 and x2 minus the total ionization energy deposited between x1 and x2).
# assumption: path given as a string, energy deposition profiles and depth values in SRIM in eV and angstrom, x1 & x2
# given in angstrom.
def totEnergyDep(path, x1, x2, method='cons'):
    # get ionization and damage energy output files
    ioniz = srim.Ioniz(path)
    damage = srim.Phonons(path)
    Nuclear = srim.EnergyToRecoils(path)
    
    # access the required data within the output files
    ionizbyion = ioniz.ions # ionization energy by ion
    ionizbyrec = ioniz.recoils # ionization energy by recoils
    dambyion = damage.ions # damage energy by ions
    dambyrec = damage.recoils # damage energy by recoils
    nuclear = Nuclear.ions # ion energy lost to nuclear stopping
    depth = ioniz.depth # depth bins used in SRIM
    
    # check if selected values of x1 and x2 are valid
    if x1 < 0 or x2 < 0:
        print("x1 and x2 must be positive")
        return None
    elif x1 >= x2:
        print("x1 must be less than x2")
        return None
    
    # calculate total ionization energy profile and total damage energy profile
    ioniztot = ionizbyion + ionizbyrec
    
    # restrict ioniztot, damtot, and depth to between x1 and x2
    subset = ((depth >= x1) & (depth <= x2))
    ioniztot = ioniztot[subset]
    ionizbyion = ionizbyion[subset]
    dambyrec = dambyrec[subset]
    dambyion = dambyion[subset]
    nuclear = nuclear[subset]
    depth = depth[subset]
    
    # calculate total energy deposition by integration (simpson's rule)
    ionizE = simpsons(depth, ioniztot)/1000 # convert from eV to keV
    
    # calculate total damage energy using selected method
    if method == 'phonons_1':
        damtot = dambyrec + dambyion
        damE = simpsons(depth, damtot)/1000 # convert from eV to keV
    elif method == 'phonons_2':
        damtot = dambyrec
        damE = simpsons(depth, damtot)/1000 # convert from eV to keV
    elif method == 'cons':
        nuclearE = simpsons(depth, nuclear)/1000 # total ion energy lost to nuclear stopping, convert from eV to keV
        electronicE = simpsons(depth, ionizbyion)/1000 # total ion energy lost to electronic stopping, in keV
        ionE = nuclearE + electronicE # total ion energy lost between x1 and x2
        damE = ionE - ionizE
    
    # calculate percent energy deposition to each
    totE = ionizE + damE
    ionizpercent = ionizE/totE*100
    dampercent = damE/totE*100
    
    return [ionizE, ionizpercent], [damE, dampercent]


# function that automatically calculates the total ion energy lost to nuclear and electronic stopping (average per ion)
# within a specified depth range using the SRIM output files.
# returns two arrays: array 1 contains the total energy lost to electronic stopping and the percent of ion energy lost 
# to electronic stopping, array 2 contains the total energy lost to nuclear stopping and the percent of ion energy lost
# to nuclear stopping. note that all returned energy values are in keV.
# input: path = file path to folder containing the SRIM output text files, x1 = minimum depth, x2 = maximum depth.
# assumption: path given as a string, energy deposition profiles and depth values in SRIM in eV and angstrom, x1 & x2
# given in angstrom.
def totStopping(path, x1, x2):
    # get ionization and nuclear stopping output files
    ioniz = srim.Ioniz(path)
    Nuclear = srim.EnergyToRecoils(path)
    
    # access the required data within the output files
    electronic = ioniz.ions # ionization energy by ion
    nuclear = Nuclear.ions # nuclear stopping power
    depth = ioniz.depth # depth bins used in SRIM
    
    # check if selected values of x1 and x2 are valid
    if x1 < 0 or x2 < 0:
        print("x1 and x2 must be positive")
        return None
    elif x1 >= x2:
        print("x1 must be less than x2")
        return None
    
    # restrict electronic, nuclear, and depth to between x1 and x2
    subset = ((depth >= x1) & (depth <= x2))
    electronic = electronic[subset]
    nuclear = nuclear[subset]
    depth = depth[subset]
    
    # calculate total energy deposition by integration (simpson's rule)
    electronicE = simpsons(depth, electronic)/1000 # convert from eV to keV
    nuclearE = simpsons(depth, nuclear)/1000 # convert from eV to keV
    
    # calculate percent energy deposition to each
    totE = electronicE + nuclearE
    electronicpercent = electronicE/totE*100
    nuclearpercent = nuclearE/totE*100
    
    return [electronicE, electronicpercent], [nuclearE, nuclearpercent]

#%% Basic plotting functions

# function that automatically generates a plot of both dpa profile and ion range profile using the SRIM output files.
# inputs: path = file path to folder containing the SRIM output text files, fluence = ion fluence, x1 = minimum depth
# (defaults to 0), x2 = maximum depth (defaults to maximum depth used in SRIM), units = indicator for what units to use 
# for ion density (either 'at_%' or 'density_per_fluence', defaults to 'at_%'), layerdepth = array of bottom depths of 
# target material layers (if not specified, reads this info from TDATA.txt), output = indicator for what to output (either
# 'plot' or 'excel', defaults to 'plot').
# assumptions: fluence given in cm^-2, path given as a string, fluence given as scalar (in units of cm^-2), x1 & x2 given 
# in angstrom, layerdepth given as an array.
def rangeAndDpa(path, fluence, x1=None, x2=None, units='at_%', layerdepth=None, output='plot'):
    # check that input for units is valid
    if not (units == 'at_%' or units == 'density_per_fluence'):
        raise ValueError("Only units 'at_%' or 'density_per_fluence' is accepted")
    
    # check that input for output is valid
    if not (output == 'plot' or output == 'excel'):
        raise ValueError("Only output 'plot' or 'excel' is accepted")
    
    # get layer information from output files
    trim_in_path = os.path.join(path, "TDATA.txt") # specify path to TDATA.txt file within 'path' directory
    with open(trim_in_path, "r") as f:
        file = f.read()
    layerdata = np.array(srim.SRIM_Output()._read_target(file)) # get target information from TDATA.txt
    
    # if no layer depths are given, read them from TDATA.txt
    if layerdepth is None:
        # access layer depth data (what depth each layer ends at)
        layerdepth = np.array([]) # empty array to add layer depths to
        for layer in layerdata:
            layerdepth = np.append(layerdepth, layer["bottom_depth_A"])
    
    # if layerdepth given, convert to numpy array
    else:
        layerdepth = np.array(layerdepth)
    
    # convert layerdepth to nm from angstrom
    layerdepth = layerdepth*0.1
    
    # access layer density data
    layerdensity = np.array([]) # empty array to add layer depths to
    for layer in layerdata:
        layerdensity = np.append(layerdensity, layer["atomic_density"])
    
    # get range, vacancy, and replacement output files
    Range = srim.Range(path)
    Vac = srim.Vacancy(path)
    NoVac = srim.NoVacancy(path)
    
    # access the required data within output files
    ionrange = Range.ions # ion range distrubution
    vac = Vac.vacancies # array of vacancy profile for all target elements
    novac = NoVac.number # replacement profile
    depth = Range.depth*0.1 # depth bins used in SRIM - convert to nm from angstrom
    
    # if only 1 target layer, set layer depth to the last depth value
    if len(layerdensity) == 1:
        layerdepth = np.array([depth[-1]])
    
    # set minimum and maximum depth
    if x1 is None:
        x1 = 0
    else:
        x1 = x1*0.1 # convert to nm from angstrom
    if x2 is None:
        x2 = depth[-1]
    else:
        x2 = x2*0.1 # convert to nm from angstrom
    
    # check if selected values of x1 and x2 are valid
    if x2 > depth[-1]:
        raise ValueError("Selected x2 value was outside of SRIM calculated depth range")
    elif x1 < 0 or x2 < 0:
        raise ValueError("x1 and x2 must be positive")
    elif x1 >= x2:
        raise ValueError("x1 must be less than x2")
    
    # combine vacancies and replacements into displacements
    displacements = np.sum(vac, axis=1) + novac # np.sum adds up vacancies of each target element
    
    # determine the indices of the layer boundaries (index of the output data a layer ends and the next layer starts)
    dx = depth[1] - depth[0] # depth increment
    # convert the array of the depths of each target layer into the index at which each layer ends
    layerindex = layerdepth//dx
    layerindex = layerindex.astype(int) # ensure layerindex entries are ints so they can be used as indices
    
    # calculate dpa, accounting for different layer atomic densities
    dpa = np.zeros(depth.size) # blank array for dpa data to be added to
    dpa[:layerindex[0]+1] = displacements[:layerindex[0]+1]*fluence/layerdensity[0]*1e8 # calculate dpa for first layer
    for i in range(layerindex.size-1): # loop through all subsequent target layers to calculate the dpa in each
        dpa[layerindex[i]+1:layerindex[i+1]+1] = (displacements[layerindex[i]+1:layerindex[i+1]+1]
                                                  *fluence/layerdensity[i+1]*1e8)
    
    if units == 'at_%': # only convert to atomic percent if option a not selected
        # convert ionrange data from ion density to atomic percent, accounting for different layer atomic densities
        ionrange[:layerindex[0]+1] = ionrange[:layerindex[0]+1]*fluence*100/layerdensity[0] # first layer
        for i in range(layerindex.size-1): # loop through all subsequent target layers
            ionrange[layerindex[i]+1:layerindex[i+1]+1] = (ionrange[layerindex[i]+1:layerindex[i+1]+1]
                                                      *fluence*100/layerdensity[i+1])
    
    # restrict ionrange & dpa to only include data between x1 and x2
    subset = ((depth >= x1) & (depth <= x2))
    ionrange = ionrange[subset]
    dpa = dpa[subset]
    depth = depth[subset]
    
    # create curve fit for ion range distribution via interpolation using a lagrange basis
    # def f(x, a, b, c): # function to pass into scipy curve fitting function (assuming ion range has gaussian shape)
    #     return a*np.exp(-(x-b)**2/(2*c**2))
    # p0 = [max(ionrange), depth[np.argmax(ionrange)], np.std(depth)] # intial guess for parameters
    # fparams, convariance = sci.optimize.curve_fit(f, depth, ionrange, p0=p0) # determine parameters for curve fit
    
    # n = 500 # 500 points to create a smooth fitted curve
    # xfit = np.linspace(depth[0], depth[-1], n)
    # yfit = f(xfit, fparams[0], fparams[1], fparams[2])
    
    # plot parameters (can be modified based on individual preferences):
    linewidth = 2 # width of lines connecting points
    markersize = 8 # size of point markers
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    linestyle = 'None' # type of lines connecting points (i.e. solid, dotted, none, etc.)
    marker1 = 'o' # marker to use for ion range points
    marker2 = '^' # marker to use for damage dose points
    
    # create plot if plot option selected
    if output == 'plot':
        # ion range plotting
        fig, ax1 = plt.subplots(figsize=(12,8))
        ln1 = ax1.plot(depth, ionrange, color='blue', linewidth=linewidth, linestyle=linestyle, marker=marker1, 
                       markersize=markersize, label="Ion Distribution")
        # ax1.plot(xfit, yfit, color='blue', linewidth=linewidth, linestyle='--')
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                        direction='in')
        ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
        ax1.set_xlabel("Depth [nm]", fontsize=axlabsize)
        if units == 'at_%':
            ax1.set_ylabel("Ion Density [at. %]", fontsize=axlabsize)
        else:
            ax1.set_ylabel("Ion Density [(Atoms/cm\u00b3)/(Atoms/cm\u00b2)]", fontsize=axlabsize)
        # damage dose plotting
        ax2 = ax1.twinx()
        ln2 = ax2.plot(depth, dpa, color='purple', linewidth=linewidth, linestyle=linestyle, marker=marker2, 
                       markersize=markersize, label="Damage Profile")
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.tick_params(axis='y', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                        direction='in')
        ax2.tick_params(axis='y', which='minor', length=minorlength, width=tickwidth, direction='in')
        ax2.set_ylabel("Damage Dose [dpa]", fontsize=axlabsize)
        # add legend and show plot
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.45, 0))
        plt.show()
    
    # export data to excel if excel option selected
    elif output == 'excel':
        # convert arrays to a pandas data frame so it can be exported to excel
        data = np.empty((len(depth),3))
        data[:,0] = depth
        data[:,1] = ionrange
        data[:,2] = dpa
        if units == 'at_%':
            label = "Ion Range (at. %)"
        else:
            label = "Ion Range ((Atoms/cm\u00b3)/(Atoms/cm\u00b2))"
        df = pd.DataFrame(data, columns=["Depth (nm)", label, "Damage Dose (dpa)"])
        
        # in case function has run multiple times for same path, rename file to be created accordingly
        num = 0
        while True:
            # create filepath to export data frame to
            export_path = os.path.join(path, f"range+dpa_data({num}).xlsx")
            
            # check if current path name exists and continue looping if it does
            if os.path.exists(export_path):
                num = num + 1
                continue
            
            # if path name does not exist, export data frame to excel
            else:
                try:
                # Export to Excel
                    df.to_excel(export_path, index=False)
                    print("Data successfully exported to: ", export_path)
                except Exception as e:
                    print(f"Error exporting to Excel: {e}")
            
                # break out of loop once data has been exported
                break
    
    return None


# function that automatically generates a plot of energy deposition profiles using the SRIM output files.
# inputs: path = file path to folder containing the SRIM output text files, x1 = minimum depth (defaults to 0), 
# x2 = maximum depth (defaults to maximum depth used in SRIM), damage = indicator for damage energy,
# ionizTot = indicator for total ionization energy, ionizIon = indicator for ionization energy by ion, 
# ionizRec = indicator for ionization energy by recoil, method = indicator for what method to use for calculation of 
# damage energy (either 'phonons_1' which calculates damage energy by adding phonons by recoils and phonons by ion from 
# PHONON.txt file, 'phonons_2' which calculates damage energy by using only phonons by recoils from PHONON.txt file, or 
# 'cons' which calculates damage energy profile by multiplying each layer of the total displacement profile by a local 
# Tdam/v factor specific to that layer, where Tdam is the damage energy calculated as incident ion energy deposited in the
# layer minus the total ionization energy deposited in the layer, and v is the total displacements in the layer), 
# layerdepth = array of bottom depths of target material layers (if not specified, reads this info from TDATA.txt), 
# scale = indicator for what scale to use for first y-axis (either 'log' or 'linear', defaults to 'linear'), 
# output = indicator for what to output (either 'plot' or 'excel', defaults to 'plot').
# note that all plotting options are turned off by default (will generate a blank figure if no options enabled)
# assumptions: path given as string, a,b,c,d given as booleans, x1 & x2 given in angstrom, layerdepth given as an array.
def energyDep(path, x1=None, x2=None, damage = False, ionizTot = False, ionizIon = False, ionizRec = False, method='cons',
              layerdepth=None, scale='linear', output='plot'):
    # check that input for damage is valid
    if not (damage == True or damage == False):
        raise ValueError("Only damage True or False is accepted")
    
    # check that input for ionizTot is valid
    if not (ionizTot == True or ionizTot == False):
        raise ValueError("Only ionizTot True or False is accepted")
    
    # check that input for ionizIon is valid
    if not (ionizIon == True or ionizIon == False):
        raise ValueError("Only ionizIon True or False is accepted")
    
    # check that input for ionizRec is valid
    if not (ionizRec == True or ionizRec == False):
        raise ValueError("Only ionizRec True or False is accepted")
    
    # check that input for method is valid
    if not (method == 'phonons_1' or method == 'phonons_2' or method == 'cons'):
        raise ValueError("Only method 'phonons_1', 'phonons_2', or 'cons' is accepted")
    
    # check that input for scale is valid
    if not (scale == 'linear' or scale == 'log'):
        raise ValueError("Only scale 'linear' or 'log' is accepted")
    
    # check that input for output is valid
    if not (output == 'plot' or output == 'excel'):
        raise ValueError("Only output 'plot' or 'excel' is accepted")
    
    # get layer information from output files
    trim_in_path = os.path.join(path, "TDATA.txt") # specify path to TDATA.txt file within 'path' directory
    with open(trim_in_path, "r") as f:
        file = f.read()
    layerdata = np.array(srim.SRIM_Output()._read_target(file)) # get target information from TDATA.txt
    
    # if no layer depths are given, read them from TDATA.txt
    if layerdepth is None:
        # access layer depth data (what depth each layer ends at)
        layerdepth = np.array([]) # empty array to add layer depths to
        for layer in layerdata:
            layerdepth = np.append(layerdepth, layer["bottom_depth_A"])
        
    # if layerdepth given, convert to numpy array
    else:
        layerdepth = np.array(layerdepth)
        
    # convert layerdepth to nm from angstrom
    layerdepth = layerdepth*0.1
    
    # get phonon energy and ionization energy output files (& other output files that may be needed)
    damage = srim.Phonons(path)
    ioniz = srim.Ioniz(path)
    Vac = srim.Vacancy(path)
    NoVac = srim.NoVacancy(path)
    
    # access the required data within output files - all angstrom units converted to nm, and eV converted to keV
    dambyion = damage.ions/100 # damage energy by ion
    dambyrec = damage.recoils/100 # damage energy by recoils
    ionizbyion = ioniz.ions/100 # ionization energy by ion
    ionizbyrec = ioniz.recoils/100 # ionization energy by recoils
    vac = np.sum(Vac.vacancies, axis=1)*10 # vacancy profile
    novac = NoVac.number*10 # replacement profile
    depth = damage.depth*0.1 # depth bins used in SRIM - convert to nm from angstrom
    
    # if only 1 target layer, set layer depth to the last depth value
    if len(layerdepth) == 1:
        layerdepth = np.array([depth[-1]])
    
    # set minimum and maximum depth
    if x1 is None:
        x1 = 0
    else:
        x1 = x1*0.1 # convert to nm from angstrom
    if x2 is None:
        x2 = depth[-1]
    else:
        x2 = x2*0.1 # convert to nm from angstrom
    
    # check if selected values of x1 and x2 are valid
    if x2 > depth[-1]:
        raise ValueError("Selected x2 value was outside of SRIM calculated depth range")
    elif x1 < 0 or x2 < 0:
        raise ValueError("x1 and x2 must be positive")
    elif x1 >= x2:
        raise ValueError("x1 must be less than x2")
    
    # calculate total ionization energy profile
    ioniztot = ionizbyion + ionizbyrec
    
    # calculate damage energy using the selected method
    if method == 'phonons_1':
        damtot = dambyrec + dambyion
    
    elif method == 'phonons_2':
        damtot = dambyrec
        
    elif method == 'cons':
        # calculate total displacement profile
        displacements = vac + novac
        
        # restrict data to only portion of displacement profile within first layer
        subset = (depth <= layerdepth[0])
        damtot = displacements[subset]
        
        # multiply this section of the displacement profile by the local Tdam/v factor to convert to damage energy
        vactot, novactot = totDisplacements(path, 0, layerdepth[0]*10)
        v = vactot + novactot # total number of displacements
        ionizE, damE = totEnergyDep(path, 0, layerdepth[0]*10, method='cons')
        damtot = damtot*damE[0]/v
        
        # repeat for all subsequent layers, and append those sections of the damage energy profile
        for i in range(1,len(layerdepth)):
            # restrict data to only portion of displacement profile within current layer
            subset = ((depth > layerdepth[i-1]) & (depth <= layerdepth[i]))
            layer_damtot = displacements[subset]
            
            # calculate damage energy and total displacements
            vactot, novactot = totDisplacements(path, layerdepth[i-1]*10, layerdepth[i]*10)
            v = vactot + novactot # total number of displacements
            ionizE, damE = totEnergyDep(path, layerdepth[i-1]*10, layerdepth[i]*10, method='cons')
            
            # multiply the current layer's displacement profile by the conversion factor, and append it to damtot
            layer_damtot = layer_damtot*damE[0]/v
            damtot = np.append(damtot, layer_damtot)
    
    # restrict arrays to only include data between x1 and x2
    subset = ((depth >= x1) & (depth <= x2))
    damtot = damtot[subset]
    ioniztot = ioniztot[subset]
    ionizbyion = ionizbyion[subset]
    ionizbyrec = ionizbyrec[subset]
    depth = depth[subset]
    
    # plot parameters (can be modified based on individual preferences):
    linewidth = 2 # width of lines connecting points
    markersize = 6 # size of point markers
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    linestyle = 'None' # type of lines connecting points (i.e. solid, dotted, none, etc.)
    color1 = 'purple' # colour for damage energy
    color2 = 'green'# colour for total ionization
    color3 = 'red' # colour for ionization by ion
    color4 = 'blue' # colour for ionization by recoils
    marker1 = 'o' # marker for damage energy
    marker2 = '^' # marker for total ionization
    marker3 = 'D' # marker for ionization by ion
    marker4 = 's' # marker for ionization by recoils
    
    # create plot if plot option selected
    if output == 'plot':
        fig, ax1 = plt.subplots(figsize=(12,8))
        if damage: # if damage energy option selected
            ax1.plot(depth, damtot, color=color1, linewidth=linewidth, linestyle=linestyle, marker=marker1, 
                       markersize=markersize, label="Damage energy")
        if ionizTot: # if total ionization option seelcted
            ax1.plot(depth, ioniztot, color=color2, linewidth=linewidth, linestyle=linestyle, marker=marker2, 
                       markersize=markersize, label="Total Ionization")
        if ionizIon: # if ionization by ions option selected
            ax1.plot(depth, ionizbyion, color=color3, linewidth=linewidth, linestyle=linestyle, marker=marker3, 
                       markersize=markersize, label="Ionization by Ions")
        if ionizRec: # if ionization by recoils option selected
            ax1.plot(depth, ionizbyrec, color=color4, linewidth=linewidth, linestyle=linestyle, marker=marker4, 
                       markersize=markersize, label="Ionization by Recoils")
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                        direction='in')
        ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
        ax1.set_xlabel("Depth [nm]", fontsize=axlabsize)
        ax1.set_ylabel("Energy Deposition [keV/nm/ion]", fontsize=axlabsize)
        if scale == 'log':
            ax1.set_yscale('log')
        plt.legend(fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.4, 0))
        plt.show()
    
    # export data to excel if excel option selected
    elif output == 'excel':
        # convert arrays to a pandas data frame so it can be exported to excel
        data = np.empty((len(depth),5))
        data[:,0] = depth
        data[:,1] = damtot
        data[:,2] = ionizbyion
        data[:,3] = ionizbyrec
        data[:,4] = ioniztot
        df = pd.DataFrame(data, columns=["Depth (nm)", "Damage Energy (keV/nm/ion)", "Ionization by Ion (keV/nm/ion)",
                                         "Ionization by Recoils (keV/nm/ion)", "Total Ionization (keV/nm/ion)"])
        
        # in case function has run multiple times for same path, rename file to be created accordingly
        num = 0
        while True:
            # create filepath to export data frame to
            export_path = os.path.join(path, f"energydep_data({num}).xlsx")
            
            # check if current path name exists and continue looping if it does
            if os.path.exists(export_path):
                num = num + 1
                continue
            
            # if path name does not exist, export data frame to excel
            else:
                try:
                # Export to Excel
                    df.to_excel(export_path, index=False)
                    print("Data successfully exported to: ", export_path)
                except Exception as e:
                    print(f"Error exporting to Excel: {e}")
            
                # break out of loop once data has been exported
                break
    
    return None


# function that automatically generates a plot of stopping powers using the SRIM output files.
# inputs: path = file path to folder containing the SRIM output text files, x1 = minimum depth (defaults to 0), 
# x2 = maximum depth (defaults to maximum depth used in SRIM), ratio = indicator for what ratio to plot, if any (either
# 'e/n', 'n/e', or None, defaults to None), scale = indicator for what scale to use for first y-axis (either 'log' or 
# 'linear', defaults to 'linear'), output = indicator for what to output (either 'plot' or 'excel', defaults to 'plot').
# note that the plotting options for the stopping power ratios are turned off by default, but electronic stopping and
# nuclear stopping will always be plotted.
# assumptions: path given as string, a & b given as booleans, x1 & x2 given in angstrom.
def stopping(path, x1=None, x2=None, ratio=None, scale='linear', output='plot'):
    # check that input for ratio is valid
    if not (ratio == None or ratio == 'e/n' or ratio == 'n/e'):
        raise ValueError("Only ratio None, 'e/n', or 'n/e' is accepted")
    
    # check that input for scale is valid
    if not (scale == 'linear' or scale == 'log'):
        raise ValueError("Only scale 'linear' or 'log' is accepted")
    
    # check that input for output is valid
    if not (output == 'plot' or output == 'excel'):
        raise ValueError("Only output 'plot' or 'excel' is accepted")
    
    # get phonon energy and ionization energy output files
    Nuclear = srim.EnergyToRecoils(path)
    ioniz = srim.Ioniz(path)
    
    # access the required data within output files - all angstrom units converted to nm, and eV converted to keV
    nuclear = Nuclear.ions/100 # nuclear stopping power
    electronic = ioniz.ions/100 # electronic stopping power
    depth = Nuclear.depth*0.1 # depth bins used in SRIM
    
    # set minimum and maximum depth
    if x1 is None:
        x1 = 0
    else:
        x1 = x1*0.1 # convert to nm from angstrom
    if x2 is None:
        x2 = depth[-1]
    else:
        x2 = x2*0.1 # convert to nm from angstrom
    
    # check if selected values of x1 and x2 are valid
    if x2 > depth[-1]:
        raise ValueError("Selected x2 value was outside of SRIM calculated depth range")
    elif x1 < 0 or x2 < 0:
        raise ValueError("x1 and x2 must be positive")
    elif x1 >= x2:
        raise ValueError("x1 must be less than x2")
    
    # restrict arrays to only include data between x1 and x2
    subset = ((depth >= x1) & (depth <= x2))
    nuclear = nuclear[subset]
    electronic = electronic [subset]
    depth = depth[subset]
    
    # plot parameters (can be modified based on individual preferences):
    linewidth = 2 # width of lines connecting points
    markersize = 6 # size of point markers
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    linestyle = 'None' # type of lines connecting points (i.e. solid, dotted, none, etc.)
    color1 = 'blue' # colour for electronic stopping
    color2 = 'purple' # colour for nuclear stopping
    color3 = 'green' # colour for electronic/nuclear ratio
    color4 = 'red' # colour for nuclear/electronic ratio
    marker1 = '^' # marker for electronic stopping
    marker2 = 'o' # marker for nuclear stopping
    marker3 = 'D' # marker for electronic/nuclear ratio
    marker4 = 's' # marker for nuclear/electronic ratio
    
    # create plot if plot option selected
    if output == 'plot':
        # base plotting (electronic and nuclear stopping)
        fig, ax1 = plt.subplots(figsize=(12,8))
        ln1 = ax1.plot(depth, electronic, color=color1, linewidth=linewidth, linestyle=linestyle, marker=marker1, 
                   markersize=markersize, label="Electronic Stopping")
        ln2 = ax1.plot(depth, nuclear, color=color2, linewidth=linewidth, linestyle=linestyle, marker=marker2, 
                   markersize=markersize, label="Nuclear Stopping")
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                        direction='in')
        ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
        ax1.set_xlabel("Depth [nm]", fontsize=axlabsize)
        ax1.set_ylabel("Energy Deposition [keV/nm/ion]", fontsize=axlabsize)
        # additional plotting options (ratios & log scale)
        if ratio is not None:
            ax2 = ax1.twinx()
            if ratio == 'e/n': # if electronic/nuclear ratio selected
                ln3 = ax2.plot(depth, electronic/nuclear, color=color3, linewidth=linewidth, linestyle=linestyle, 
                               marker=marker3, markersize=markersize, label="Se/Sn")
            elif ratio == 'n/e': # if nuclear/electronic ratio selected
                ln4 = ax2.plot(depth, nuclear/electronic, color=color4, linewidth=linewidth, linestyle=linestyle, 
                               marker=marker4, markersize=markersize, label="Sn/Se")
            ax2.yaxis.set_minor_locator(AutoMinorLocator())
            ax2.tick_params(axis='y', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                            direction='in')
            ax2.tick_params(axis='y', which='minor', length=minorlength, width=tickwidth, direction='in')
            ax2.set_ylabel("Fraction", fontsize=axlabsize)
        if scale == 'log': # if log scale option (for primary y-axis) selected
            ax1.set_yscale('log')
        # add legend and show plot
        lns = ln1 + ln2
        if ratio == 'e/n':
            lns = lns + ln3
        if ratio == 'n/e':
            lns = lns + ln4
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.5, 0))
        plt.show()
    
    elif output == 'excel':
       # convert arrays to a pandas data frame so it can be exported to excel
       data = np.empty((len(depth),5))
       data[:,0] = depth
       data[:,1] = nuclear
       data[:,2] = electronic
       data[:,3] = electronic/nuclear
       data[:,4] = nuclear/electronic
       df = pd.DataFrame(data, columns=["Depth (nm)", "Nuclear Stopping (keV/nm/ion)", "Electronic Stopping (keV/nm/ion)", 
                                        "Se/Sn Ratio", "Sn/Se Ratio"])
       
       # in case function has run multiple times for same path, rename file to be created accordingly
       num = 0
       while True:
           # create filepath to export data frame to
           export_path = os.path.join(path, f"stopping_data({num}).xlsx")
           
           # check if current path name exists and continue looping if it does
           if os.path.exists(export_path):
               num = num + 1
               continue
           
           # if path name does not exist, export data frame to excel
           else:
               try:
               # Export to Excel
                   df.to_excel(export_path, index=False)
                   print("Data successfully exported to: ", export_path)
               except Exception as e:
                   print(f"Error exporting to Excel: {e}")
           
               # break out of loop once data has been exported
               break
    
    return None


# function that automatically generates a plot of damage energy profiles and prints the total damage energy calculated 
# using the 3 methods (Phonon Method 1, Phonon Method 2, Energy conservation method) using the SRIM output files. Phonon 
# Method 1 calculates the damage energy profile by summing the phonons by ion and phonons by recoils profiles from the 
# PHONON.txt file. Phonon Method 2 calculates the damage energy profile only using phonons by recoils from the PHONON.txt 
# file. Energy Conservation Method calculates the damage energy profile by multiplying each layer of the total 
# displacement profile by a local Tdam/v factor specific to that layer, where Tdam is the damage energy calculated as 
# incident ion energy deposited in the layer minus the total ionization energy deposited in the layer, and v is the total 
# displacements in the layer.
# inputs: path = file path to folder containing the SRIM output text files, x1 = minimum depth (defaults to 0), 
# x2 = maximum depth (defaults to maximum depth used in SRIM), layerdepth = array of bottom depths of target material 
# layers (if not specified, reads this info from TDATA.txt), output = indicator for what to output (either 'plot' or 
# 'excel', defaults to 'plot').
# assumptions: path given as string, a,b,c,d given as booleans, x1 & x2 given in angstrom, layerdepth given as an array
def TdamCompare(path, x1=None, x2=None, layerdepth=None, output='plot'):
    # check that input for output is valid
    if not (output == 'plot' or output == 'excel'):
        raise ValueError("Only output 'plot' or 'excel' is accepted")
    
    # get layer information from output files
    trim_in_path = os.path.join(path, "TDATA.txt") # specify path to TDATA.txt file within 'path' directory
    with open(trim_in_path, "r") as f:
        file = f.read()
    layerdata = np.array(srim.SRIM_Output()._read_target(file)) # get target information from TDATA.txt
    
    # if no layer depths are given, read them from TDATA.txt
    if layerdepth is None:
        # access layer depth data (what depth each layer ends at)
        layerdepth = np.array([]) # empty array to add layer depths to
        for layer in layerdata:
            layerdepth = np.append(layerdepth, layer["bottom_depth_A"])
        
    # if layerdepth given, convert to numpy array
    else:
        layerdepth = np.array(layerdepth)
        
    # convert layerdepth to nm from angstrom
    layerdepth = layerdepth*0.1
    
    # get required SRIm output files
    damage = srim.Phonons(path)
    Vac = srim.Vacancy(path)
    NoVac = srim.NoVacancy(path)
    
    # access the required data within output files - all angstrom units converted to nm, and eV converted to keV
    dambyion = damage.ions/100 # damage energy by ion
    dambyrec = damage.recoils/100 # damage energy by recoils
    vac = np.sum(Vac.vacancies, axis=1)*10 # vacancy profile
    novac = NoVac.number*10 # replacement profile
    depth = damage.depth*0.1 # depth bins used in SRIM - convert to nm from angstrom
    
    # if only 1 target layer, set layer depth to the last depth value
    if len(layerdepth) == 1:
        layerdepth = np.array([depth[-1]])
    
    # set minimum and maximum depth
    if x1 is None:
        x1 = 0
    else:
        x1 = x1*0.1 # convert to nm from angstrom
    if x2 is None:
        x2 = depth[-1]
    else:
        x2 = x2*0.1 # convert to nm from angstrom
    
    # check if selected values of x1 and x2 are valid
    if x2 > depth[-1]:
        raise ValueError("Selected x2 value was outside of SRIM calculated depth range")
    elif x1 < 0 or x2 < 0:
        raise ValueError("x1 and x2 must be positive")
    elif x1 >= x2:
        raise ValueError("x1 must be less than x2")
    
    # calculate damage energy using phonon method 1 (using phonons by ion and phonons by recoils)
    damphon1 = dambyion + dambyrec
    
    # calculate damage energy using phonon method 2 (only using phonons by recoils)
    damphon2 = dambyrec
    
    # calculate damage energy using energy conservation method
    # calculate total displacement profile
    displacements = vac + novac
    
    # restrict data to only portion of displacement profile within first layer
    subset = (depth <= layerdepth[0])
    damcons = displacements[subset]
    
    # multiply this section of the displacement profile by the local Tdam/v factor to convert to damage energy
    vactot, novactot = totDisplacements(path, 0, layerdepth[0]*10)
    v = vactot + novactot # total number of displacements
    ionizE, damE = totEnergyDep(path, 0, layerdepth[0]*10, method='cons')
    damcons = damcons*damE[0]/v
    
    # repeat for all subsequent layers, and append those sections of the damage energy profile
    for i in range(1,len(layerdepth)):
        # restrict data to only portion of displacement profile within current layer
        subset = ((depth > layerdepth[i-1]) & (depth <= layerdepth[i]))
        layer_damtot = displacements[subset]
        
        # calculate damage energy and total displacements
        vactot, novactot = totDisplacements(path, layerdepth[i-1]*10, layerdepth[i]*10)
        v = vactot + novactot # total number of displacements
        ionizE, damE = totEnergyDep(path, layerdepth[i-1]*10, layerdepth[i]*10, method='cons')
        
        # multiply the current layer's displacement profile by the conversion factor, and append it to damtot
        layer_damtot = layer_damtot*damE[0]/v
        damcons = np.append(damcons, layer_damtot)
        
    # restrict arrays to only include data between x1 and x2
    subset = ((depth >= x1) & (depth <= x2))
    damphon1 = damphon1[subset]
    damphon2 = damphon2[subset]
    damcons = damcons[subset]
    depth = depth[subset]
    
    # plot parameters (can be modified based on individual preferences):
    linewidth = 2 # width of lines connecting points
    markersize = 6 # size of point markers
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    linestyle = 'None' # type of lines connecting points (i.e. solid, dotted, none, etc.)
    color1 = 'purple' # colour for phonon method 1
    color2 = 'blue' # colour for phonon method 2
    color3 = 'red' # colour for energy conservation method
    marker1 = '^' # marker for phonon method 1
    marker2 = 'o' # marker for phonon method 2
    marker3 = 'D' # marker for energy conservation method
    
    # create plot if plot option selected
    if output == 'plot':
        fig, ax1 = plt.subplots(figsize=(12,8))
        ax1.plot(depth, damphon1, color=color1, linewidth=linewidth, linestyle=linestyle, marker=marker1, 
                       markersize=markersize, label="Phonon Method 1 (ion & recoils)")
        ax1.plot(depth, damphon2, color=color2, linewidth=linewidth, linestyle=linestyle, marker=marker2, 
                       markersize=markersize, label="Phonon Method 2 (only recoils)")
        ax1.plot(depth, damcons, color=color3, linewidth=linewidth, linestyle=linestyle, marker=marker3, 
                       markersize=markersize, label="Energy Conservation Method")
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                        direction='in')
        ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
        ax1.set_xlabel("Depth [nm]", fontsize=axlabsize)
        ax1.set_ylabel("Energy Deposition [keV/nm/ion]", fontsize=axlabsize)
        plt.legend(fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.55, 0))
        plt.show()
    
    # export data to excel if excel option selected
    elif output == 'excel':
        # convert arrays to a pandas data frame so it can be exported to excel
        data = np.empty((len(depth),4))
        data[:,0] = depth
        data[:,1] = damphon1
        data[:,2] = damphon2
        data[:,3] = damcons
        df = pd.DataFrame(data, columns=["Depth (nm)", "Phonon method 1 (keV/nm/ion)", "Phonon method 2 (keV/nm/ion)", 
                                         "Energy conservation method"])
        
        # in case function has run multiple times for same path, rename file to be created accordingly
        num = 0
        while True:
            # create filepath to export data frame to
            export_path = os.path.join(path, f"damage_compare_data({num}).xlsx")
            
            # check if current path name exists and continue looping if it does
            if os.path.exists(export_path):
                num = num + 1
                continue
            
            # if path name does not exist, export data frame to excel
            else:
                try:
                # Export to Excel
                    df.to_excel(export_path, index=False)
                    print("Data successfully exported to: ", export_path)
                except Exception as e:
                    print(f"Error exporting to Excel: {e}")
            
                # break out of loop once data has been exported
                break
    
    # calculate total damage energy using each method
    ionizE, damphon1_tot = totEnergyDep(path, x1*10, x2*10, method='phonons_1')
    ionizE, damphon2_tot = totEnergyDep(path, x1*10, x2*10, method='phonons_2')
    ionizE, damcons_tot = totEnergyDep(path, x1*10, x2*10, method='cons')
    
    # print total damage energy from each method
    print("Total damage energy (avg per ion) using Phonon Method 1 (ion & recoils) [keV]: ", damphon1_tot[0])
    print("Total damage energy (avg per ion) using Phonon Method 2 (only recoils) [keV]: ", damphon2_tot[0])
    print("Total damage energy (avg per ion) using Energy Conservation Method [keV]: ", damcons_tot[0])
    
    return None

#%% Advanced plotting functions

# function that automatically generates a plot of the collision distribution within the specifie depth range, projected 
# onto the XY or XZ plane using the SRIM Collision.txt output file. a secondary plot of the projection of the collision
# distribution on the x-axis is also generated.
# inputs: path = file path to folder containing the SRIM output text files, x1 = minimum depth value, x2 = maximum depth
# value, n1 = number of depth bins (defaults to 100), d2 = indicator for which second dimension to plot (either 'y'
# or 'z', defaults to 'z'), r = "radius" for second dimension (defaults to the maximum collision radius), n2 = number of 
# second dimension bins (defaults to the same as n1), ion_start = what ion index to start reading from, num_ions = 
# number of ions to plot (uses the first num_ions after ion_start from the SRIM simulation, defaults to using all of the 
# simulated ions after ion_start), coll_type = indicator for whether to plot only ion paths, or all target atom recoils 
# (defaults to only ion paths), scale = indicator for what type of scale to use for the colourbar (either 'linear' or
# 'log', defaults to 'linear').
# assumptions: path given as a string, COLLISON.txt file within 'path' directory, x1 & x2 are scalars and are given in 
# angstroms, r is a scalar and is given in units of angstroms, n1 & n2 are integers, ion_start and num_ions are given as
# integers.
def collisionPlotDepth(path, x1, x2, n1=100, d2='z', r=None, n2=None, num_ions=None, ion_start=0, coll_type='ions', 
                       scale='linear'):
    # get collision data and range data from SRIM output files
    coll = srim.Collision(path)
    Range = srim.Range(path)
    
    # access depth data
    depth = Range.depth*0.1 # convert to nm from angstrom
    
    # convert x1 and x2 to nm from angstrom
    x1 = x1*0.1
    x2 = x2*0.1
    
    # check if selected values of x1 and x2 are valid
    if x2 > depth[-1]:
        raise ValueError("Selected x2 value was outside of SRIM calculated depth range")
    elif x1 < 0 or x2 < 0:
        raise ValueError("x1 and x2 must be positive")
    elif x1 >= x2:
        raise ValueError("x1 must be less than x2")
    
    # check that num_ions is valid if specified
    if num_ions is not None:
        # check if the number of ions chosen is within the number of ions calculated by SRIM
        if not num_ions <= len(coll) - ion_start:
            raise ValueError("The selected range of ions was not within than that calculated by SRIM")
        
        # check if the number of ions chosen is greater than zero
        if not num_ions > 0:
            raise ValueError("The selected number of ions must be greater than zero")
    
    # get number of ions to caclulate if not specified
    else:
        num_ions = len(coll) - ion_start
    
    # set what data will be plotted (ion tracks or recoil cascades)
    if coll_type == 'ions':
        coll_type = 'PKA_positions'
    elif coll_type == 'recoils':
        coll_type = 'collision_positions'
    else:
        raise ValueError("Only coll_type 'ions' or 'recoils' is accepted")
    
    # check that specified scale is valid
    if not (scale == 'linear' or scale == 'log'):
        raise ValueError("Only scale type 'linear' or 'log' is accepted")
    
    # set the second dimension
    if d2 == 'y':
        d2 = 1
    elif d2 == 'z':
        d2 = 2
    else:
        raise ValueError("Only 'y' or 'z' is accepted as the second dimension to plot")
    
    # get data for the target atom recoils for the first ion (ion_start)
    recoils = coll[ion_start][coll_type]
    ions_used = 1
    
    # loop through the selected number of ions and append their recoil data
    for i in range(ion_start + 1, num_ions + ion_start):
        try:
            recoils = np.append(recoils, coll[i][coll_type], axis=0)
            ions_used = ions_used + 1
        except (IndexError, AttributeError):
            break
    
    # print number of ions that could be used (did not return errors)
    print("Number of ions used: ", ions_used)
    
    # convert recoil coordinates to nm from angstrom
    recoils = recoils*0.1
    
    # set n2 to n1 if no value was specified
    if n2 is None:
        n2 = n1
    
    # array to store number of recoils within each area element
    numColl = np.empty((n1, n2))
    
    # only use collision data between x1 and x2 - restrict recoils to this range
    subset = ((recoils[:,0] >= x1) & (recoils[:,0] <= x2))
    recoils = recoils[subset]
    
    # get the minimum and maximum second dimension
    if r is None:
        r = np.amax(np.abs(recoils[:,d2])) # find maximum d2 coordinate
        a = -r
        b = r
    else:
        a = -r*0.1 # minumum
        b = r*0.1 # maximum
    
    # dimension arrays - elements within these arrays are the boundaries of the area elements
    xs = np.linspace(x1, x2, n1+1)
    yzs = np.linspace(a, b, n2+1)
    
    # loop through all area elements and count the number of recoils that occurred within each
    for i in range(n1):
        for j in range(n2):
            subset = ((recoils[:,0] > xs[i]) & (recoils[:,0] <= xs[i+1]) & (recoils[:,d2] > yzs[j]) 
                      & (recoils[:,d2] <= yzs[j+1]))
            area_recs = recoils[subset]
            numColl[i,j] = area_recs[:,0].size
    
    # mask area elements within of numColl with 0 recoils for better plotting visualization
    numColl = ma.masked_where(numColl == 0, numColl)
    
    # make meshgrid for plot
    X, YZ = np.meshgrid(xs[:-1], yzs[:-1], indexing='ij')
    
    # project collision distribution onto x-axis for secondary plot
    numColl_x = np.sum(numColl, axis=1)
    
    # plot parameters (can be modified based on individual preferences):
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    linestyle = 'None'
    linewidth = 2 # width of lines connecting points
    markersize = 6 # size of point markers
    cmap = 'jet' # colourmap
    color = 'blue' # colour for x-axis projection plot
    marker = 'o' # marker for x-axis projection plot
    
    # 2D collision distribution plotting
    fig, ax1 = plt.subplots(figsize=(10,8))
    if scale == 'linear':
        ax = ax1.contourf(X, YZ, numColl, cmap=cmap, levels=200)
        cbar = plt.colorbar(ax)
    else:
        ax = ax1.contourf(X, YZ, numColl, cmap=cmap, levels=200, locator=ticker.LogLocator(subs='auto'))
        cbar = plt.colorbar(ax)
        cbar.set_ticks(ticker.LogLocator())
    cbar.set_label(label="Number of Recoils", fontsize=legendlabsize)
    cbar.ax.tick_params(labelsize=ticklabsize)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='in')
    ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
    ax1.set_xlabel("X [nm]", fontsize=axlabsize)
    if d2 == 1:
        ax1.set_ylabel("Y [nm]", fontsize=axlabsize)
    else:
        ax1.set_ylabel("Z [nm]", fontsize=axlabsize)
    plt.show()
    # 1D collision distribution (x-axis projection) plotting
    fig, ax1 = plt.subplots(figsize=(10,8))
    ax1.plot(xs[:-1], numColl_x, color=color, linewidth=linewidth, linestyle=linestyle, marker=marker, 
                   markersize=markersize)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='in')
    ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='in')
    ax1.set_xlabel("X [nm]", fontsize=axlabsize)
    ax1.set_ylabel("Number of recoils", fontsize=axlabsize)
    if scale == 'log':
        ax1.set_yscale('log')
    plt.show()
    
    return None


# function that automatically generates a plot of the collision distribution within the specified depth range, projected 
# onto the YZ plane using the SRIM Collision.txt output file.
# inputs: path = file path to folder containing the SRIM output text files, x1 = minimum depth value, x2 = maximum depth
# value, n = number of bins to use for the y and z dimensions (defaults to 100), r = "radius" for second dimension 
# (defaults to the maximum collision radius), ion_start = what ion index to start reading from, num_ions = number of ions 
# to plot (uses the first num_ions after ion_start from the SRIM simulation, defaults to using all of the simulated ions 
# after ion_start), coll_type = indicator for whether to plot only ion paths, or all target atom recoils (defaults to only
# ion paths), scale = indicator for what type of scale to use for the colourbar (either 'linear' or 'log', defaults to 
# 'linear').
# assumptions: path given as a string, COLLISON.txt file within 'path' directory, x1 & x2 are scalars and are given in 
# angstroms, r is a scalar and is given in units of angstroms, n1 & n2 are integers, ion_start and num_ions are given as
# integers.
def collisionPlotXsection(path, x1, x2, n=100, r=None, num_ions=None, ion_start=0, coll_type='ions', scale='linear'):
    # get collision data from SRIM output files
    coll = srim.Collision(path)
    Range = srim.Range(path)
    
    # access depth data
    depth = Range.depth*0.1 # convert to nm from angstrom
    
    # convert x1 and x2 to nm from angstrom
    x1 = x1*0.1
    x2 = x2*0.1
    
    # check if selected values of x1 and x2 are valid
    if x2 > depth[-1]:
        raise ValueError("Selected x2 value was outside of SRIM calculated depth range")
    elif x1 < 0 or x2 < 0:
        raise ValueError("x1 and x2 must be positive")
    elif x1 >= x2:
        raise ValueError("x1 must be less than x2")
    
    # check that num_ions is valid if specified
    if num_ions is not None:
        # check if the number of ions chosen is within the number of ions calculated by SRIM
        if not num_ions <= len(coll) - ion_start:
            raise ValueError("The selected range of ions was not within than that calculated by SRIM")
        
        # check if the number of ions chosen is greater than zero
        if not num_ions > 0:
            raise ValueError("The selected number of ions must be greater than zero")
    
    # get number of ions to calculate if not specified
    else:
        num_ions = len(coll) - ion_start
    
    # set what data will be plotted (ion tracks or recoil cascades)
    if coll_type == 'ions':
        coll_type = 'PKA_positions'
    elif coll_type == 'recoils':
        coll_type = 'collision_positions'
    else:
        raise ValueError("Only coll_type 'ions' or 'recoils' is accepted")
    
    # check that specified scale is valid
    if not (scale == 'linear' or scale == 'log'):
        raise ValueError("Only scale type 'linear' or 'log' is accepted")
    
    # get data for the target atom recoils for the first ion (ion_start)
    recoils = coll[ion_start][coll_type]
    ions_used = 1
    
    # loop through the selected number of ions and append their recoil data
    for i in range(ion_start + 1, num_ions + ion_start):
        try:
            recoils = np.append(recoils, coll[i][coll_type], axis=0)
            ions_used = ions_used + 1
        except (IndexError, AttributeError):
            break
    
    # print number of ions that could be used (did not return errors)
    print("Number of ions used: ", ions_used)
    
    # convert recoil coordinates to nm from angstrom
    recoils = recoils*0.1
    
    # array to store number of recoils within each area element
    numColl = np.empty((n, n))
    
    # only use collision data between x1 and x2 - restrict recoils to this range
    subset = ((recoils[:,0] >= x1) & (recoils[:,0] <= x2))
    recoils = recoils[subset]
    
    # get the minimum and maximum dimensions from radius
    if r is None:
        r = max(np.amax(np.abs(recoils[:,1])), np.amax(np.abs(recoils[:,2]))) # maximum radius of y and z dimensions
        a = -r
        b = r
    else:
        a = -r*0.1 # minumum
        b = r*0.1 # maximum
    
    # dimension arrays - elements within these arrays are the boundaries of the area elements
    yzs = np.linspace(a, b, n+1) # convert x1 & x2 to nm from angstroms
    
    # loop through all area elements and count the number of recoils that occurred within each
    for i in range(n):
        for j in range(n):
            subset = ((recoils[:,1] > yzs[i]) & (recoils[:,1] <= yzs[i+1]) & (recoils[:,2] > yzs[j]) 
                      & (recoils[:,2] <= yzs[j+1]))
            area_recs = recoils[subset]
            numColl[i,j] = area_recs[:,0].size
    
    # mask area elements within of numColl with 0 recoils for better plotting visualization
    numColl = ma.masked_where(numColl == 0, numColl)
    
    # make meshgrid for plot
    Y, Z = np.meshgrid(yzs[:-1], yzs[:-1], indexing='ij')
    
    # plot parameters (can be modified based on individual preferences):
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    cmap = 'jet' # colourmap
    
    # plotting
    fig, ax1 = plt.subplots(figsize=(10,8))
    if scale == 'linear':
        ax = ax1.contourf(Y, Z, numColl, cmap=cmap, levels=200)
        cbar = plt.colorbar(ax)
    else:
        ax = ax1.contourf(Y, Z, numColl, cmap=cmap, levels=200, locator=ticker.LogLocator(subs='auto'))
        cbar = plt.colorbar(ax)
        cbar.set_ticks(ticker.LogLocator())
    cbar.set_label(label="Number of Recoils", fontsize=legendlabsize)
    cbar.ax.tick_params(labelsize=ticklabsize)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='in')
    ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
    ax1.set_xlabel("Y [nm]", fontsize=axlabsize)
    ax1.set_ylabel("Z [nm]", fontsize=axlabsize)
    plt.show()
    
    return None


# function that automatically generates a plot of damage dose profile and a plot of ion range profile from multiple sets 
# of SRIM output files. also prints the mean ion density and mean damage dose using all points and only the highest and
# lowest points if the 'combined' option is selected.
# inputs: paths = 1D array where elements are file paths to folders containing SRIM output text files, labels = 1D array 
# where the elements are strings corresponding to the labels for the data in paths, fluence = ion fluence, x1 = minimum 
# depth value, x2 = maximum depth value, units = indicator for what units to use for ion density (either 'at_%' or 
# 'density_per_fluence', defaults to 'at_%'), plot_style = indicator for plotting the ion distribution and damage dose 
# profiles as single combined profiles (by summing the individual profiles), ref_layer = what layer to use as the 
# reference (zero) for the depth axis (defaults to the first layer), layerdepth = an object array where each element 
# corresponds to a path and is an array with the bottom depths of each target material layer for that path.
# assumptions: layer widths input into SRIM in angstrom, fluence and atomic density given in cm^-2 and cm^-3, path 
# given as a string, fluence given as scalar, size of paths and labels is the same.
def multiRangeAndDpa(paths, labels, fluence, x1=None, x2=None, units='at_%', plot_style='individual', ref_layer=1, 
                     layerdepth=None):
    # check if input for units is valid
    if not (units == 'at_%' or units == 'density_per_fluence'):
        raise ValueError("Only units 'at_%' or 'density_per_fluence' is accepted")
    
    # check if input for plot_style is valid
    if not (plot_style == 'individual' or plot_style == 'combined'):
        raise ValueError("Only plot_style 'individual' or 'combined' is accepted")
    
    # check if input for ref_layer is valid
    if ref_layer < 1:
        raise ValueError("ref_layer must be greater than 0")
    
    # loop through all paths to get layer data from output files
    layerdata = [] # empty array to add layer data to
    srim_output = srim.SRIM_Output()
    for i in range(len(paths)):
        trim_in_path = os.path.join(paths[i], "TDATA.txt") # specify path to TDATA.txt file within 'path' directory
        with open(trim_in_path, "r") as f:
            output = f.read()
        new_data = np.array(srim_output._read_target(output)) # get target information from TDATA.txt
        layerdata.append(new_data)
        f.close()
    
    # empty array to add layer depths to, rows correspond to paths, columns correspond to depths of each layer
    layerdepth_read = np.empty((len(layerdata)), dtype=object)
    
    # empty array to add layer densities to, rows correspond to paths, columns correspond to densities of each layer
    layerdensity = np.empty((len(layerdata)), dtype=object)
    
    # loop through all paths to get extract layer depth and density data from layerdata
    for i in range(len(layerdata)):
        # if no layer depths were specified, read them from TDATA.txt
        if layerdepth is None:
            # access layer depth data (what depth each layer ends at)
            layerdepth_read[i] = np.array([]) # empty array to add layer depths to
            for layer in layerdata[i]:
                layerdepth_read[i] = np.append(layerdepth_read[i], layer["bottom_depth_A"])
                
                # convert to nm from angstrom
                layerdepth_read[i] = layerdepth_read[i]*0.1
        
        # access layer density data
        layerdensity[i] = np.array([])
        for layer in layerdata[i]:
            layerdensity[i] = np.append(layerdensity[i], layer["atomic_density"])
    
    # if no layer depths were specified, store the read layer depths in layerdepth
    if layerdepth is None:
        layerdepth = layerdepth_read
        
    # if layer depths specified, convert to nm from angstrom
    else:
        for i in range(len(paths)):
            layerdepth[i] = layerdepth[i]*0.1
    
    # empty array to add range data to, rows correspond to paths, columns correspond to ion distrubution at each depth
    ionrange = np.empty((len(paths)), dtype=object)
    
    # empty array to add dmage dose data to
    dpa = np.empty((len(paths)), dtype=object)
    
    # empty array to add depth data to, rows correspond to paths, columns correspond to depth bins
    depth = np.empty((len(paths)), dtype=object)
    
    # extract data from all paths
    for i in range(len(paths)):
        # get range, vacancy, and replacement output files
        Range = srim.Range(paths[i])
        Vac = srim.Vacancy(paths[i])
        NoVac = srim.NoVacancy(paths[i])
        
        # access the required data within output files
        ionrange[i] = Range.ions # ion range distrubution
        depth[i] = Range.depth*0.1 # depth bins used in SRIM - convert to nm from angstrom
        vac = Vac.vacancies # array of vacancy profile for all target elements
        novac = NoVac.number # replacement profile
        
        # if only 1 target layer, set layer depth to last depth value
        if len(layerdensity[i]) == 1:
            layerdepth[i] = np.array([depth[i][-1]])
        
        # calculate total discplacements
        displacements = np.sum(vac, axis=1) + novac
        
        # determine the indices of the layer boundaries (index of the output data a layer ends and the next layer starts)
        dx = depth[i][1] - depth[i][0] # depth bins
        # convert the array of the depths of each target layer into the index at which each layer ends
        layerindex = layerdepth[i]//dx
        layerindex = layerindex.astype(int) # ensure layerindex entries are ints so they can be used as indices
        
        # calculate dpa, accounting for different layer atomic densities
        dpa[i] = np.empty(100) # initialize size of dpa[i] so slicing can be used
        # use 1D array indexing if current path only has 1 layer, otherwise use 2D indexing
        if len(layerdensity[i]) < 2:
            dpa[i][:layerindex[0]+1] = displacements[:layerindex[0]+1]*fluence/layerdensity[i]*1e8
        else:
            dpa[i][:layerindex[0]+1] = displacements[:layerindex[0]+1]*fluence/layerdensity[i][0]*1e8 # first layer
            for j in range(layerindex.size-1): # loop through all subsequent target layers to calculate the dpa in each
                dpa[i][layerindex[j]+1:layerindex[j+1]+1] = (displacements[layerindex[j]+1:layerindex[j+1]+1]
                                                          *fluence/layerdensity[i][j+1]*1e8)
        
        # convert to atomic percent if specified
        if units == 'at_%':
            # convert to atomic percent using 1D array indexing if current path only has 1 layer
            if len(layerdensity[i]) < 2:
                # convert ionrange data from ion density to atomic percent
                ionrange[i][:layerindex[0]+1] = ionrange[i][:layerindex[0]+1]*fluence*100/layerdensity[i]
            else:
                # convert ionrange data from ion density to atomic percent, accounting for different layer densities
                ionrange[i][:layerindex[0]+1] = ionrange[i][:layerindex[0]+1]*fluence*100/layerdensity[i][0]
                for j in range(layerindex.size-1): # loop through all subsequent target layers
                    ionrange[i][layerindex[j]+1:layerindex[j+1]+1] = (ionrange[i][layerindex[j]+1:layerindex[j+1]+1]
                                                              *fluence*100/layerdensity[i][j+1])
    
    # if option to plot range as range into a specific layer selected, modify ionrange and depth accordingly
    if not ref_layer == 1:
        # loop through all paths to modify ionrange and depth
        for i in range(len(paths)):
            # skip any paths where the number of layers is les than the specified reference layer
            if len(layerdepth[i]) < ref_layer:
                continue
            
            # restrict ionrange and dpa to not include data before reference layer
            subset = (depth[i] >= layerdepth[i][ref_layer-2])
            ionrange[i] = ionrange[i][subset]
            dpa[i] = dpa[i][subset]
            
            # restrict depth to not include data befoer reference layer, and subtract depth of start reference layer to 
            # allow ion range to be plotted as the range into the reference layer
            depth[i] = depth[i][subset] - layerdepth[i][ref_layer-2]
    
    # if option selected, sum all ion distribution and damage dose profiles into single profiles
    if plot_style == 'combined':
        # append all depth arrays into a single total depth array
        depth_tot = np.array([])
        for i in range(len(paths)):
            depth_tot = np.append(depth_tot, depth[i])
        
        # sort total depth array and remove duplicate depth values
        depth_tot = np.unique(depth_tot)
        
        # empty arrays to add interpolated data to for each path to
        ionrange_tot = np.empty((len(paths), len(depth_tot)))
        dpa_tot = np.empty((len(paths), len(depth_tot)))
        
        # interpolate data for all paths for the depth bins in depth_tot
        for i in range(len(paths)):
            # compute interpolation at each point invdividually (required due to how cubicSpline is written)
            for j in range(len(depth_tot)):
                ionrange_tot[i,j] = cubicSpline(depth[i], ionrange[i], depth_tot[j])
                dpa_tot[i,j] = cubicSpline(depth[i], dpa[i], depth_tot[j])
                
            # set all nan values to zero, to take care of trying to interpolate to a depth outside of the data range
            ionrange_tot[i] = np.nan_to_num(ionrange_tot[i], nan=0.0)
            dpa_tot[i] = np.nan_to_num(dpa_tot[i], nan=0.0)
        
        # sum the ionrange and dpa profiles from all paths into total profiles
        ionrange_tot = np.sum(ionrange_tot, axis=0)
        dpa_tot = np.sum(dpa_tot, axis=0)
    
    # restrict arrays to only include data between x1 and x2
    if plot_style == 'individual':
        # find maximum depth of for each path
        maxDepth = np.empty(len(paths))
        for i in range(len(paths)):
            maxDepth[i] = np.max(depth[i])
        
        # set minimum and maximum depth
        if x1 is None:
            x1 = 0
        else:
            x1 = x1*0.1 # convert to nm from angstrom
        if x2 is None:
            x2 = np.max(maxDepth)
        else:
            x2 = x2*0.1 # convert to nm from angstrom
        
        # check if selected values of x1 and x2 are valid
        if x2 > np.max(maxDepth):
            print("Selected x2 value was outside of SRIM calculated depth range")
            return None
        elif x1 < 0 or x2 < 0:
            print("x1 and x2 must be positive")
            return None
        elif x1 >= x2:
            print("x1 must be less than x2")
            return None
        
        # loop through all paths and restrict arrays to only include data between x1 and x2
        for i in range(len(paths)):
            subset = ((depth[i] >= x1) & (depth[i] <= x2))
            ionrange[i] = ionrange[i][subset]
            dpa[i] = dpa[i][subset]
            depth[i] = depth[i][subset]
            
    elif plot_style == 'combined':
        # set minimum and maximum depth
        if x1 is None:
            x1 = 0
        else:
            x1 = x1*0.1 # convert to nm from angstrom
        if x2 is None:
            x2 = depth_tot[-1]
        else:
            x2 = x2*0.1 # convert to nm from angstrom
        
        # check if selected values of x1 and x2 are valid
        if x2 > depth_tot[-1]:
            print("Selected x2 value was outside of SRIM calculated depth range")
            return None
        elif x1 < 0 or x2 < 0:
            print("x1 and x2 must be positive")
            return None
        elif x1 >= x2:
            print("x1 must be less than x2")
            return None
        
        # restrict arrays to only include data between x1 and x2
        subset = ((depth_tot >= x1) & (depth_tot <= x2))
        ionrange_tot = ionrange_tot[subset]
        dpa_tot = dpa_tot[subset]
        depth_tot = depth_tot[subset]
    
    # calculate means and errors for total profiles if combined option selected
    if plot_style == 'combined':
        # find mean value and error of both profiles (using all points)
        ionrange_mean1 = np.mean(ionrange_tot)
        dpa_mean1 = np.mean(dpa_tot)
        ionrange_error1 = np.max(ionrange_tot) - ionrange_mean1
        dpa_error1 = np.max(dpa_tot) - dpa_mean1
        
        # find mean value and error of both profiles (using only max and min values)
        ionrange_mean2 = (np.max(ionrange_tot) + np.min(ionrange_tot))/2
        ionrange_error2 = np.max(ionrange_tot) - ionrange_mean2
        dpa_mean2 = (np.max(dpa_tot) + np.min(dpa_tot))/2
        dpa_error2 = np.max(dpa_tot) - dpa_mean2
    
    # plot parameters (can be modified based on individual preferences):
    linewidth = 2 # width of lines connecting points
    markersize = 8 # size of point markers
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    linestyle = '--' # type of lines connecting points (i.e. solid, dotted, none, etc.)
    marker = 'o' # marker for all plots
    
    # ion range plotting
    fig, ax1 = plt.subplots(figsize=(12,8))
    if plot_style == 'combined':
        ax1.plot(depth_tot, ionrange_tot, linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=markersize)
        ax1.fill_between(depth_tot, ionrange_tot, 0, alpha=0.5)
        # print means and errors
        print(f"Mean ion density (using all points): {ionrange_mean1}+/-{ionrange_error1}")
        print(f"Mean damage dose (using all points): {dpa_mean1}+/-{dpa_error1}")
        print()
        print(f"Mean ion density (using only max and min): {ionrange_mean2}+/-{ionrange_error2}")
        print(f"Mean damage dose (using only max and min): {dpa_mean2}+/-{dpa_error2}")
    else:
        for i in range(len(paths)):
            ax1.plot(depth[i], ionrange[i], linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=markersize,
                     label=labels[i])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='in')
    ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
    ax1.set_xlabel("Depth [nm]", fontsize=axlabsize)
    if units == 'at_%':
        ax1.set_ylabel("Ion Density [at. %]", fontsize=axlabsize)
    else:
        ax1.set_ylabel("Ion Density [(Atoms/cm\u00b3)/(Atoms/cm\u00b2)]", fontsize=axlabsize)
    # add legend and show plot
    if plot_style == 'individual':
        plt.legend(fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.25, 0))
    plt.show()
    
    # damage dose plotting
    fig, ax1 = plt.subplots(figsize=(12,8))
    if plot_style == 'combined':
        ax1.plot(depth_tot, dpa_tot, linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=markersize)
        ax1.fill_between(depth_tot, dpa_tot, 0, alpha=0.5)
    else:
        for i in range(len(paths)):
            ax1.plot(depth[i], dpa[i], linewidth=linewidth, linestyle=linestyle, marker=marker, markersize=markersize,
                     label=labels[i])
    # ax1.plot(xfit, yfit, color='blue', linewidth=linewidth, linestyle='--')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='in')
    ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='in')
    ax1.set_xlabel("Depth [nm]", fontsize=axlabsize)
    ax1.set_ylabel("Damage Dose [dpa]", fontsize=axlabsize)
    # add legend and show plot
    if plot_style == 'individual':
        plt.legend(fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.25, 0))
    plt.show()
    
    return None


# function that generates a plot of the PKA recoil spectra for all SRIM simulations given in paths, on the same plot. 
# Uses the COLLISON.txt file from each path. Unless a lower minE is specified, the plots will only show the PKA recoil 
# spectra above 1 eV.
# inputs: paths = 1D array where the elements are file paths to a folder containing the SRIM output text files,
# labels = 1D array where the elements are strings corresponding to the labels for the data in paths, n = number of energy
# bins to use, ion_start = what ion index to start reading from, num_ions = number of of ions to plot (uses the first 
# num_ions from the SRIM simulation, defaults to using all of the simulated ions after ion_start), minE = minimum energy
# to plot (defaults to the lowest energy of a collision), maxE = maximum energy to plot (defaults to the highest energy of
# a collision), xscale = what scale to use for the energy binning and plot x-axis (either 'linear' or 'log', defaults to
# 'log'), yscale = what scale to use for the plot y-axis (either 'linear' or 'log', defaults to 'linear'), 
# graph_style = how to plot the recoil spectra (either 'line' or 'bar', defaults to 'bar'), spectra = what type of 
# spectra to calculate and plot (either 'cumulative' or 'total', defaults to 'cumulative'), full_range = indicator for
# generating a second plot of the energy, excelpath = path to an excel file with additional data to plot (defaults to not 
# plotting additional data).
# assumptions: path given as a string, COLLISON.txt file within 'path' directory (option must be selected before SRIM
# simulation is run to produce this file), x1 & x2 are scalars and are given in angstroms, n,num_ions,ion_start are 
# integers, size of paths and labels is the same.
def recoilSpectra(paths, labels, n=100, num_ions=None, ion_start=0, minE=None, maxE=None, graph_style='line', 
                  xscale='log', yscale='linear', spectra='cumulative', full_range=False, excelpath=None):
    # check that input for n is valid
    if n < 1:
        raise ValueError("n must be greater than 0")
    
    # check that input for num_ions is valid if specified
    if num_ions is not None:
        if not num_ions > 0:
            raise ValueError("The selected number of ions must be greater than zero")
    
    # check that input for graph_style is valid
    if not (graph_style == 'line' or graph_style == 'bar'):
        raise ValueError("Only graph_style 'line' or 'bar' is excepted")
    
    # check that input for xscale is valid
    if not (xscale == 'linear' or xscale == 'log'):
        raise ValueError("Only xscale 'linear' or 'log' is accepted")
    
    # check that input for yscale is valid
    if not (yscale == 'linear' or yscale == 'log'):
        raise ValueError("Only yscale 'linear' or 'log' is accepted")
    
    # check that input for spectra is valid
    if not (spectra == 'cumulative' or spectra == 'total'):
        raise ValueError("Only spectra 'cumulative' or 'total' is accepted")
    
    # check that input for full_range is valid
    if not (full_range == True or full_range == False):
        raise ValueError("Only full_range True or False is accepted")
    
    # empty array to store recoil energy data for each path
    rec_energies = np.empty((len(paths)), dtype=object)
    
    # extract data from all paths
    for i in range(len(paths)):
        # get collision output file
        coll = srim.Collision(paths[i])
        
        # get number of ions to calculate if not specified
        if num_ions is None:
            num_ions = len(coll) - ion_start
        
        # empty array to add recoil energy data to
        recE = np.empty([])
        
        # loop through all ions calculated and append their recoil energy data
        ions_used = 0
        for j in range(ion_start, num_ions + ion_start):
            try:
                recE = np.append(recE, coll[j]['PKA_energies'])
                ions_used = ions_used + 1
            except (IndexError, AttributeError):
                break
            
        # print number of ions that could be used (did not return errors)
        print(f"Number of ions used for path {i}: ", ions_used)
        
        # store recoil energies for the current path in rec_energies
        rec_energies[i] = recE
    
    # set maximum energy to plot if not specified
    if maxE == None:
        # find maximum recoil energy for all paths
        maxE = np.empty(len(paths))
        for i in range(len(paths)):
            maxE[i] = np.max(rec_energies[i])
        b = np.max(maxE)
    
    else:
        b = maxE
    
    # set minimum energy to plot if not specified
    if minE == None:
        # set to 1 eV
        a = 1
        
    else:
        a = minE
        
    # create array with boundaries of the energy bins
    if xscale == 'linear':
        energy = np.linspace(a, b, n+1) # n bins so n+1 bin boundaries
    elif xscale == 'log':
        energy = np.logspace(np.log10(a), np.log10(b), n+1) # n bins so n+1 bin boundaries
    
    # calculate widths for bar graph
    widths = np.diff(energy)
    
    # empty array to store number of recoils with each energy for each path
    numRec = np.zeros((len(paths), n))
    
    # loop through all paths to get recoil spectra
    for i in range(len(paths)):
        # loop through the energy bins and count the number of recoils that occurred within each
        for j in range(n):
            # calculate number of recoils based on the spectra type
            if spectra == 'total':
                subset = ((rec_energies[i] > energy[j]) & (rec_energies[i] <= energy[j+1]))
            elif spectra == 'cumulative':
                subset = (rec_energies[i] <= energy[j+1])
            energy_recs = rec_energies[i][subset]
            numRec[i,j] = energy_recs.size
        
            # for the first energy bin, manually sum the number of recoils at the minimum energy since these recoils are
            # skipped by the logic above (for the total spectra option)
            if j == 0 and spectra == 'total':
                subset = (rec_energies[i] == energy[j])
                energy_recs = rec_energies[i][subset]
                numRec[i,j] = numRec[i,j] + energy_recs.size
    
    # normalize recoil spectrum for each path by the total number of recoils for that path
    for i in range(len(paths)):
        numRec[i] = numRec[i]/rec_energies[i].size
        
        # also calculate and print what fraction of the total number of recoils will be shown in the plot
        subset = ((rec_energies[i] >= energy[0]) & (rec_energies[i] <= energy[-1]))
        energy_recs = rec_energies[i][subset]
        frac = energy_recs.size/rec_energies[i].size
        print(f"Fraction of total number of recoils shown in main plot for {labels[i]}: ", frac)
    
    # repeat above steps for entire energy range (for second plot) if full_range option enabled
    if full_range:
        # find maximum recoil energy for all paths
        maxE = np.empty(len(paths))
        for i in range(len(paths)):
            maxE[i] = np.max(rec_energies[i])
            
        # create array with boundaries of the energy bins
        if xscale == 'linear':
            energy_full = np.linspace(1, np.max(maxE), n+1) # n bins so n+1 bin boundaries
        elif xscale == 'log':
            energy_full = np.logspace(0, np.log10(np.max(maxE)), n+1) # n bins so n+1 bin boundaries
            
        # calculate widths for bar graph
        widths_full = np.diff(energy_full)
            
        # empty array to store number of recoils with each energy for each path
        numRec_full = np.zeros((len(paths), n))
            
        # loop through all paths to get recoil spectra for specified range
        for i in range(len(paths)):
            # loop through the energy bins and count the number of recoils that occurred within each
            for j in range(n):
                if spectra == 'total':
                    subset = ((rec_energies[i] > energy_full[j]) & (rec_energies[i] <= energy_full[j+1]))
                elif spectra == 'cumulative':
                    subset = (rec_energies[i] <= energy_full[j+1])
                energy_recs = rec_energies[i][subset]
                numRec_full[i,j] = energy_recs.size
                
                # for the first energy bin, manually sum the number of recoils at the minimum energy since these recoils 
                # are skipped by the logic above (for the total spectra option)
                if j == 0 and spectra == 'total':
                    subset = (rec_energies[i] == energy_full[j])
                    energy_recs = rec_energies[i][subset]
                    numRec_full[i,j] = numRec_full[i,j] + energy_recs.size
            
        # normalize recoil spectrum for each path by the total number of recoils for that path
        for i in range(len(paths)):
                numRec_full[i] = numRec_full[i]/rec_energies[i].size
    
    # read excel file if specified
    if excelpath is not None:
        x_energy = pd.read_excel(excelpath, usecols=[0], skiprows=0, header=None).squeeze()
        x_energy = pd.to_numeric(x_energy, errors='coerce')
        x_energy = np.array(x_energy.dropna())
        x_spectrum = pd.read_excel(excelpath, usecols=[1], skiprows=0, header=None).squeeze()
        x_spectrum = pd.to_numeric(x_spectrum, errors='coerce')
        x_spectrum = np.array(x_spectrum.dropna())
        
        # calculate widths to use for bar graph
        x_widths = np.diff(x_energy)
        
        # # restrict excel data to only include data within the specified energy range
        # subset = ((x_energy >= a) & (x_energy <= b))
        # x_energy_1 = x_energy[subset]
        # x_spectrum_1 = x_spectrum[subset]
    
    # plot parameters (can be modified based on individual preferences):
    linewidth = 2 # width of lines connecting points
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    opacity = 0.5 # opacity for bar graph
    linestyle = '-'
    
    # plotting for restricted energy range
    fig, ax1 = plt.subplots(figsize=(10,8))
    for i in range(len(paths)):
        if graph_style == 'bar':
            ax1.bar(energy[:-1], numRec[i], width=widths, label=labels[i], alpha=opacity, align='edge')
        elif graph_style == 'line':
            ax1.plot(energy[:-1], numRec[i], label=labels[i], linewidth=linewidth, linestyle=linestyle)
    if excelpath is not None:
        if graph_style == 'bar':
            ax1.bar(x_energy, x_spectrum, width=x_widths, label=labels[-1], alpha=opacity, align='edge')
        elif graph_style == 'line':
            ax1.plot(x_energy, x_spectrum, linewidth=linewidth, linestyle=linestyle, label=labels[-1])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='out')
    ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='out')
    ax1.set_xlabel("PKA Recoil Energy [eV]", fontsize=axlabsize)
    ax1.set_ylabel("Fraction of PKA Recoils", fontsize=axlabsize)
    if xscale == 'log':
        ax1.set_xscale('log')
    if yscale == 'log':
        ax1.set_yscale('log')
    # add legend and show plot
    if len(paths) > 1:
        plt.legend(fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.325, 0))
    plt.show()
    
    # plotting for the full energy range if full_range option enabled
    if full_range:
        fig, ax1 = plt.subplots(figsize=(10,8))
        for i in range(len(paths)):
            if graph_style == 'bar':
                ax1.bar(energy_full[:-1], numRec_full[i], width=widths_full, label=labels[i], alpha=opacity, align='edge')
            elif graph_style == 'line':
                ax1.plot(energy_full[:-1], numRec_full[i], label=labels[i], linewidth=linewidth, linestyle=linestyle)
        if excelpath is not None:
            if graph_style == 'bar':
                ax1.bar(x_energy, x_spectrum, width=x_widths, label=labels[-1], alpha=opacity, align='edge')
            elif graph_style == 'line':
                ax1.plot(x_energy, x_spectrum, linewidth=linewidth, linestyle=linestyle, label=labels[-1])
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                        direction='out')
        ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='out')
        ax1.set_xlabel("PKA Recoil Energy [eV]", fontsize=axlabsize)
        ax1.set_ylabel("Fraction of PKA Recoils", fontsize=axlabsize)
        if xscale == 'log':
            ax1.set_xscale('log')
        if yscale == 'log':
            ax1.set_yscale('log')
        # add legend and show plot
        if len(paths) > 1:
            plt.legend(fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.325, 0))
        plt.show()
    
    return None


# function that generates a plot of the weighted PKA recoil spectra for all SRIM simulations given in paths, on the same 
# plot. Uses the COLLISON.txt file from each path.
# inputs: paths = 1D array where the elements are file paths to a folder containing the SRIM output text files,
# labels = 1D array where the elements are strings corresponding to the labels for the data in paths, n = number of energy
# bins to use, ion_start = what ion index to start reading from, num_ions = number of of ions to plot (uses the first 
# num_ions from the SRIM simulation, defaults to using all of the simulated ions after ion_start), minE = minimum energy
# to plot (defaults to the lowest energy of a collision), maxE = maximum energy to plot (defaults to the highest energy of
# a collision), xscale = what scale to use for the energy binning and plot x-axis (either 'linear' or 'log', defaults to
# 'log'), yscale = what scale to use for the plot y-axis (either 'linear' or 'log', defaults to 'linear'), 
# graph_style = how to plot the recoil spectra (either 'line' or 'bar', defaults to 'line'), spectra = what type of 
# spectra to calculate and plot (either 'cumulative' or 'total', defaults to 'cumulative'), excelpath = path to an excel
# file with additional data to plot (defaults to not plotting additional data).
# assumptions: path given as a string, COLLISON.txt file within 'path' directory (option must be selected before SRIM
# simulation is run to produce this file), maxE and minE are scalars given in eV, n,num_ions,ion_start are integers, 
# size of paths and labels is the same.
def weightedRecoilSpectra(paths, labels, n=100, num_ions=None, ion_start=0, minE=None, maxE=None, graph_style='line', 
                          xscale='log', yscale='linear', spectra='cumulative', excelpath=None):
    # check that input for n is valid
    if n < 1:
        raise ValueError("n must be greater than 0")
    
    # check that input for num_ions is valid if specified
    if num_ions is not None:
        if not num_ions > 0:
            raise ValueError("The selected number of ions must be greater than zero")
    
    # check that input for graph_style is valid
    if not (graph_style == 'line' or graph_style == 'bar'):
        raise ValueError("Only graph_style 'line' or 'bar' is excepted")
    
    # check that input for xscale is valid
    if not (xscale == 'linear' or xscale == 'log'):
        raise ValueError("Only xscale 'linear' or 'log' is accepted")
    
    # check that input for yscale is valid
    if not (yscale == 'linear' or yscale == 'log'):
        raise ValueError("Only yscale 'linear' or 'log' is accepted")
    
    # check that input for spectra is valid
    if not (spectra == 'cumulative' or spectra == 'total'):
        raise ValueError("Only spectra 'cumulative' or 'total' is accepted")
    
    # empty array to store recoil energy data for each path
    rec_energies = np.empty((len(paths)), dtype=object)
    
    # empty array to store recoil displacement data for each path
    rec_disps = np.empty((len(paths)), dtype=object)
    
    # extract data from all paths
    for i in range(len(paths)):
        # get collision output file
        coll = srim.Collision(paths[i])
        
        # get number of ions to calculate if not specified
        if num_ions is None:
            num_ions = len(coll) - ion_start
        
        # empty array to add recoil energy data to
        recE = np.array([])
        
        # empty array to add recoil displacement data to
        recD = np.array([])
        
        # loop through all ions calculated and append their recoil energy data
        ions_used = 0
        for j in range(ion_start, num_ions + ion_start):
            try:
                recE = np.append(recE, coll[j]['PKA_energies'])
                recD = np.append(recD, coll[j]['PKA_displacements'])
                ions_used = ions_used + 1
            except (IndexError, AttributeError):
                break
            
        # print number of ions that could be used (did not return errors)
        print(f"Number of ions used for path {i}: ", ions_used)
        
        # store recoil energies and displacements for the current path in rec_energies and rec_disps respectively
        rec_energies[i] = recE
        rec_disps[i] = recD
    
    # set maximum energy to plot if not specified
    if maxE == None:
        # find maximum recoil energy for all paths
        maxE = np.empty(len(paths))
        for i in range(len(paths)):
            maxE[i] = np.max(rec_energies[i])
        b = np.max(maxE)
    
    else:
        b = maxE
    
    # set minimum energy to plot if not specified
    if minE == None:
        # set minimum energy to 1 eV
        a = 1
        
    else:
        a = minE
        
    # create array with boundaries of the energy bins
    if xscale == 'linear':
        energy = np.linspace(a, b, n+1) # n bins so n+1 bin boundaries
    elif xscale == 'log':
        energy = np.logspace(np.log10(a), np.log10(b), n+1) # n bins so n+1 bin boundaries
    
    # calculate widths for bar graph
    widths = np.diff(energy)
    
    # empty array to store number of recoils with each energy for each path
    numRec = np.zeros((len(paths), n))
    
    # empty array to store number of displacements for each PKA energy bin, for each path
    numDisps = np.zeros((len(paths), n))
    
    # loop through all paths to get data for weighted recoil spectra
    for i in range(len(paths)):
        # loop through the energy bins and count the number of recoils and displacements that occurred within each
        for j in range(n):
            if spectra == 'total':
                subset = ((rec_energies[i] > energy[j]) & (rec_energies[i] <= energy[j+1]))
            elif spectra == 'cumulative':
                subset = (rec_energies[i] <= energy[j+1])
            energy_recs = rec_energies[i][subset]
            disp_recs = rec_disps[i][subset]
            numRec[i,j] = energy_recs.size
            numDisps[i,j] = np.sum(disp_recs)
        
            # for the first energy bin, manually sum the number of recoils and displacements at the minimum energy since 
            # these recoils are skipped by the logic above (for the total spectra option)
            if j == 0 and spectra == 'total':
                subset = (rec_energies[i] == energy[j])
                energy_recs = rec_energies[i][subset]
                disp_recs = rec_disps[i][subset]
                numRec[i,j] = numRec[i,j] + energy_recs.size
                numDisps[i,j] = numDisps[i,j] + np.sum(disp_recs)
    
    # empty array to store weighted recoil spectra data
    weightedSpectra = np.empty((len(paths), n))
    
    # multiply number of recoils and displacements by each recoil energy together and normalize by the total number of
    # recoils an ddisplacements for that path to get the weighted recoil spectra
    for i in range(len(paths)):
        weightedSpectra[i] = numRec[i]*numDisps[i]/(rec_energies[i].size*np.sum(rec_disps[i]))
        
        # also calculate and print what fraction of the total number of recoils will be shown in the plot
        subset = ((rec_energies[i] >= energy[0]) & (rec_energies[i] <= energy[-1]))
        energy_recs = rec_energies[i][subset]
        frac = energy_recs.size/rec_energies[i].size
        print(f"Fraction of total number of recoils shown in main plot for {labels[i]}: ", frac)
    
    # read excel file if specified
    if excelpath is not None:
        x_energy = pd.read_excel(excelpath, usecols=[0], skiprows=0, header=None).squeeze()
        x_energy = pd.to_numeric(x_energy, errors='coerce')
        x_energy = np.array(x_energy.dropna())
        x_spectrum = pd.read_excel(excelpath, usecols=[1], skiprows=0, header=None).squeeze()
        x_spectrum = pd.to_numeric(x_spectrum, errors='coerce')
        x_spectrum = np.array(x_spectrum.dropna())
        x_dpa = pd.read_excel(excelpath, usecols=[2], skiprows=0, header=None).squeeze()
        x_dpa = pd.to_numeric(x_dpa, errors='coerce')
        x_dpa = np.array(x_dpa.dropna())
        
        # calculate widths to use for bar graph
        x_widths = np.diff(x_energy)
        
        # convert to weighted spectrum
        totdpa = np.sum(x_dpa)
        for i in range(len(x_energy)):
            x_spectrum[i] = x_spectrum[i]*np.sum(x_dpa[:i+1])/totdpa
        
        # # restrict excel data to only include data within the specified energy range
        # subset = ((x_energy >= a) & (x_energy <= b))
        # x_energy = x_energy[subset]
        # x_spectrum = x_spectrum[subset]
        # x_dpa = x_dpa[subset]
    
    # plot parameters (can be modified based on individual preferences):
    linewidth = 2 # width of lines connecting points
    majorlength = 8 # length of major tick marks
    minorlength = 4 # length of minor tick marks
    tickwidth = 2 # width of tick marks
    ticklabsize = 16 # font size of tick labels
    axlabsize = 22 # font size of axis labels
    legendlabsize = 18 # font size of legend labels
    opacity = 0.5 # opacity for bar graphs
    linestyle = '-'
    
    # plotting
    fig, ax1 = plt.subplots(figsize=(10,8))
    for i in range(len(paths)):
        if graph_style == 'bar':
            ax1.bar(energy[:-1], weightedSpectra[i], width=widths, label=labels[i], alpha=opacity, align='edge')
        elif graph_style == 'line':
            ax1.plot(energy[:-1], weightedSpectra[i], linewidth=linewidth, linestyle=linestyle, label=labels[i])
    if excelpath is not None:
        if graph_style == 'bar':
            ax1.bar(x_energy, x_spectrum, width=x_widths, label=labels[-1], alpha=opacity, align='edge')
        elif graph_style == 'line':
            ax1.plot(x_energy, x_spectrum, linewidth=linewidth, linestyle=linestyle, label=labels[-1])
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis='both', which='major', length=majorlength, width=tickwidth, labelsize=ticklabsize, 
                    direction='out')
    ax1.tick_params(axis='both', which='minor', length=minorlength, width=tickwidth, direction='out')
    ax1.set_xlabel("PKA Recoil Energy [eV]", fontsize=axlabsize)
    ax1.set_ylabel("Weighted Fraction of PKA Recoils", fontsize=axlabsize)
    if xscale == 'log':
        ax1.set_xscale('log')
    if yscale == 'log':
        ax1.set_yscale('log')
    # add legend and show plot
    plt.legend(fontsize=legendlabsize, loc='lower right', bbox_to_anchor=(1.45, 0))
    plt.show()
    
    return None
