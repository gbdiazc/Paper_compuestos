# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:22:26 2022

@author: gbdiaz
"""
import matplotlib.pyplot as plt    
import numpy as np
import matplotlib.colors as mcolors
colors = ["red","blue","green","magenta","cyan","purple","darkred","darkcyan","deeppink"]


def plot_v(feasible,imname,name):
    fs = 28
    plt.figure(figsize=(18,12))
    plt.tick_params( axis='both')
    plt.tick_params( which='major', direction = 'in', length=fs-2, width=1.2, colors='k', left=1, right=1, top =1)
    plt.tick_params( which='minor', direction = 'in', length=fs-2, width=1.2, colors='k',  left=1, labelbottom=False)
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs }
    plt.rc('font', **font)
    for i in range(9):
        plt.plot(i*np.ones(len(feasible[i])),feasible[i],"*",markersize = fs-6, color = colors[i])
        
    # plt.grid(linewidth=1)
    plt.xlabel("$v$", fontsize=fs+2)
    plt.ylabel("Value", fontsize=fs+2)
    plt.title(name, fontsize=fs+4)
    plt.tight_layout()
    plt.savefig(imname)
    print("Plot figure saved in: ",imname)
    
def plot_OF(X_obj_f,imname,name):
    fs = 28
    plt.figure(figsize=(18,12))
    plt.tick_params( axis='both')
    plt.tick_params( which='major', direction = 'in', length=fs-2, width=1.2, colors='k', left=1, right=1, top =1)
    plt.tick_params( which='minor', direction = 'in', length=fs-2, width=1.2, colors='k',  left=1, labelbottom=False)
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs }
    plt.rc('font', **font)
    plt.plot(range(len(X_obj_f)),X_obj_f,"*",markersize = fs-6 )
    #plt.grid(linewidth=1)
    plt.title(name, fontsize=fs+2)
    plt.xlabel("Number of run", fontsize=fs+2)
    plt.ylabel("Objective function", fontsize=fs+4)
    plt.tight_layout()
    plt.savefig(imname)
    print("Plot figure saved in: ",imname)
    
def plot_hists(curves,imname,name):
    x_max, x_min =  12, 0
    y_max, y_min = 50,0
    n_bins = 30
    fs = 24

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, figsize=(20,16), sharey=True)
    for i in range(6):
        exec('ax{}.tick_params( axis="x", labelbottom=False)'.format(i+1))
        
    for i in range(9):
        exec('ax{}.tick_params( which="major", direction = "in", length=11, width=1.2, colors="k", left=1, right=1, top =1)'.format(i+1))
        exec('ax{}.tick_params( which="minor", direction = "in", length=10, width=1.2, colors="k",  left=1)'.format(i+1))
    ax1.tick_params( axis='y', labelsize=fs)
    ax4.tick_params( axis='y', labelsize=fs)
    ax7.tick_params( axis='y', labelsize=fs)
    ax7.tick_params( axis='x', labelsize=fs)
    ax8.tick_params( axis='x', labelsize=fs)
    ax9.tick_params( axis='x', labelsize=fs)

    for i in range(9):
        exec('ax{}.set_xlim(x_min, x_max)'.format(i+1))
        exec('ax{}.set_ylim(y_min,y_max)'.format(i+1))
        exec('ax{}.set_xlabel("$v_{}$", fontsize=fs)'.format(i+1,i+1))
        exec('ax{}.tick_params( which="major", direction = "in", left=1, right=1, top =1)'.format(i+1))
        exec('ax{}.hist(curves["v{}"], n_bins, histtype="step", stacked=True, fill=False,color = colors[i], linewidth=2)'.format(i+1,i+1))
        #exec('ax{}.grid(linewidth=1.5)'.format(i+1))
    ax1.set_ylabel("$bins$", fontsize=fs)
    ax4.set_ylabel("$bins$", fontsize=fs)
    ax7.set_ylabel("$bins$", fontsize=fs)
    fig.suptitle(name, fontsize=fs+4)
    plt.tight_layout()
    plt.savefig(imname)
    print("Plot figure saved in: ",imname)    
    
def plot_sols(curves,imname,name):
    x_max, x_min =  12, 0
    y_max, y_min = 0, -15
    fs = 24

    font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs}
    plt.rc('font', **font)

    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, figsize=(20,16), sharey=True)
    for i in range(6):
        exec('ax{}.tick_params( axis="x", labelbottom=False)'.format(i+1))
        
    for i in range(9):
        exec('ax{}.tick_params( which="major", direction = "in", length=11, width=1.2, colors="k", left=1, right=1, top =1)'.format(i+1))
        exec('ax{}.tick_params( which="minor", direction = "in", length=10, width=1.2, colors="k",  left=1)'.format(i+1))
    ax1.tick_params( axis='y', labelsize=fs)
    ax4.tick_params( axis='y', labelsize=fs)
    ax7.tick_params( axis='y', labelsize=fs)
    ax7.tick_params( axis='x', labelsize=fs)
    ax8.tick_params( axis='x', labelsize=fs)
    ax9.tick_params( axis='x', labelsize=fs)
    cmap, norm = mcolors.from_levels_and_colors([0, 2, 5, 6], ['red', 'green', 'blue'])

    for i in range(9):
        exec('ax{}.set_xlim(x_min, x_max)'.format(i+1))
        exec('ax{}.set_ylim(y_min,y_max)'.format(i+1))
        exec('ax{}.set_xlabel("$v_{}$", fontsize=fs)'.format(i+1,i+1))
        exec('ax{}.set_ylabel("$OFN$", fontsize=fs)'.format(i+1))
        exec('ax{}.tick_params( which="major", direction = "in", left=1, right=1, top =1)'.format(i+1))
#        exec('ax{}.plot(curves["v{}"], curves["OF"],"+",markersize = 20, color = cmap)'.format(i+1,i+1))
        exec('ax{}.plot(curves["v{}"], curves["OF"],"*",markersize = 20, color = colors[i])'.format(i+1,i+1))
        exec('ax{}.grid(linewidth=1.5)'.format(i+1))
        
    fig.suptitle(name, fontsize=fs+8)
    plt.tight_layout()
    plt.savefig(imname)
    print("Plot figure saved in: ",imname)    
    
def plot_corr(curves_sols,imname,name):
    x_max, x_min =  12, -5
    y_max, y_min = 15, -5
    fs = 24
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs }
    plt.rc('font', **font)

    fig, axis = plt.subplots(9,9, figsize=(20,20),sharex='col', sharey='row')#, gridspec_kw={'hspace': 0.1, 'wspace': 0.1},
    for j in np.arange(1,9):
        axis[0,j].axis('off')  
        for i in np.arange(j+1,9):
            axis[j,i].axis('off')                 

    for i in range(9):
        for j in range(9):
            axis[i,j].tick_params( axis='both')
            axis[i,j].tick_params( which='major', direction = 'in', length=1, width=1.2, colors='k', left=1, right=1, top =1)
            axis[i,j].tick_params( which='minor', direction = 'in', length=1, width=1.2, colors='k',  left=1, labelbottom=False)
    for j in np.arange(0,9):
        for i in range(j,9):
            axis[i,0].set_ylabel("$v_"+str(i+1)+"$")
            axis[i,j].set_ylim(y_min,y_max)
            axis[i,j].set_xlim(x_min,x_max)    
            axis[i,j].grid(linewidth=0.85)
            axis[i,j].scatter(curves_sols["v"+str(i+1)],curves_sols["v"+str(j+1)],c=curves_sols["OF"], marker = "o", cmap="plasma", s=200, vmin=-12, vmax=0)
            axis[8,j].set_xlabel("$v_"+str(j+1)+"$")
    im = axis[8,8].scatter(curves_sols["v"+str(i+1)],curves_sols["v"+str(j+1)],c=curves_sols["OF"], marker = "o", cmap="plasma", s=200, vmin=-12, vmax=0)

    cb_ax = fig.add_axes([0.80, 0.41, 0.02,0.4])
    cbar = fig.colorbar(im, cax=cb_ax, orientation='vertical',label=r"$OFN$")
    fig.suptitle(name, fontsize=fs+4)
    plt.tight_layout()
    plt.savefig(imname)
    print("Plot figure saved in: ",imname)    
    