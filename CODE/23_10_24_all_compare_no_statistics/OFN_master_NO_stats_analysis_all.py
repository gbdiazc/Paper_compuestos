# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:54:52 2022

@author: gbdiaz
"""

from functions import *
from functions_plots_stat import *
import matplotlib.pyplot as plt   
import pandas as pd
import time 
import pickle
import os
import sys

# =============================================================================
# --------------------------  Initial parameters   ----------------------------
# -------------------------------  Search   -----------------------------------
# =============================================================================
Compuestos = True
components = 14
TBEv = np.array([341.88, 353.93, 353.24, 371.57, 391.95, 383.79, 398.83, 
                 424.30, 409.35, 423.97, 451.12, 447.30, 517.83, 469.08])
# [RVM, RMW, RCH, RTB]
cases = [[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[1,1,1,1]]
cases = [[1,1,1,0]]
IFG    = 1
iCEOS  = 2
compuestos = {0:'CH3', 1:'CH2', 2:'CH(CHAIN)', 3:'C(CHAIN)',
       4:'ACH', 5:'AC', 6: 'CH2(CYCLIC)', 7:'CH(CYCLIC)', 8:'C(CYCLIC)',9:'Fobj',
       10:'ln(Fobj)'}
Compuestos_n = ("Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
                "Indane", "Decane", "1-Methylnaphthalene", "Undecane")
dir_df_ex = "DATA/"
comp = []
df_all = []
sol_OF_all_C =[]
sols_data =[]
TBE_exp    = False
HCE_exp    = True

eps=2
for component in np.arange(0,components):
    print("Compuesto: ", component)
    if TBE_exp:
        TBE = TBEv[component]
    
    for case in cases:
        RVM = case[0]
        RMW = case[1]
        RCH = case[2]
        RTB = case[3]
        print("RVM : ",RVM)
        print("RMW : ",RMW)

        print("RCH : ",RCH)
        print("RTB : ",RTB) 
        df_datos = dir_df_ex+ "Critical.xlsx"

        if TBE_exp:
            dir_name = "RESULTS/Compuestos_TB_exp_" 
            if HCE_exp==True:
                sheet_n = "RHC_exp_TB_exp"
                dir_name += "RHC_exp/"
            else:
                dir_name += "RHC_corr/"
                sheet_n = "RHC_corr_TB_exp"

        else:
            dir_name = "RESULTS/Compuestos_TB_corr_"
            if HCE_exp==True:
                dir_name += "RHC_exp/"
                sheet_n = "RHC_exp_TB_corr"
            else:
                sheet_n = "RHC_corr_TB_corr"
                dir_name += "RHC_corr/"
        if Compuestos:
            with open(dir_name+"/parameters_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component)+".pkl", 'rb') as f:
                parameters = pickle.load(f)
        else:
            with open(dir_name+"/parameters_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component)+".pkl", 'rb') as f:
                parameters = pickle.load(f)
            
        n_dims = parameters['n_dims']
        S_runs = parameters['S_runs']
        runs   = parameters['runs']
        IFG    = parameters['IFG']
        iCEOS  = parameters['iCEOS']
        RVM = parameters['RVM']
        RMW = parameters['RMW']
        RCH = parameters['RCH']
        RTB = parameters['RTB'] 
        #components = parameters['components'] 
        #Compuestos = parameters['Compuestos'] 

        df_data = pd.read_excel(df_datos, sheet_name="Critical")
        df_sols = pd.read_excel(df_datos, sheet_name="Sols")

        l_runs  = len(runs)
  
        #%%
        if Compuestos:
            dir_plots = "PLOTS/Plots_compuestos_"
            if TBE_exp:
                dir_plots += "TB_exp_"
                if HCE_exp:
                    dir_plots +=  "RHC_exp/" 
                else:
                    dir_plots += "RHC_corr/"
            else:
                dir_plots += "TB_corr_"
                if HCE_exp:
                    dir_plots += "RHC_exp/"
                else:
                    dir_plots += "RHC_corr/"
        else:
            dir_plots = "PLOTS/Plots_componentes_"
            if TBE_exp:
                dir_plots += "_exp/"
            else:
                dir_plots += "_corr/"
        if not os.path.exists(dir_plots):
            os.makedirs(dir_plots)
        
        
        OF_s    = np.empty((l_runs,S_runs))
        OF_min  = np.empty((l_runs,1))
        OF_max  = np.empty((l_runs,1))
        OF_std  = np.empty((l_runs,1))
        OF_mean  = np.empty((l_runs,1))
        FI_s    = np.empty((l_runs,S_runs))
        FI_min  = np.empty((l_runs,1))
        FI_max  = np.empty((l_runs,1))
        FI_std  = np.empty((l_runs,1))
        FI_mean  = np.empty((l_runs,1))
        v_min   = np.empty((l_runs,n_dims))
        FOc = []
        vc= []
        
        csv_file = dir_plots +'Datos.csv'
        data = {'C':[],'runs':[],'Av FI':[],'max OF':[],'min OF':[],'Av OF':[],'Std OF':[],'$v_1$':[],'$v_2$':[],
                '$v_3$':[],'$v_4$':[],'$v_5$':[],'$v_6$':[],'$v_7$':[],'$v_8$':[],'$v_9$':[]}
        df = pd.DataFrame(data)
        
        
        #for component in [0]:#components:#[11]:#
        print(component)
        
        if Compuestos:
            dir_merged = dir_name+"/MergedOuts_"+str(component+1)+"/"
        else:
            dir_merged = dir_name+"/MergedOuts_"+str(component+1)+"/"
    
        #dir_merged = "MergedOuts_c_"+str(component+1)+"/"
        for var in ['OF','MW','gamma','RHC','TB','FI','v','time','OF_all']:
            exec('file_'+var+'= dir_merged + "mpi_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_'+var+'.npy"')    
            #exec('print(file_'+var+')')
            if var == 'OF_all':
                exec('sol_'+var+ '=np.load(file_'+var+')')
                sol_OF_all_C.append(sol_OF_all)
            else:
                exec('sol_'+var+ '=np.load(file_'+var+')')
        
        for run_c in np.arange(len(runs)):
     
            
            OF_min[run_c]  = np.min(sol_OF[run_c,:])
            OF_max[run_c]  = np.max(sol_OF[run_c,:])
            OF_s[run_c]   = sol_OF[run_c,:]
            FI_s[run_c]    = sol_FI[run_c]
            FI_min[run_c]  = np.min(sol_FI[run_c,:])
            FI_max[run_c]  = np.max(sol_FI[run_c,:])
            FI_mean[run_c] = np.mean(sol_FI[run_c,:])
            FI_std[run_c]  = np.std(sol_FI[run_c,:])
            OF_mean[run_c] = np.mean(sol_OF[run_c,:])
            OF_std[run_c]  = np.std(sol_OF[run_c,:])
            min_OF_idx = np.argmin(sol_OF[run_c,:])
            v_min[run_c,:]  = sol_v[run_c,min_OF_idx,:]
            
            #TB_py  = CrTB_f(TBE,IFG,sol_v[run_c,min_OF_idx,:])
            if TBE_exp== False:
                TBE = TBS_MG(v_min,IFG)
            TB_py  = TB_f(sol_v[run_c,min_OF_idx,:],IFG)
            Pc_py  = PCGC_f(IFG,sol_v[run_c,min_OF_idx,:])
            Tc_py  = TCGC_f(IFG,sol_v[run_c,min_OF_idx,:],TBE)
            Ac_py =  ACGC_f(TB_py, Pc_py, Tc_py)
            Tb_err_py = 100*np.abs(TB_py-df_data["Tb"][component])/np.abs(df_data["Tb"][component])
            Pc_err_py = 100*np.abs(Pc_py-df_data["Pc"][component])/np.abs(df_data["Pc"][component])
            Tc_err_py = 100*np.abs(Tc_py-df_data["Tc"][component])/np.abs(df_data["Tc"][component])
            Ac_err_py = 100*np.abs(Ac_py-df_data["Omega"][component])/np.abs(df_data["Omega"][component])


                
            data_1 =  {'C':int(component+1),'runs':runs[run_c],
                       'max OF':np.max(sol_OF[run_c,:]),
                       'min OF':np.min(sol_OF[run_c,:]),
                       'Av OF':-np.mean(np.abs(sol_OF[run_c,:])),
                       'Std OF':-np.std(np.abs(sol_OF[run_c,:])),
                       'Av FI':np.mean(sol_FI[run_c,:]),
                       '$v_1$':sol_v[run_c,min_OF_idx,0],'$v_2$':sol_v[run_c,min_OF_idx,1],
                       '$v_3$':sol_v[run_c,min_OF_idx,2],'$v_4$':sol_v[run_c,min_OF_idx,3],
                       '$v_5$':sol_v[run_c,min_OF_idx,4],'$v_6$':sol_v[run_c,min_OF_idx,5],
                       '$v_7$':sol_v[run_c,min_OF_idx,6],'$v_8$':sol_v[run_c,min_OF_idx,7],
                       '$v_9$':sol_v[run_c,min_OF_idx,8]}
            df = df.append(data_1, ignore_index = True)
        df_all.append(df)
        # =============================================================================
        # --------------------------------------  OF   --------------------------------
        # =============================================================================    
        name = "Objective function, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
        name = ""
        imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component+1)+".png"  

        fs =20
        plt.figure(figsize=(9,6))
        for i in range(len(runs)):
            for j in range(np.shape(sol_OF)[1]):
                
                if j == np.shape(sol_OF)[1]-1:
                    plt.plot(runs[i],sol_OF[i,j],'*',color=colors[i],label = i)
                else:
                    plt.plot(runs[i],sol_OF[i,j],'*',color=colors[i])
        #plt.title(name, fontsize=fs+2)
        plt.xlabel("Number of runs", fontsize=fs+2)
        plt.ylabel("Min of objective function", fontsize=fs+4)
        #plt.legend(runs,loc='center left', bbox_to_anchor=(1, 0.5))
        y_l = [np.min(sol_OF)*1.1,np.max(sol_OF)*0.9]
        plt.ylim(y_l)
        plt.tight_layout()
        
        plt.savefig(imname)
        print("Plot figure saved in: ",imname)
        plt.show()
        #%%
        # =============================================================================
        # -------------------------------------- Time  --------------------------------
        # ============================================================================= 
      
        name = "Time, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
        imname =  dir_plots + "time_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component+1)+".png"  
        
        fs =20
        plt.figure(figsize=(9,6))
        for i in range(len(runs)):
            for j in range(np.shape(sol_time)[1]):
            
                plt.plot(runs[i],sol_time[i,j],'*',color=colors[i], label = i)
        #plt.title(name, fontsize=fs+2)
        plt.xlabel("Number of runs", fontsize=fs+2)
        plt.ylabel("Time", fontsize=fs+4)
        #plt.legend(runs,loc='center left', bbox_to_anchor=(1, 0.5))
            
        plt.tight_layout()
        plt.savefig(imname)
        print("Plot figure saved in: ",imname)    
        
        # =============================================================================
        # -------------------------------------- Hist  IF--------------------------------
        # ============================================================================= 
        #%%
        
        n_bins = 50
        fs = 20
        if Compuestos:
            OF_c = [-2.046,-2.437,-3.836,-3.28,-3.915,-3.569,-4.496,-4.496,-5.21,-5.028,-5.69,-7.43,-11.26,-8.03]
        else:
            OF_c = [-7.91,-14.18,-17.52,-19.76,-21.46,-22.88,-24.16,-25.40,-26.73,-28.65,-31.25,-36.18]
        sols = df_sols[df_sols["Compuesto"]== Compuestos_n[component]]  
        if TBE_exp== False:
            TBE = TBS_MG(sols.iloc[0,2:],IFG)
        sols_data.append(OFN(9,iCEOS,IFG,sols.iloc[0,2:],TBE))
        OF_c[component] = OFN(9,iCEOS,IFG,sols.iloc[0,2:],TBE)
        x_lims_hist = [int(OF_c[component]*1.2),int(OF_c[component]*0.8)]
        y_lims_hist = [0,101]
        x_lims_hist_fi = [-1,int(runs[run_c]/5000)+1]
        y_lims_hist_fi = [0,int(runs[run_c]*0.1/1000)+1]
        fs = 20
        s = 0
        plt.style.use('seaborn-poster')
        plt.figure( figsize=(10,8))

        #%%
        
        name = "Feasible solutions, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)+", c: "+str(component+1)
        x_lim =[]
        y_lim = [np.min(FI_min)*0.5,np.max(FI_max)*1.2]
        y_lab = "Factible solutions"
        plot_name = dir_plots + "stats_FI_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component+1)+ ".png"

       # plot_stat_FI(FI_mean,FI_min,FI_std,FI_max,plot_name,name,x_lim,y_lim,runs,y_lab)
        
       
        plt.figure(figsize=(10,8))
        plt.tick_params( axis='both')
        plt.tick_params( which='major', direction = 'in', length=fs-2, width=1.2, colors='k', left=1, right=1, top =1)
        plt.tick_params( which='minor', direction = 'in', length=fs-2, width=1.2, colors='k',  left=1, labelbottom=False)
        font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs }
        plt.rc('font', **font)
        plt.plot( runs, FI_mean,'*')
        plt.ylim(y_lim)
        #plt.title(name, fontsize=fs+2)
        plt.xlabel("Runs", fontsize=fs+2)
        plt.ylabel(y_lab, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(plot_name)


        #%%
        name = "Objective Function, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)+", c: "+str(component+1)
        x_lim =[0,int(runs[-1])+10000]
        y_lim = [np.min(sol_OF)*1.1,np.max(sol_OF)*0.9]
       # y_lim = [int(np.mean(OF_min)*1.5),int(np.mean(OF_min)*0.95)]
        y_lab = "Objective Function"
        plot_name = dir_plots + "stats_OF_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component+1)+".png"
        
        plt.figure(figsize=(10,8))
        plt.tick_params( axis='both')
        plt.tick_params( which='major', direction = 'in', length=fs-2, width=1.2, colors='k', left=1, right=1, top =1)
        plt.tick_params( which='minor', direction = 'in', length=fs-2, width=1.2, colors='k',  left=1, labelbottom=False)
        font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs }
        plt.rc('font', **font)
        plt.plot( runs, OF_min,'*')
        plt.ylim(y_lim)
       # plt.title(name, fontsize=fs+2)
        plt.xlabel("Runs", fontsize=fs+2)
        plt.ylabel(y_lab, fontsize=fs+4)
        plt.tight_layout()
        plt.savefig(plot_name)
        
        #plot_stat(OF_mean,OF_min,OF_std,plot_name,name,x_lim,y_lim,runs,y_lab) 
     

        
    #%%    
 
df_all = pd.concat(df_all)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_all['C'] = df_all['C'].astype('int')
df_all['runs'] = df_all['runs'].astype('int')
df_all['Av FI'] = df_all['Av FI'].astype('int')
df_all['M'] = df_all['runs'].astype('int')
df_all = df_all.set_index(['C', 'runs'])
latex_code = df_all.to_latex()
latex_code = latex_code.replace("\\bottomrule", "\hline")
latex_code = latex_code.replace("\midrule", " \hline")
latex_code = latex_code.replace("\\toprule", " \hline")
latex_code = latex_code.replace("\$", "$")
latex_code = latex_code.replace("\_", "_")
with open(dir_plots+'file.txt', 'w') as f:
    print(latex_code, file=f)
with pd.option_context("max_colwidth", 1000):
    with open(dir_plots+'file.txt', 'w') as f:
        print (latex_code, file=f)
        
###### PLOT NO STATISTICS #####

#%%
fs = 20  

Para = [3,  11, 13]
Naf = [1,4, 7]
Aro = [2,5,8,10]

import numpy as np
name = "Feasible solutions, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)
y_lab = "Feasible solutions"
plot_name = dir_plots + "NO_stats_FI_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Par"+ ".png"
plt.figure(figsize=(10,8))

for i in Para:
    xplot = np.array(runs)
    yplot = np.array(df_all.loc[i+1]['Av FI']) 
    plt.plot(xplot,yplot, '*',markersize =15, label= Compuestos_n[i])
plt.ylim([0,120])
plt.title("Paraffins", fontsize=fs+2)
plt.xlabel("Runs", fontsize=fs+2)
plt.ylabel(y_lab, fontsize=fs+4)
#
plt.legend(loc='upper center', bbox_to_anchor=(0.5,0.99),
          fancybox=True, shadow=True, ncol=4, fontsize=16)
plt.tight_layout()
plt.savefig(plot_name)
plt.show()
#%%
plot_name = dir_plots + "NO_stats_FI_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Naf"+ ".png"
plt.figure(figsize=(10,8))
for i in Naf:
    xplot = np.array(runs)
    yplot = np.array(df_all.loc[i+1]['Av FI']) 
    plt.plot(xplot,yplot, '*',markersize =15, label= Compuestos_n[i])
plt.ylim([0,220])
plt.title("Naphthenes", fontsize=fs+2)
plt.xlabel("Runs", fontsize=fs+2)
plt.ylabel(y_lab, fontsize=fs+4)
#
plt.legend(loc='upper center', bbox_to_anchor=(0.5,0.99),
          fancybox=True, shadow=True, ncol=4, fontsize=16)
plt.tight_layout()
plt.savefig(plot_name)
plt.show()
#%%
plot_name = dir_plots + "NO_stats_FI_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Aro"+ ".png"
plt.figure(figsize=(10,8))
for i in Aro:
    xplot = np.array(runs)
    yplot = np.array(df_all.loc[i+1]['Av FI']) 
    plt.plot(xplot,yplot, '*',markersize =15, label= Compuestos_n[i])
plt.ylim([0,200])
plt.title("Aromatics", fontsize=fs+2)
plt.xlabel("Runs", fontsize=fs+2)
# plt.xscale("log")
# plt.xticks([7000,10000,20000, 30000,40000,50000,70000,100000,200000,500000],["7$e^3$","1$e^4$","2$e^4$","3$e^4$","4$e^4$","5$e^4$","7$e^4$","1$e^5$","2$e^5$","5$e^5$"]    )
# plt.ylabel(y_lab, fontsize=fs+4)
# plt.xticks(rotation=270)
plt.legend(loc='upper center', bbox_to_anchor=(0.5,0.99),
          fancybox=True, shadow=True, ncol=4, fontsize=16)
plt.tight_layout()
plt.savefig(plot_name)
plt.show()
#%%

# import numpy as np
# name = "Feasible solutions, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)
# y_lab = "Feasible solutions"
# plot_name = dir_plots + "NO_stats_FI_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Naf"+ ".png"
# plt.figure(figsize=(10,8))

# for i in Naf:
#     xplot = np.array(runs)
#     yplot = np.array(df_all.loc[i+1]['Av FI']) 
#     plt.plot(xplot,yplot, '*', label= Compuestos_n[i])
# plt.ylim([0,320])
# plt.title("Naphthenes", fontsize=fs+2)
# plt.xlabel("Runs", fontsize=fs+2)
# plt.ylabel(y_lab, fontsize=fs+4)
# #
# plt.legend(loc='upper center', bbox_to_anchor=(0.5,0.99),
#           fancybox=True, shadow=True, ncol=3, fontsize=16)
# plt.tight_layout()
# plt.savefig(plot_name)
# plt.show()

# #%%

# import numpy as np
# name = "Feasible solutions, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)
# y_lab = "Feasible solutions"
# plot_name = dir_plots + "NO_stats_FI_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Aro"+ ".png"
# plt.figure(figsize=(10,8))

# for i in Aro:
#     xplot = np.array(runs)
#     yplot = np.array(df_all.loc[i+1]['Av FI']) 
#     plt.plot(xplot,yplot, '*', label= Compuestos_n[i])
# plt.ylim([0,90])
# plt.title("Aromatics", fontsize=fs+2)
# plt.xlabel("Runs", fontsize=fs+2)
# plt.ylabel(y_lab, fontsize=fs+4)
# #
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
#           fancybox=True, shadow=True, ncol=3, fontsize=16)
# plt.tight_layout()
# plt.savefig(plot_name)
# plt.show()

#%%%
y_lab= "Objective function"
import numpy as np
name = "Objective Function, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)
plot_name = dir_plots + "NO_stats_0F_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Par"+ ".png"
plt.figure(figsize=(10,8))
for i in Para:
    xplot = np.array(runs)
    yplot = np.array(df_all.loc[i+1]['min OF']) 
    plt.plot(xplot,yplot, '*', markersize =15,label= Compuestos_n[i])
plt.ylim([-7,-1])
plt.title("Paraffins", fontsize=fs+2)
plt.xlabel("Runs", fontsize=fs+2)
plt.ylabel(y_lab, fontsize=fs+4)
#plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
          fancybox=True, shadow=True, ncol=4, fontsize=16)
plt.tight_layout()
plt.savefig(plot_name)
plt.show()

#%%
import numpy as np
name = "Objective Function, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)
plot_name = dir_plots + "NO_stats_0F_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Naf"+ ".png"
plt.figure(figsize=(10,8))
for i in Naf:
    xplot = np.array(runs)
    yplot = np.array(df_all.loc[i+1]['min OF']) 
    plt.plot(xplot,yplot, '*',markersize =15, label= Compuestos_n[i])
plt.ylim([-4.3,-1.5])
plt.title("Naphthenes", fontsize=fs+2)
plt.xlabel("Runs", fontsize=fs+2)
plt.ylabel(y_lab, fontsize=fs+4)
#plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5,0.99),
          fancybox=True, shadow=True, ncol=3, fontsize=16)
plt.tight_layout()
plt.savefig(plot_name)
plt.show()

#%%
import numpy as np
name = "Objective Function, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)
plot_name = dir_plots + "NO_stats_0F_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_Aro"+ ".png"
plt.figure(figsize=(10,8))
for i in Aro:
    xplot = np.array(runs)
    yplot = np.array(df_all.loc[i+1]['min OF']) 
    plt.plot(xplot,yplot, '*', markersize =15,label= Compuestos_n[i])
plt.ylim([-6.5,-1])
plt.title("Aromatics", fontsize=fs+2)
plt.xlabel("Runs", fontsize=fs+2)
plt.ylabel(y_lab, fontsize=fs+4)
#plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
          fancybox=True, shadow=True, ncol=3, fontsize=16)
plt.tight_layout()
plt.savefig(plot_name)
plt.show()
#%%
components =[3,11,10,13,1,4,7,2,5,8]
OF_c = np.zeros(14)
#sols = [-2.046,-2.437,-3.836,-3.28,-3.915,-3.569,-4.496,-5.21,-5.028,-5.69,-7.430176469,-6.896,0,-8.078]
for C in components:
    sols = df_sols[df_sols["Compuesto"]== Compuestos_n[C]]  
    
    if TBE_exp== False:
        TBE = TBS_MG(sols.iloc[0,2:],IFG)
    OF_c[C] =OFN(9,iCEOS,IFG,sols.iloc[0,2:],TBE)
    print(Compuestos_n[C])
    print(sols)
    print(OF_c[C])

    name = "Objective Function, IFG: "+str(IFG)+", iCEOS: "+str(iCEOS)+", C: "+str(C+1)
    plot_name = dir_plots + "NO_stats_0F_all_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_set_3_C_"+str(C+1)+ ".png"
    plt.figure(figsize=(10,8))
    xplot = np.array(runs)
    
    sol_OF_all = sol_OF_all_C[C]
    for i in range(len(xplot)):  
        
        yplot = []
        x_plot = []
        [ yplot.append(sol_OF_all[i,0,j]) for j in range(500) if sol_OF_all[i,0,j]<-1e-10 ]
        [ x_plot.append(xplot[i]) for j in range(500) if sol_OF_all[i,0,j]<-1e-10 ]
        #plt.plot(x_plot,yplot, '*')
        plt.plot(x_plot,yplot, '*',markersize =15, alpha = 0.25)
        #plt.axhline(y = OF_c[C], color = 'k', linestyle = ':') 
    #plt.ylim(y_lim)
    plt.title(Compuestos_n[C], fontsize=fs+2)
    plt.xlabel("Runs", fontsize=fs+2)
    plt.ylabel("Objective function", fontsize=fs+4)
    #plt.xscale("log")
    #plt.xticks([7000,10000,20000, 30000,40000,50000,70000,100000,200000,500000],["7$e^3$","1$e^4$","2$e^4$","3$e^4$","4$e^4$","5$e^4$","7$e^4$","1$e^5$","2$e^5$","5$e^5$"]    )
    #plt.ylabel(y_lab, fontsize=fs+4)
    plt.xticks(rotation=270)
    #plt.legend()
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
    #          fancybox=True, shadow=True, ncol=4, fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_name)
    plt.show()



