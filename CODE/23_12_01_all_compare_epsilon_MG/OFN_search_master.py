# -*- coding: utf-8 -*-
# mpirun -n 5 python mpi_OFN_stats_test.py
"""
Created on Mon Oct 24 11:54:52 2022
@author: gbdiaz
"""

IFG    = 1 # Functional group 1/2 IFG
iCEOS  = 2

MG_case = 1
if MG_case ==1:
    from functions import *
else:
    from functions_2 import *
    
from functions_plots import *
import matplotlib.pyplot as plt   
import pandas as pd
import time
import math 
import os
import numpy as np
import pickle 
# =============================================================================
# --------------------------  Initial parameters   ----------------------------
# -------------------------------  Fluids   -----------------------------------
# =============================================================================
# Casos posibles

#cases = [[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[1,1,1,1]]

cases = [[1,1,1,0]]
Compuestos_n = ("Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
                "Indane", "Decane", "1-Methylnaphthalene", "Undecane")
parallel   = False
Compuestos = True
TBE_exp    = True
HCE_exp    = True
epsilon_v =0* -1e-6


epMW = 0*1e-2
epMV = 0*1e-2
epHC = 0*1e-2
epTB = 0*1e-2
# =============================================================================
# eps=2
# epMW = 1*1e-2
# epMV = 1*1e-2
# epHC = 1*1e-2
# epTb = 1*1e-2
# =============================================================================

# =============================================================================
# epMW = 10**-eps
# epMV = 10**-eps
# epHC = 10**-eps
# epTb = 10**-eps
# =============================================================================




components = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
components =[0,1,2]
if Compuestos:
    if TBE_exp:
        dir_save = "Compuestos_TB_exp_"
        if HCE_exp:
            #dir_save += "RHC_exp/"
            dir_save += "RHC_exp/" 
        else:
            dir_save += "RHC_corr/"
    else:
       dir_save = "Compuestos_TB_corr_" 
       if HCE_exp:
           dir_save += "RHC_exp/"
       else:
           dir_save += "RHC_corr/"
else:
    dir_save = "Componentes/"
if not os.path.exists(dir_save):
    os.makedirs(dir_save)
    
if Compuestos ==True:
    MWEv = np.array([86.18,84.16,78.11,100.20,98.19,92.14,114.23,112.21,106.17,
                     128.26,118.18,142.29,142.20,156.31])
    SGEv = np.array([0.6568,0.7762,0.8762,0.6815,0.8090,0.8647,0.7006,0.8345,
                     0.8651,0.7158,0.9625,0.7283,1.0200,0.7387])
    HCEv = np.array([2.33,2.00,1.00,2.29,2.00,1.14,2.25,2.00,1.25,2.22,1.11,
                     2.20,0.91,2.18])
    TBEv = np.array([341.88, 353.93, 353.24, 371.57, 391.95, 383.79, 398.83, 
                     424.30, 409.35, 423.97, 451.12, 447.30, 517.83, 469.08])

else:    
    MWEv = np.array([129.86,213.10,284.13,353.01,423.49,498.33,580.54,674.21,
                     786.09,929.44,1138.47,1607.05])
    SGEv = np.array([0.8321,0.8947,0.9268,0.9501,0.9693,0.9864,1.002,1.018,
                     1.034,1.052,1.074,1.112])
    HCEv = np.array([1.880,1.734,1.659,1.605,1.560,1.520,1.483,1.446,1.408,
                     1.367,1.316,1.227])
    TBEv = np.array([341.88, 353.93, 353.24, 371.57, 391.95, 383.79, 398.83, 
                     424.30, 409.35, 423.97, 451.12, 447.30, 517.83, 469.08])


N  = 9       # Number of variables
DH2O   = 0.99910111111
T      = 288.705556
#TBEv = 363.53*np.ones(len(components))
# =============================================================================
# --------------------------  Initial parameters   ----------------------------
# -------------------------------  Search   -----------------------------------
# =============================================================================

#runs   = [100,200,300,400,500,600,700,800,900,1000,2000,5000,10000]       # Runs of each experiment
#runs   = [7000,10000,20000, 30000,40000,50000,70000,100000,200000]  
runs = [50000]
#runs    = [1] 
S_runs  = 1   # Runs to perform the statistics
l_runs  = len(runs)

n_dims = 9
idx_array = np.array([0,1,2,3,4,5,6,7,8])

#idx_array = np.array([0,1,4])

# ============================================================================
# ======================= Initialize virtualization ========================== 
# ============================================================================
if parallel:
	from mpi4py import MPI
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	n_ranks = comm.Get_size()
else:
	rank =0 
	n_ranks = 1

runs_rank = int(S_runs/n_ranks)
if rank == 0:
    runs_rank = S_runs - runs_rank*(n_ranks-1)
print(runs_rank ," runs in rank ", rank)



for component in components:
    print("Compuesto: ",Compuestos_n[component])
    print("Compuesto: ",component)
    OF_s    = np.empty((l_runs,runs_rank))
    OF_min  = np.empty((l_runs,1))
    OF_std  = np.empty((l_runs,1))
    OF_mean  = np.empty((l_runs,1))
    v_min   = np.empty((l_runs,n_dims))
    MW_s    = np.empty((l_runs,runs_rank))
    gamma_s = np.empty((l_runs,runs_rank))
    RHC_s   = np.empty((l_runs,runs_rank))
    TB_s    = np.empty((l_runs,runs_rank))
    MW_s    = np.empty((l_runs,runs_rank))
    FI_s    = np.empty((l_runs,runs_rank))
    Pc_s    = np.empty((l_runs,runs_rank))
    Tc_s    = np.empty((l_runs,runs_rank))
    Tbc_s   = np.empty((l_runs,runs_rank))
    Ac_s    = np.empty((l_runs,runs_rank))
    v_s     = np.empty((l_runs,runs_rank,n_dims))
    time_s  = np.empty((l_runs,runs_rank))
        
    FOc = []
    vc= []
    
    MWE =MWEv[component]
    SGE =SGEv[component]

    if HCE_exp:
        print("HCE experimental")
        HCE = HCEv[component]
    else:
        print("HCE calculada")
        HCE = -2.329*SGE+ 3.8176
    if TBE_exp:
        print("TBE experimental")
        TBE =  TBEv[component]
    else:  
        print("TBE calculada")
        TBE = TBS_f(MWE,SGE)
        
    print("TBE: ",TBE)        
    print("MWE: ",MWE)
    print("SGE: ",SGE)
    print("HCE: ",HCE)

    
    DV, VME, ResVM, RHS_VM = CSG_var(N,MWE,SGE,DH2O)
    MWI, MWE, ResMW, RHS_MW = CMW_var(N, MWE)
    HCI, HCE, ResCH, RHS_CH = CHC_var(N, HCE)
    TBI, TBE, ResTB, RHS_TB = TB_var(IFG, TBE)
    
    for case in cases:
    
        RVM = case[0]
        RMW = case[1]
        RCH = case[2]
        RTB = case[3]
        print("RVM : ",RVM)
        print("RMW : ",RMW)
        print("RCH : ",RCH)
        print("RTB : ",RTB)
        
        VR = [RVM, RMW, RCH, RTB]
        MW_UB = MWE/np.min(MWI)
        MW_LB = MWE/np.max(MWI)
        
        Lden_max = VME/np.min(DV)
        Lden_min = VME/np.max(DV)
        
        # =============================================================================
        # ---------------------------- Search algorithm   -----------------------------
        # =============================================================================
        # ----------------------------  Initialization --------------------------------
        # =============================================================================
        # Loop to repeat experiments for the statistics
        for run_s in range(runs_rank):
            run_c = 0
            for run_its in runs:
        
                start = time.time()
                
                v = np.empty(n_dims)         #  
                ks = np.arange(n_dims)       # Index vector
                
                # parameters to save feasible solutions
                fi    = 0          # Number of feasible solutions      
                feasible = []      # Feasible solutions vectors v
                kst_list      = []      # Index with feasible solutions
                X_obj_f  = []      # Objective function values
                gamma_f  = []      # Gamma constraint values
                mw_f     = []      # Mass constraint values
                Pc_f     = []      # Critical pressure
                Tc_f     = []      # Critical temperature
                Tbc_f    = []      # Normal boiling temperature
                Ac_f     = []      # Accentric factor
                
                
                # Loop to find feasible solutions
                for i in range(run_its):
                    # Initialization of v and selection of indices
                    v =  np.random.rand(9)*np.random.randint(120)
     
                    if VR == [1,1,0,0]:
                        v, A, kst = Ab_2x2 (ResVM, ResMW, RHS_VM, RHS_MW, v, idx_array, epMV, epMW, N)
                        
                    if VR == [1,0,1,0]:
                        v, A, kst = Ab_2x2 (ResVM, ResCH, RHS_VM, RHS_CH, v, idx_array, epMV, epMW, N)
                        
                    if VR == [1,0,0,1]:
                        v, A, kst = Ab_2x2 (ResVM, ResTB, RHS_VM, RHS_TB, v, idx_array, epMV, epMW, N)
                
                    if VR == [1,1,1,0]:
                        v, A, kst = Ab_3x3 (ResVM, ResMW, ResCH, RHS_VM, RHS_MW, RHS_CH, v, idx_array, epMV, epMW, epHC, N)
                        
                    if VR == [1,1,0,1]:
                        v, A, kst = Ab_3x3 (ResVM, ResMW, ResTB, RHS_VM, RHS_MW, RHS_TB, v, idx_array, epMV, epMW, epHC, N)
                        
                    if VR == [1,0,1,1]:
                        v, A, kst = Ab_3x3 (ResVM, ResCH, ResTB, RHS_VM, RHS_CH, RHS_TB, v, idx_array, epMV, epMW, epHC, N)
                        
                    if VR == [1,1,1,1]:
                        v, A, kst = Ab_4x4 (ResMW, ResVM, ResCH, ResTB, RHS_MW, RHS_VM, RHS_CH, RHS_TB, v, idx_array, epMV, epMW, epHC, epTB, N)
    
                    count = 0 # Counter to avoid an infinite loop
                    
                   # v= np.array([2,4,0,0,0,0,0,0,0])
                    
                    # If a feasible solution is found, the correspondig values are saved
                    if  (min(v)>=epsilon_v ):
                        MW    = CMW_f(MWE,v)
                        gamma = CSG_f(MWE,SGE,DH2O,v)
                        RHC   = CHC_f(HCE,v)
                        Pc    = PCGC_f(IFG,v)
                        Tbc   = TB_f(v,IFG)
                        if IFG ==2 and TBE_exp== False:
                            TBE = TBS_MG(v)
                        Tc    = TCGC_f(IFG,v,TBE)
                        Ac    = ACGC_f(TBE, Pc, Tc)
                        vlist = list(v)
   
                        if (vlist not in feasible) and math.isinf(OFN(N,iCEOS,IFG,v,TBE)) ==False :
                            feasible.append(vlist)
                            gamma_f.append(np.abs(CSG_f(MWE,SGE,DH2O,v)) )
                            mw_f.append(np.abs(CMW_f(MWE,v)))
                            X_obj_f.append(OFN(N,iCEOS,IFG,v,TBE))
                            Pc_f.append(Pc)
                            Tbc_f.append(Tbc) 
                            Tc_f.append(Tc) 
                            Ac_f.append(Ac)  
                            fi += 1
                            #print(kst, 'kst')
                            kst_list.append(kst)
                    
                end = time.time()
                total_time = end-start
                
                if fi ==0:
                    print("Not solutions found")
                    break
                min_OF = np.argmin(X_obj_f)
                v = feasible[min_OF]
                if IFG ==2 and TBE_exp== False:
                    TBE = TBS_MG(v)
                OF_s[run_c,run_s]    = OFN(N,iCEOS,IFG,v,TBE) 
                MW_s[run_c,run_s]    = np.abs(CMW_f(MWE,v))
                gamma_s[run_c,run_s] = np.abs(CSG_f(MWE,SGE,DH2O,v))
                RHC_s[run_c,run_s]   = CHC_f(HCE,v)
                TB_s[run_c,run_s]    = CTB_f(TBE,IFG,v)
                FI_s[run_c,run_s]    = fi
                Pc_s[run_c,run_s]    = PCGC_f(IFG,v)
                Tc_s[run_c,run_s]    = TCGC_f(IFG,v,TBE)
                Tbc_s[run_c,run_s]   = TB_f(v,IFG)   #Normal boiling temperature
                Ac_s[run_c,run_s]    = ACGC_f(TBE,PCGC_f(IFG,v),TCGC_f(IFG,v,TBE))
                v_s[run_c,run_s,:]   = v
                time_s[run_c,run_s]  = total_time
                run_c += 1
                #print( run_s,run_its,i,OFN(N,iCEOS,IFG,v,TBE))
        #%%
        # =============================================================================
        # ----------------------------  Solutions --------------------------------
        # =============================================================================
        for run_c in np.arange(len(runs)):
            
            OF_min[run_c]  = np.min(OF_s[run_c,:])
            OF_mean[run_c] = np.mean(OF_s[run_c,:])
            OF_std[run_c]  = np.std(OF_s[run_c,:])
            min_OF_idx = np.argmin(OF_s[run_c,:])
            v_min[run_c,:]  = v_s[run_c,min_OF_idx,:]
        min_OF = np.argmin(X_obj_f)
        v_min= feasible[min_OF] 
        print("Number of feasible solutions:", fi)
        print("Min of OF:", X_obj_f[min_OF])
        print("v min:", feasible[min_OF])
        v_min= feasible[min_OF] 
        print("Tbc(min v):",TBE)
        print("Pcc(min v):",PCGC_f(IFG,v_min))
        print("Tc(min v):",TCGC_f(IFG,v_min,TBE))
        print("Ac(min v):",ACGC_f(TBE,PCGC_f(IFG,v_min),TCGC_f(IFG,v_min,TBE)))
    
        #%%
        # ============================================================================
        # ============================ Saving results ================================ 
        # ============================================================================ 
        if Compuestos:
            dir_save_o = dir_save +"/Outs_"+str(component+1)+"/"
        else:
            dir_save_o = dir_save + "/Outs_"+str(component+1)+"/"
        if not os.path.exists(dir_save_o):
            os.makedirs(dir_save_o)
        
        print("Solutions saved in: ",dir_save)
        for var in ['OF','MW','gamma','RHC','TB','FI','v','time']:
            exec('file_'+var+'= dir_save_o + "mpi_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_rank_"+str(rank)+"_'+var+'.npy"')    
            #exec('print(file_'+var+')')
            exec('np.save(file_'+var+','+var+'_s)')
        if Compuestos:
            dir_merged = dir_save +"MergedOuts_"+str(component+1)+"/"
        else:
            dir_merged = dir_save +"MergedOuts_"+str(component+1)+"/"
        #print('concat')
        if not os.path.exists(dir_merged):
            os.makedirs(dir_merged)
        if parallel:
            comm.Barrier()
            
        if rank == 0:
        
            def concat(n_ranks,dir_save,dir_merged):
                ranks = n_ranks
                
                for var in ['OF','MW','gamma','RHC','TB','FI','v','time']:
                    for rank in range(ranks):
                        exec('file_'+var+'= dir_save + "mpi_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_rank_"+str(rank)+"_'+var+'.npy"')    
                        #exec('print(file_'+var+')')
                        if(rank==0):
                            exec('arr_'+var+' = np.load(file_'+var+')')
                        else:
                            exec('newarr_'+var+' = np.load(file_'+var+')')
                            exec('arr_'+var+' = np.concatenate((arr_'+var+',newarr_'+var+'), axis=1)')
                    exec('file_'+var+'= dir_merged + "mpi_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_'+var+'.npy"')    
                    exec('np.save(file_'+var+', arr_'+var+')')
                   # exec('print(" merged conv array saved in: ", file_'+var+')')
        
            concat(n_ranks,dir_save_o,dir_merged)
        
        
            for var in ['OF','MW','gamma','RHC','TB','FI','v','time']:
                exec('file_'+var+'= dir_merged + "mpi_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_'+var+'.npy"')    
                exec('sol_'+var+'=np.load(file_'+var+')')
            #%% 
            parameters = {"n_dims":n_dims,"S_runs":S_runs,"runs":runs,"IFG":IFG,
                          "iCEOS":iCEOS,"RVM":RVM,"RMW":RMW,"RCH":RCH,"RTB":RTB,
                          'componente':component}
            if Compuestos ==True:
                with open(dir_save+"/parameters_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component)+".pkl", 'wb') as f:
                    pickle.dump(parameters, f)
            else:
                with open(dir_save+"/parameters_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_c_"+str(component)+".pkl", 'wb') as f:
                    pickle.dump(parameters, f)
#%%

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
            

#%% 
# # =============================================================================
# # ----------------------------  Plots --------------------------------------
# # =============================================================================


# if S_runs ==1:
#     dir_plots = dir_save +"Plots"
    
#     if not os.path.exists(dir_plots):
#         os.makedirs(dir_plots)
    
#     #%%
#     imname_sols = dir_plots + "/v_" +str(IFG)+ "_"+str(iCEOS)+ ".png"
#     name = "Soluciones $v$, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)
#     plot_v(feasible,imname_sols,name)
#     #%%
#     imname_sols = dir_plots + "/OF_" +str(IFG)+ "_"+str(iCEOS)+ ".png"
#     name = "Función objetivo, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)
#     plot_OF(X_obj_f,imname_sols,name)
#     #%%
#     curves_sols = {}
#     l_feasible = np.array(feasible)
#     for i in range(9):
#         exec('curves_sols["v{}"] = {}'.format(i+1, list(l_feasible[:,i])))
#     curves_sols["OF"] = X_obj_f
#     imname_sols = dir_plots + "/hists_" +str(IFG)+ "_"+str(iCEOS)+ ".png"
#     name = "Histogramas de las soluciones $v$, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)
#     plot_hists(curves_sols,imname_sols,name)
#     #%%
#     imname_sols = dir_plots + "/sols_" +str(IFG)+ "_"+str(iCEOS)+ ".png"
#     name = "Soluciones $v$, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)
#     plot_sols(curves_sols,imname_sols,name)
#     #%%
#     imname_sols =dir_plots + "/corr_" +str(IFG)+ "_"+str(iCEOS)+ ".png"
#     name = "Correlaciones de las variables $v$, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)
#     plot_corr(curves_sols,imname_sols,name)

#%%
