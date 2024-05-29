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
MG_case = 1
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
dir_df_ex = "Data/"
comp = []
labels =[]
sols_py = []
sols_f  = []
sols_fnew  = []
sols_ex = []
v_py = []
v_f  = []
v_fnew  = []
v_ex = []
Crit_val = {"Tb": np.zeros((components,5)),"Pc": np.zeros((components,5))
            ,"Tc":np.zeros((components,5)), "Ac":np.zeros((components,5)), 
            "Tb_err": np.zeros((components,4)),"Pc_err": np.zeros((components,4)),
            "Tc_err":np.zeros((components,4)), "Ac_err":np.zeros((components,4))}
df_all = []
TBE_exp    = False
HCE_exp    = True
excel = "ex"
df_datos = dir_df_ex+ "Critical.xlsx"
df_data = pd.read_excel(df_datos, sheet_name="Critical")
df_sols = pd.read_excel(df_datos, sheet_name="Sols")
for component in  [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
    print("Compuesto: ", component)
    TBE = TBEv[component]
    comp.append(component)
    labels.append(Compuestos_n[component])
    sols = df_sols[df_sols["Compuesto"]== Compuestos_n[component]]   
    crit_sol = df_data[df_data["Compuesto"]== Compuestos_n[component]]
    # data = {'Case':[],'C':[],'min OF':[],'$v_1$':[],'$v_2$':[],
    #         '$v_3$':[],'$v_4$':[],'$v_5$':[],'$v_6$':[],'$v_7$':[],
    #         '$v_8$':[],'$v_9$':[],'$Tb$':[], '$Tb(%Error)$':[], '$Pc$':[],'$Pc(%Error)$':[],  
    #         "Tc":[], '$Tc(%Error)$':[],"Ac":[]}
    # df = pd.DataFrame(data)
    data = {'Case':[],'C':[],'min OF':[],'$v_1$':[],'$v_2$':[],
            '$v_3$':[],'$v_4$':[],'$v_5$':[],'$v_6$':[],'$v_7$':[],
            '$v_8$':[],'$v_9$':[],'$Tb$':[], '$Tb(%Error)$':[], '$Pc$':[],'$Pc(%Error)$':[],  
            "Tc":[], '$Tc(%Error)$':[],"Ac":[]}
    df = pd.DataFrame()
    data_t =  {'Case':"T",'C':Compuestos_n[component],#int(component+1),
               'min OF':sols["OF"].iloc[0],
               '$v_1$':sols["v1"].iloc[0],'$v_2$':sols["v2"].iloc[0],
               '$v_3$':sols["v3"].iloc[0],'$v_4$':sols["v4"].iloc[0],
               '$v_5$':sols["v5"].iloc[0],'$v_6$':sols["v6"].iloc[0],
               '$v_7$':sols["v7"].iloc[0],'$v_8$':sols["v8"].iloc[0],
               '$v_9$':sols["v9"].iloc[0],
               '$Tb$':crit_sol["Tb"].iloc[0], '$Pc(%Error)$':"",'$Pc$':crit_sol["Pc"].iloc[0], 
               '$Tb(%Error)$':"","Tc":crit_sol["Tc"].iloc[0],'$Tc(%Error)$':"", 
               "Ac":crit_sol["Omega"].iloc[0], '$Ac(%Error)$':""}
    # df = pd.DataFrame(data_1)
    # df_all.append(df)
    
    #df = df.append(data_1, ignore_index = True)
    for case in cases:
        RVM = case[0]
        RMW = case[1]
        RCH = case[2]
        RTB = case[3]
        print("RVM : ",RVM)
        print("RMW : ",RMW)

        print("RCH : ",RCH)
        print("RTB : ",RTB) 

        MWEv = np.array([86.18,84.16,78.11,100.20,98.19,92.14,114.23,112.21,106.17,
                         128.26,118.18,142.29,142.20,156.31])
        SGEv = np.array([0.6568,0.7762,0.8762,0.6815,0.8090,0.8647,0.7006,0.8345,
                         0.8651,0.7158,0.9625,0.7283,1.0200,0.7387])
        HCEv = np.array([2.33,2.00,1.00,2.29,2.00,1.14,2.25,2.00,1.25,2.22,1.11,
                         2.20,0.91,2.18])
        TBEv = np.array([341.88, 353.93, 353.24, 371.57, 391.95, 383.79, 398.83, 
                         424.30, 409.35, 423.97, 451.12, 447.30, 517.83, 469.08])

        ##df_datos = dir_df_ex+ "Critical.csv"

        MWE = MWEv[component]
        SGE = SGEv[component]
        HCE = HCEv[component]
        if TBE_exp == True:
            TBE = TBEv[component]

        #print("TBE: ",TBE)        
        print("MWE: ",MWE)
        print("SGE: ",SGE)
        print("HCE: ",HCE)
        df_name_f =  dir_df_ex+"df_FG_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+ "_F.xlsx"
        df_name_fnew =  dir_df_ex+"df_FG_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+ "_F_GD_LL.xlsx"
        df_name =  dir_df_ex+"df_FG_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+ "_ex.xlsx"
        if TBE_exp:
            dir_name = "Compuestos_case_"+str(MG_case)+"_TB_exp_"
            if HCE_exp==True:
                sheet_n = "RHC_exp_TB_exp"
                dir_name += "RHC_exp/"
            else:
                dir_name += "RHC_corr/"
                sheet_n = "RHC_corr_TB_exp"

        else:
            dir_name = "Compuestos_TB_corr_"
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
        
        
        #df   = pd.read_csv(df_name)  
        df_ex   = pd.read_excel(df_name,sheet_name=sheet_n)
        df_f = pd.read_excel(df_name_f,sheet_name=sheet_n)
        df_fnew = pd.read_excel(df_name_fnew,sheet_name=sheet_n)
        #df_data = pd.read_csv(df_datos)
        df_comp   = df_ex.iloc[component]
        df_comp_f = df_f.iloc[component]
        df_comp_fnew = df_fnew.iloc[component]
        l_runs  = len(runs)
  
        #%%
        if Compuestos:
            dir_plots = "Plots_compuestos_"
            if TBE_exp:
                dir_plots += "TB_exp_"
                if HCE_exp:
                    dir_plots += "RHC_exp/"
                else:
                    dir_plots += "RHC_corr/"
            else:
                dir_plots += "TB_corr_"
                if HCE_exp:
                    dir_plots += "RHC_exp/"
                else:
                    dir_plots += "RHC_corr/"
        else:
            dir_plots = "Plots_componentes_"
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
        # data = {'Case':[],'C':[],'min OF':[],'$v_1$':[],'$v_2$':[],
        #         '$v_3$':[],'$v_4$':[],'$v_5$':[],'$v_6$':[],'$v_7$':[],
        #         '$v_8$':[],'$v_9$':[],'$Tb$':[], '$Tb(%Error)$':[], '$Pc$':[],'$Pc(%Error)$':[],  
        #         "Tc":[], '$Tc(%Error)$':[],"Ac":[]}
        # df = pd.DataFrame(data)
        
        
        #for component in [0]:#components:#[11]:#
        print(component)
        
        if Compuestos:
            dir_merged = dir_name+"/MergedOuts_"+str(component+1)+"/"
        else:
            dir_merged = dir_name+"/MergedOuts_"+str(component+1)+"/"
    
        #dir_merged = "MergedOuts_c_"+str(component+1)+"/"
        for var in ['OF','MW','gamma','RHC','TB','FI','v','time','X_obj_f']:
            exec('file_'+var+'= dir_merged + "mpi_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_'+var+'.npy"')    
            #exec('print(file_'+var+')')
            exec('sol_'+var+'=np.load(file_'+var+')')
        
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
            TB_py  = TBE
            Pc_py  = PCGC_f(IFG,sol_v[run_c,min_OF_idx,:])
            Tc_py  = TCGC_f(IFG,sol_v[run_c,min_OF_idx,:],TBE)
            Ac_py =  ACGC_f(TB_py, Pc_py, Tc_py)
            Tb_err_py = 100*np.abs(TB_py-df_data["Tb"][component])/np.abs(df_data["Tb"][component])
            Pc_err_py = 100*np.abs(Pc_py-df_data["Pc"][component])/np.abs(df_data["Pc"][component])
            Tc_err_py = 100*np.abs(Tc_py-df_data["Tc"][component])/np.abs(df_data["Tc"][component])
            Ac_err_py = 100*np.abs(Ac_py-df_data["Omega"][component])/np.abs(df_data["Omega"][component])
   
            Crit_val["Tb"][component][0] = TB_py
            Crit_val["Pc"][component][0] = Pc_py
            Crit_val["Tc"][component][0] = Tc_py
            Crit_val["Ac"][component][0] = Ac_py
            Crit_val["Tb_err"][component][0] = Tb_err_py
            Crit_val["Pc_err"][component][0] = Pc_err_py
            Crit_val["Tc_err"][component][0] = Tc_err_py
            Crit_val["Ac_err"][component][0] = Ac_err_py
            

            data_rs =  {'Case':"RS",'C':"",#int(component+1),
                       'min OF':np.min(sol_OF[run_c,:]),
                       '$v_1$':sol_v[run_c,min_OF_idx,0],'$v_2$':sol_v[run_c,min_OF_idx,1],
                       '$v_3$':sol_v[run_c,min_OF_idx,2],'$v_4$':sol_v[run_c,min_OF_idx,3],
                       '$v_5$':sol_v[run_c,min_OF_idx,4],'$v_6$':sol_v[run_c,min_OF_idx,5],
                       '$v_7$':sol_v[run_c,min_OF_idx,6],'$v_8$':sol_v[run_c,min_OF_idx,7],
                       '$v_9$':sol_v[run_c,min_OF_idx,8],
                       '$Tb$':TB_py, '$Pc(%Error)$':Pc_err_py,'$Pc$':Pc_py, 
                       '$Tb(%Error)$':Tb_err_py,"Tc":Tc_py,'$Tc(%Error)$':Tc_err_py, 
                       "Ac":Ac_py, '$Ac(%Error)$':Ac_err_py}
            
            sols_py.append(np.min(sol_OF[run_c,:]))
            v_py.append(v_min)
            #df = df.append(data_1, ignore_index = True)
            
            if TBE_exp== False:
                TBE = TBS_MG(df_comp[0:9],IFG)
            TB_ex  = TBE
            Pc_ex  = PCGC_f(IFG,np.array(df_comp[0:9]))
            Tc_ex  = TCGC_f(IFG,np.array(df_comp[0:9]),TB_ex)
            Ac_ex  = ACGC_f(TB_ex, Pc_ex, Tc_ex) 
            Tb_err_ex = 100*np.abs(TB_ex-df_data["Tb"][component])/np.abs(df_data["Tb"][component])
            Pc_err_ex = 100*np.abs(Pc_ex-df_data["Pc"][component])/np.abs(df_data["Pc"][component])
            Tc_err_ex = 100*np.abs(Tc_ex-df_data["Tc"][component])/np.abs(df_data["Tc"][component])
            Ac_err_ex = 100*np.abs(Ac_ex-df_data["Omega"][component])/np.abs(df_data["Omega"][component])
            Crit_val["Tb"][component][1] = TB_ex
            Crit_val["Pc"][component][1] = Pc_ex
            Crit_val["Tc"][component][1] = Tc_ex
            Crit_val["Ac"][component][1] = Ac_ex
            Crit_val["Tb_err"][component][1] = Tb_err_ex
            Crit_val["Pc_err"][component][1] = Pc_err_ex
            Crit_val["Tc_err"][component][1] = Tc_err_ex
            Crit_val["Ac_err"][component][1] = Ac_err_ex
            data_e =  {'Case':"E",'C':"",
                       'min OF':df_comp[compuestos[10]],
                       '$v_1$':df_comp[compuestos[0]],'$v_2$':df_comp[compuestos[1]],
                       '$v_3$':df_comp[compuestos[2]],'$v_4$':df_comp[compuestos[3]],
                       '$v_5$':df_comp[compuestos[4]],'$v_6$':df_comp[compuestos[5]],
                       '$v_7$':df_comp[compuestos[6]],'$v_8$':df_comp[compuestos[7]],
                       '$v_9$':df_comp[compuestos[8]],
                       '$Tb$':TB_ex, '$Pc(%Error)$':Pc_err_ex,'$Pc$':Pc_ex, 
                       '$Tb(%Error)$':Tb_err_ex,"Tc":Tc_ex,'$Tc(%Error)$':Tc_err_ex, 
                       "Ac":Ac_ex, '$Ac(%Error)$':Ac_err_ex}
            
            sols_ex.append(df_comp[compuestos[10]])
            v_ex.append(df_comp[0:9])
            
            
          
    df = df.append(data_t, ignore_index = True)
    df = df.append(data_e, ignore_index = True)
    df = df.append(data_rs, ignore_index = True)
    df_all.append(df)
    
    #%%
    fs = 20
    s = 0
    n_bins = 50
    #plt.style.use('seaborn-poster')
    name = "Objective function hist, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
    if TBE_exp:
        imname =  dir_plots + "FO_hist_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"TB_exp_" 
        if HCE_exp:
            imname += "HC_exp" +".png"
        else:
           imname += "HC_corr" +".png" 
    else:
        imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"TB_corr_" 
        if HCE_exp:
            imname += "HC_exp" +".png" 
        else:
           imname += "HC_corr" +".png" 

    plt.figure( figsize=(10,8))   
    plt.hist(sol_X_obj_f,bins=n_bins, density=True, histtype='stepfilled', facecolor='lightseagreen',
               alpha=0.5)
    name = Compuestos_n[component]
    plt.title(name, fontsize=fs+5)
    plt.xlabel("OFN", fontsize=fs+2)
    plt.ylabel("bins", fontsize=fs+2)
    #plt.xlim(x_lim)
    #plt.ylim(y_lim)
    #plt.legend()
    plt.tight_layout()
    plt.savefig(imname)
    print("Plot figure saved in: ",imname)    
#%%

    #%%    
    
Crit_val["Tb"][:,4] = df_data["Tb"]
Crit_val["Pc"][:,4] = df_data["Pc"]
Crit_val["Tc"][:,4] = df_data["Tc"]  
Crit_val["Ac"][:,4] = df_data["Omega"]     
df = pd.concat(df_all)   

pd.set_option('display.float_format', lambda x: '%.2f' % x)
#df['C'] = df['C'].astype('int')
#df['Case'] = df['Case'].astype('int')
df = df.set_index(['C'])
C_n = [ "Heptane","Decane","Indane", "Undecane", "Cyclohexane", "Cycloheptane","Cyclooctane",
                 "Benzene", "Toluene", "Ethylbenzene"]
df2 = df.copy()
df2 = df2.rename(columns={'$v_1$':'CH3', '$v_2$':'CH2', '$v_3$':'CH(ch)', '$v_4$':'C(ch)',
       '$v_5$':'ACH', '$v_6$':'AC', '$v_7$': 'CH2(cy)', '$v_8$':'CH(cy)', '$v_9$':'C(cy)'}, errors="raise")
writer = pd.ExcelWriter("FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".xlsx")
df.to_excel(writer)
writer.save()


writer = pd.ExcelWriter("FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_names.xlsx")
df2.to_excel(writer)
writer.save()
latex_code = df2[['Case', 'min OF','$Tb$','$Tb(%Error)$', '$Pc$', '$Pc(%Error)$','Tc','$Tc(%Error)$','Ac','$Ac(%Error)$']].to_latex()
latex_code = latex_code.replace("\\bottomrule", "\hline")
latex_code = latex_code.replace("\midrule", " \hline")
latex_code = latex_code.replace("\\toprule", " \hline")
latex_code = latex_code.replace("\$", "$")
latex_code = latex_code.replace("\_", "_")
if TBE_exp:
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_OF.txt"
else:
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr_OF.txt"

with open(dir_plots+filename, 'w') as f:
    print(latex_code, file=f)
with pd.option_context("max_colwidth", 1000):
    with open(dir_plots+filename, 'w') as f:
        print (latex_code, file=f)
#%%

latex_code = df2[['Case','min OF','CH3', 'CH2', 'CH(ch)', 'C(ch)', 'ACH', 'AC',
       'CH2(cy)', 'CH(cy)', 'C(cy)']].to_latex()
latex_code = latex_code.replace("\\bottomrule", "\hline")
latex_code = latex_code.replace("\midrule", " \hline")
latex_code = latex_code.replace("\\toprule", " \hline")
latex_code = latex_code.replace("\$", "$")
latex_code = latex_code.replace("\_", "_")
if TBE_exp:
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_C_"+str(component)+"_v.txt"
else:
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_C_"+str(component)+"_v.txt"

with open(dir_plots+filename, 'w') as f:
    print(latex_code, file=f)
with pd.option_context("max_colwidth", 1000):
    with open(dir_plots+filename, 'w') as f:
        print (latex_code, file=f)
#%%
# =============================================================================
# --------------------------------------  OF   --------------------------------
# =============================================================================    

colors = ["navy","crimson","lightseagreen","green","lightcoral","maroon","indigo","peru","slateblue","deeppink","navy", "darkorange","pink","blue","brown"]
#labels = ["Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
#                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
#                "Indane", "Decane", "1-Methylnaphthalene", "Undecane"]
#Compuestos_n = ("Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
#                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
#                "Indane", "Decane", "1-Methylnaphthalene", "Undecane")
#Compuestos_n = Compuestos_n[0:components]
name = "Objective function, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"TB_exp_" 
    if HCE_exp:
        imname += "HC_exp" +".png"
    else:
       imname += "HC_corr" +".png" 
else:
    imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"TB_corr_" 
    if HCE_exp:
        imname += "HC_exp" +".png" 
    else:
       imname += "HC_corr" +".png" 

fs = 18
plt.figure(figsize=(9,6))
x = np.arange(len(labels))
plt.plot(x,sols_py,'*',color=colors[0],markersize =12, label="Random Search")
plt.plot(x,sols_ex,'d',color=colors[2],markersize =10,label="Excel")
#plt.plot(x,sols_f,'o',color=colors[1],label="Fortran")
#plt.plot(x,sols_fnew,'<',color=colors[3],label="Random Search")
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Min OF", fontsize=fs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, shadow=True, ncol=4, fontsize=fs-6)
#y_l = [np.min(sol_OF)*1.1,np.max(sol_OF)*0.9]
#plt.ylim(y_l)
plt.xticks(x, fontsize=fs-6)
plt.grid()
plt.tight_layout()

plt.savefig(imname)
print("Plot figure saved in: ",imname)
plt.show()

#%%
# =============================================================================
# --------------------------------------  OF  selected --------------------------------
# =============================================================================    

colors = ["navy","crimson","lightseagreen","green","lightcoral","maroon","indigo","peru","slateblue","deeppink","navy", "darkorange","pink","blue","brown"]
#labels = ["Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
#                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
#                "Indane", "Decane", "1-Methylnaphthalene", "Undecane"]
#Compuestos_n = ("Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
#                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
#                "Indane", "Decane", "1-Methylnaphthalene", "Undecane")
#Compuestos_n = Compuestos_n[0:components]
name = "Objective function, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"TB_exp_" 
    if HCE_exp:
        imname += "HC_exp" +".png"
    else:
       imname += "HC_corr" +".png" 
else:
    imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"TB_corr_" 
    if HCE_exp:
        imname += "HC_exp" +".png" 
    else:
       imname += "HC_corr" +".png" 

fs = 18
plt.figure(figsize=(9,6))
x = np.arange(len(labels))
labl  = []
sls_py = []
sls_ex = []
sls_f = []
s = 0
for xi in [3,11,13,1,4,7,2,5,8,10]:
    if xi==3:
        plt.plot(s,sols_py[xi],'*',color=colors[0],markersize =12, label="Random Search")
        plt.plot(s,sols_ex[xi],'d',color=colors[2],markersize =10,label="Excel")
        #plt.plot(s,sols_f[xi],'o',color=colors[1],label="Fortran")
    else:
        plt.plot(s,sols_py[xi],'*',color=colors[0],markersize =12)
        plt.plot(s,sols_ex[xi],'d',color=colors[2],markersize =10)
        #plt.plot(s,sols_f[xi],'o',color=colors[1])
    sls_py.append(sols_py[xi])
    sls_ex.append(sols_ex[xi])
    #sls_f.append(sols_f[xi])
    labl.append(labels[xi])
    s+=1
plt.axvline(x = 2.5, color = 'b', linestyle = ':') 
plt.axvline(x = 5.5, color = 'b', linestyle = ':') 
#plt.plot(x,sols_fnew,'<',color=colors[3],label="Random Search")
plt.xticks( [0,1,2,3,4,5,6,7,8,9], labl, rotation=75, fontsize=fs-6)
plt.ylabel("Min OF", fontsize=fs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, shadow=True, ncol=4, fontsize=fs-6)
#y_l = [np.min(sol_OF)*1.1,np.max(sol_OF)*0.9]
#plt.ylim(y_l)
#plt.xticks(x, fontsize=fs-6)
plt.grid()
plt.tight_layout()

plt.savefig(imname)
print("Plot figure saved in: ",imname)
plt.show()
#%%

r = 0
for c in [3,11,13,1,4,7,2,5,8,10]:
    
    x_max, x_min =  4, -1
    y_max, y_min = -10, 0

    font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs}
    plt.rc('font', **font)

    cmap, norm = mcolors.from_levels_and_colors([0, 2, 5, 6], ['red', 'green', 'blue'])
    vpy = v_py[r][0]
    vex = v_ex[r]
    #vf  = v_f[r]
    #vfnew  = v_fnew[r]

if TBE_exp:
    imname =  dir_plots + "IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_OF.png" 
else:
    imname =  dir_plots + "IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_OF.png" 


VC_dic = {
    #"Python" : sols_py,
    "Excel"  : sls_ex,
    #"Fortran": sls_f,
    "Random search": sls_py
    }

x = np.arange(len(sls_py))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0
plt.figure(figsize=(11,6))
fs =20
i = 0
for attribute, measurement in VC_dic.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i])
    i+=1
    #ax.bar_label(rects, padding=3)
    multiplier += 1
plt.xticks(x, labl, rotation=75, fontsize=fs-7)
plt.ylabel("OF", fontsize=fs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),#bbox_to_anchor=(0.3, 0.23),
          fancybox=True, shadow=True, ncol=4, fontsize=fs-6)
plt.axvline(x = 2.7, color = 'maroon', linestyle = ':') 
plt.axvline(x = 5.7, color = 'maroon', linestyle = ':') 
plt.text(0.5, -8.5, 'Paraffins', fontsize = 15, color = 'maroon') 
plt.text(3.5, -8.5, 'Naphthenes', fontsize = 15, color = 'maroon') 
plt.text(7, -8.5, 'Aromatics', fontsize = 15, color = 'maroon') 
plt.ylim(y_min,y_max)
plt.grid(axis='y')
#plt.title(sheet_n, fontsize=fs+2)
plt.tight_layout()
plt.savefig(imname,bbox_inches='tight')
print("Plot figure saved in: ",imname)

#%%
comp = [3,11,13,1,4,7,2,5,8,10]
fs = 18
name = "Critical Constants, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CC_plot_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  
else:
    imname =  dir_plots + "CC_plot_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  

Var_C = Crit_val["Pc"]
VC_dic_Pc = {
    "Excel"  : Var_C[:,1][comp],
    #"Fortran"  : Var_C[:,2][comp],
    "Random Search"  : Var_C[:,0][comp],
    "Data": Var_C[:,4][comp]
    }
Var_C = Crit_val["Tc"]
VC_dic_Tc = {
    "Excel"  : Var_C[:,1][comp],
    #"Fortran"  : Var_C[:,2][comp],
    "Random Search"  : Var_C[:,0][comp],
    "Data": Var_C[:,4][comp]
    }
Var_C = Crit_val["Tb"]
VC_dic_Tb = {
    "Excel"  : Var_C[:,1][comp],
    #"Fortran"  : Var_C[:,2][comp],
    "Random Search"  : Var_C[:,0][comp],
    "Data": Var_C[:,4][comp]
    }
Var_C = Crit_val["Ac"]
VC_dic_Ac = {
    "Excel"  : Var_C[:,1][comp],
    #"Fortran"  : Var_C[:,2][comp],
    "Random Search"  : Var_C[:,0][comp],
    "Data": Var_C[:,4][comp]
    }        

x = np.arange(len(labels))  # the label locations
width = 0.18  # the width of the bars
i = 0
#fig = plt.figure(figsize=(11,30))
fig, axs = plt.subplots(4, sharex=True,figsize=(12,18))
x = np.arange(0,10)
Names = ["Critical Pressure", "Critical Temperature", "Normal Boiling Temperature", "Accentric Factor"]
y_ulim = [5.5,800,540,0.6] 
y_llim = [0,450,300,0]
for sp in [VC_dic_Pc,VC_dic_Tc,VC_dic_Tb,VC_dic_Ac]:
    multiplier = 0
    for attribute, measurement in sp.items():
        offset = width * multiplier
        if sp == "VC_dic_Pc":
            axs[i].bar(x + offset, measurement, width,label=attribute,color=colors[multiplier])
            #(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                      #fancybox=True, shadow=True, ncol=5, fontsize=fs-4)
        else:
            axs[i].bar(x + offset, measurement, width, color=colors[multiplier],label=attribute)
        
        #ax.bar_label(rects, padding=3)
        multiplier += 1
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
              fancybox=True, shadow=True, ncol=5, fontsize=fs-4)
    axs[i].grid(axis='y')
    axs[i].set_ylabel(Names[i], fontsize=fs)
    axs[i].set_ylim(y_llim[i],y_ulim[i])
    axs[i].axvline(x = 2.7, color = 'maroon', linestyle = ':') 
    axs[i].axvline(x = 5.7, color = 'maroon', linestyle = ':') 
    i+=1
    #plt.ylabel(, fontsize=fs)
plt.xticks(x, labl, rotation=75, fontsize=fs-2)
plt.text(0.5, 2.7, 'Paraffins', fontsize = 15, color = 'maroon') 
plt.text(3.5, 2.7, 'Naphthenes', fontsize = 15, color = 'maroon') 
plt.text(7.1,2.7, 'Aromatics', fontsize = 15, color = 'maroon') 
# plt.text(1, 1090, 'Paraffins', fontsize = 15, color = 'maroon') 
# plt.text(4.5, 1090, 'Naphtenes', fontsize = 15, color = 'maroon') 
# plt.text(7.7,1090, 'Aromatics', fontsize = 15, color = 'maroon') 
#plt.legend([ax[0], ax[1], ax[2]],["HHZ 1", "HHN", "HHE"])
#fig.legend(ax[0].get_legend_handles_labels(),
#            loc='upper center', ncol=4)
#plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%
r = 0
fig, axs = plt.subplots(1, 2, sharex=True,sharey=True,figsize=(12,4),gridspec_kw={'hspace': 0.2, 'wspace': 0.1})
for c in [0,1]:
    
    x_max, x_min =  4, -1
    y_max, y_min = 0, -5

    font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs}
    plt.rc('font', **font)
    
    vpy = v_py[r][0]
    vex = v_ex[r]
    #vf  = v_f[r]
    #vfnew  = v_fnew[r]
    
    VC_dic = {

        "Excel"  : vex,
        #"Fortran": vf,
        "Random Search": vpy,
        }
    
    
    

    x = np.arange(len(vpy))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0
    #fs =20
    i = 0
    labels_1 = ['CH3','CH2','CH(ch)', 'C(ch)','ACH','AC','CH2(cy)','CH(cy)','C(cy)']#["$v_1$","$v_2$","$v_3$","$v_4$","$v_5$","$v_6$","$v_7$","$v_8$","$v_9$"]
    for attribute, measurement in VC_dic.items():
        offset = width * multiplier
        axs[r].bar(x + offset, measurement, width, label=attribute,color=colors[i])
        i+=1
        axs[r].set_title(Compuestos_n[c], fontsize=18) 
        #ax.bar_label(rects, padding=3)
        multiplier += 1
    r +=1
    
axs[0].set_ylim(0,5)

axs[0].set_xticks(x) 
axs[0].set_xticklabels(labels_1, rotation = 45, fontsize=14)
axs[1].set_xticklabels(labels_1, rotation = 45, fontsize=14)
#axs[0].set_yticklabels([1,2,3,4,5],fontsize=15)
axs[1].legend(loc='upper center', bbox_to_anchor=(0.78, 1), fancybox=True, shadow=True, ncol=1,fontsize=11)
imname =  dir_plots + "IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_v_1.png" 
plt.tight_layout()
plt.savefig(imname,bbox_inches='tight')
print("Plot figure saved in: ",imname)



