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
case  = [1,1,1,0]
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
df = []
TBE_exp    = False
HCE_exp    = True
excel = "ex"
df_datos = dir_df_ex+ "Critical.xlsx"
df_data = pd.read_excel(df_datos, sheet_name="Critical")
df_sols = pd.read_excel(df_datos, sheet_name="Sols")#[1]:# 
for component in [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
    print("Compuesto: ", component)
    print("Compuesto: ", Compuestos_n[component])
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
    TBE = TBEv[component]

    print("TBE: ",TBE)        
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
    #df_ex   = pd.read_excel(df_name,sheet_name=sheet_n)
    #df_f = pd.read_excel(df_name_f,sheet_name=sheet_n)
    #df_fnew = pd.read_excel(df_name_fnew,sheet_name=sheet_n)
    #df_data = pd.read_csv(df_datos)
    #df_comp   = df_ex.iloc[component]
    #df_comp_f = df_f.iloc[component]
    #df_comp_fnew = df_fnew.iloc[component]
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
    
    
    OF_v =[]
    TB_v = []
    Pc_v = []
    Tc_v = []
    Ac_v = []
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
    for var in ['OF','MW','gamma','RHC','TB','FI','v','time','X_obj_f','feasible']:
        exec('file_'+var+'= dir_merged + "mpi_FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_'+var+'.npy"')    
        #exec('print(file_'+var+')')
        exec('sol_'+var+'=np.load(file_'+var+')')
    
    
    #%%
    fs = 20
    s = 0
    n_bins = 50
    #plt.style.use('seaborn-poster')
    name = "Objective function hist, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
    if TBE_exp:
        imname =  dir_plots + "FO_hist_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"TB_exp_" 
        if HCE_exp:
            imname += "HC_exp_" + str(component) +".png"
        else:
           imname += "HC_corr" + str(component) +".png" 
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
    for fact in sol_feasible:
        OF_v.append(OFN(9,iCEOS,IFG,fact,TBE))
        TB_v.append(TBE)
        Pc_v.append(PCGC_f(IFG,fact))
        Tc_v.append(TCGC_f(IFG,fact,TBE))
        Ac_v.append(ACGC_f(TBE, PCGC_f(IFG,fact),TCGC_f(IFG,fact,TBE))) 
    Pc_err = np.abs(100*(Pc_v-df_data["Pc"].iloc[component])/df_data["Pc"].iloc[component])
    Tc_err = np.abs(100*(Tc_v-df_data["Tc"].iloc[component])/df_data["Tc"].iloc[component])
    Ac_err = np.abs(100*(Ac_v-df_data["Omega"].iloc[component])/df_data["Omega"].iloc[component])
    data = {"min OF":OF_v,'$v_1$':sol_feasible[:,0],'$v_2$':sol_feasible[:,1],
    '$v_3$':sol_feasible[:,2],'$v_4$':sol_feasible[:,3],
    '$v_5$':sol_feasible[:,4],'$v_6$':sol_feasible[:,5],
    '$v_7$':sol_feasible[:,6],'$v_8$':sol_feasible[:,7],
    '$v_9$':sol_feasible[:,8],"$Pc$":Pc_v,'$Pc(%Error)$':Pc_err,
    "$Tc$":Tc_v,'$Tc(%Error)$':Tc_err,"$Ac$":Ac_v,'$Ac(%Error)$':Ac_err}
    
    df = pd.DataFrame(data)
    C_n = [ "Heptane","Decane","Indane", "Undecane", "Cyclohexane", "Cycloheptane","Cyclooctane",
                     "Benzene", "Toluene", "Ethylbenzene"]
    df2 = df.copy()
    #df2 = df2[df2['$Ac(%Error)$']<15]
    df2 = df2.rename(columns={'$v_1$':'CH3', '$v_2$':'CH2', '$v_3$':'CH(ch)', '$v_4$':'C(ch)',
           '$v_5$':'ACH', '$v_6$':'AC', '$v_7$': 'CH2(cy)', '$v_8$':'CH(cy)', '$v_9$':'C(cy)'}, errors="raise")
    writer = pd.ExcelWriter("FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_C_"+str(component)+".xlsx")
    df.to_excel(writer)
    writer.save()


    writer = pd.ExcelWriter("FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_C_"+str(component)+".xlsx")
    df2.to_excel(writer)
    writer.save()
    latex_code = df2[[ 'min OF', '$Pc$', '$Pc(%Error)$','$Tc$','$Tc(%Error)$','$Ac$','$Ac(%Error)$']].to_latex()
    latex_code = latex_code.replace("\\bottomrule", "\hline")
    latex_code = latex_code.replace("\midrule", " \hline")
    latex_code = latex_code.replace("\\toprule", " \hline")
    latex_code = latex_code.replace("\$", "$")
    latex_code = latex_code.replace("\_", "_")
    if TBE_exp:
        filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_C_"+str(component)+"_OF.txt"
    else:
        filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_C_"+str(component)+"_corr_OF.txt"

    with open(dir_plots+filename, 'w') as f:
        print(latex_code, file=f)
    with pd.option_context("max_colwidth", 1000):
        with open(dir_plots+filename, 'w') as f:
            print (latex_code, file=f)
#%%
v=[0,	0,	0,	0,	0,	0,	7,	0,	0]

Ac = ACGC_f(TBE, PCGC_f(IFG,v),TCGC_f(IFG,v,TBE))

Ac_err = np.abs(100*(Ac-df_data["Omega"].iloc[component])/df_data["Omega"].iloc[component])
print(Ac_err)
print(OFN(9,iCEOS,IFG,v,TBE))