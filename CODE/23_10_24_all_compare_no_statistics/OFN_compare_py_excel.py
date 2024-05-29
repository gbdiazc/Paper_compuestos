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
dir_df_ex = "Comp_ex/"
comp = []
sols_py = []
sols_f  = []
sols_ex = []
Crit_val = {"Tb": np.zeros((components,3)),"Pc": np.zeros((components,3)),"Tc":np.zeros((components,3)), "Ac":np.zeros((components,3)), "Tb_err": np.zeros((components,2)),"Pc_err": np.zeros((components,2)),"Tc_err":np.zeros((components,2)), "Ac_err":np.zeros((components,2))}
df_all = []
TBE_exp    = False
HCE_exp    = True
excel = "F"
for component in np.arange(0,components):
    print("Compuesto: ", component)
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
        df_datos = dir_df_ex+ "Critical.csv"
        if TBE_exp:
            dir_name = "Compuestos_exp"
            df_name =  dir_df_ex+ "df_FG_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+ "_excel.csv"
            df_name_1 =  dir_df_ex+"df_FG_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+ "_F.csv"
            

        else:
            dir_name = "Compuestos_corr"
            df_name = dir_df_ex+ "df_FG_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+ "_corr_ex.csv"
            df_name_1 = dir_df_ex+ "df_FG_" +str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+ "_corr_F.csv"
        
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
        
        
        df   = pd.read_csv(df_name)  
        df_ex   = pd.read_csv(df_name)  
        df_1 = pd.read_csv(df_name_1)
        df_data = pd.read_csv(df_datos)
        df_comp   = df.iloc[component]
        df_comp_1 = df_1.iloc[component]
        l_runs  = len(runs)
        #%%
        if Compuestos:
            dir_plots = "Plots_compuestos"
            if TBE_exp:
                dir_plots += "_exp/"
            else:
                dir_plots += "_corr/"
        else:
            dir_plots = "Plots_componentes"
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
        data = {'Case':[],'C':[],'min OF':[],'$v_1$':[],'$v_2$':[],
                '$v_3$':[],'$v_4$':[],'$v_5$':[],'$v_6$':[],'$v_7$':[],
                '$v_8$':[],'$v_9$':[],'$Tb$':[], '$Tb(%Error)$':[], '$Pc$':[],'$Pc(%Error)$':[],  
                "Tc":[], '$Tc(%Error)$':[],"Ac":[]}
        df = pd.DataFrame(data)
        
        
        #for component in [0]:#components:#[11]:#
        print(component)
        
        if Compuestos:
            dir_merged = dir_name+"/MergedOuts_"+str(component+1)+"/"
        else:
            dir_merged = dir_name+"/MergedOuts_"+str(component+1)+"/"
    
        #dir_merged = "MergedOuts_c_"+str(component+1)+"/"
        for var in ['OF','MW','gamma','RHC','TB','FI','v','time']:
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
            TB_py  = TB_f(sol_v[run_c,min_OF_idx,:],IFG)
            Pc_py  = PCGC_f(IFG,sol_v[run_c,min_OF_idx,:])
            Tc_py  = TCGC_f(IFG,TB_py,sol_v[run_c,min_OF_idx,:],TBE)
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
            

            data_1 =  {'Case':"P",'C':Compuestos_n[component],#int(component+1),
                       'min OF':np.min(sol_OF[run_c,:]),
                       '$v_1$':sol_v[run_c,min_OF_idx,0],'$v_2$':sol_v[run_c,min_OF_idx,1],
                       '$v_3$':sol_v[run_c,min_OF_idx,2],'$v_4$':sol_v[run_c,min_OF_idx,3],
                       '$v_5$':sol_v[run_c,min_OF_idx,4],'$v_6$':sol_v[run_c,min_OF_idx,5],
                       '$v_7$':sol_v[run_c,min_OF_idx,6],'$v_8$':sol_v[run_c,min_OF_idx,7],
                       '$v_9$':sol_v[run_c,min_OF_idx,8],
                       '$Tb$':TB_py, '$Pc(%Error)$':Pc_err_py,'$Pc$':Pc_py, 
                       '$Tb(%Error)$':Tb_err_py,"Tc":Tc_py,'$Tc(%Error)$':Tc_err_py, 
                       "Ac":Ac_py, '$Ac(%Error)$':Ac_err_py}
            comp.append(component+1)
            sols_py.append(np.min(sol_OF[run_c,:]))
            df = df.append(data_1, ignore_index = True)
            
            TB_ex  = TB_f(np.array(df_comp[0:9]),IFG)
            Pc_ex  = PCGC_f(IFG,np.array(df_comp[0:9]))
            Tc_ex  = TCGC_f(IFG,TB_ex,np.array(df_comp[0:9]),TBE)
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
            data_f1 =  {'Case':"E",'C':"",#int(component+1),
                       'min OF':df_comp[compuestos[10]],
                       '$v_1$':df_comp[compuestos[0]],'$v_2$':df_comp[compuestos[1]],
                       '$v_3$':df_comp[compuestos[2]],'$v_4$':df_comp[compuestos[3]],
                       '$v_5$':df_comp[compuestos[4]],'$v_6$':df_comp[compuestos[5]],
                       '$v_7$':df_comp[compuestos[6]],'$v_8$':df_comp[compuestos[7]],
                       '$v_9$':df_comp[compuestos[8]],
                       '$Tb$':TB_ex, '$Pc(%Error)$':Pc_err_ex,'$Pc$':Pc_ex, 
                       '$Tb(%Error)$':Tb_err_ex,"Tc":Tc_ex,'$Tc(%Error)$':Tc_err_ex, 
                       "Ac":Ac_ex, '$Ac(%Error)$':Ac_err_ex}
            df = df.append(data_f1, ignore_index = True)
            sols_ex.append(df_comp[compuestos[10]])
            
            
          
            
            TB_fo  = TB_f(np.array(df_comp_1[0:9]),IFG)
            Pc_fo  = PCGC_f(IFG,np.array(df_comp_1[0:9]))
            Tc_fo  = TCGC_f(IFG,TB_fo,np.array(df_comp_1[0:9]),TBE)
            Crit_val["Tb"][component][2] = TB_fo
            Crit_val["Pc"][component][2] = Pc_fo
            Crit_val["Tc"][component][2] = Tc_fo
            Crit_val["Ac"][component][2] = Tc_fo
            data_f2=  {'Case':"F",'C':int(component+1),
                       'min OF':df_comp_1[compuestos[10]],
                       '$v_1$':df_comp_1[compuestos[0]],'$v_2$':df_comp_1[compuestos[1]],
                       '$v_3$':df_comp_1[compuestos[2]],'$v_4$':df_comp_1[compuestos[3]],
                       '$v_5$':df_comp_1[compuestos[4]],'$v_6$':df_comp_1[compuestos[5]],
                       '$v_7$':df_comp_1[compuestos[6]],'$v_8$':df_comp_1[compuestos[7]],
                       '$v_9$':df_comp_1[compuestos[8]],
                       '$Tb$':TB_fo, '$Pc$':Pc_fo, "Tc":Tc_fo}
            #df = df.append(data_f2, ignore_index = True)
            #sols_f.append(df_comp_1[compuestos[10]])
            df_all.append(df)
   

    #%%    
    
Crit_val["Tb"][:,2] = df_data["Tb"]
Crit_val["Pc"][:,2] = df_data["Pc"]
Crit_val["Tc"][:,2] = df_data["Tc"]  
Crit_val["Ac"][:,2] = df_data["Omega"]     
df = pd.concat(df_all)   
pd.set_option('display.float_format', lambda x: '%.2f' % x)
#df['C'] = df['C'].astype('int')
#df['Case'] = df['Case'].astype('int')
df = df.set_index(['C'])
df2 = df.copy()
df2 = df2.rename(columns={'$v_1$':'CH3', '$v_2$':'CH2', '$v_3$':'CH(ch)', '$v_4$':'C(ch)',
       '$v_5$':'ACH', '$v_6$':'AC', '$v_7$': 'CH2(cy)', '$v_8$':'CH(cy)', '$v_9$':'C(cy)'}, errors="raise")
writer = pd.ExcelWriter("FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".xlsx")
df.to_excel(writer)
writer.save()


writer = pd.ExcelWriter("FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_names.xlsx")
df2.to_excel(writer)
writer.save()
latex_code = df2[['Case', 'min OF','$Tb$','$Tb(%Error)$', '$Pc$', '$Pc(%Error)$','Tc','$Tc(%Error)$']].to_latex()
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

latex_code = df2[['Case','CH3', 'CH2', 'CH(ch)', 'C(ch)', 'ACH', 'AC',
       'CH2(cy)', 'CH(cy)', 'C(cy)']].to_latex()
latex_code = latex_code.replace("\\bottomrule", "\hline")
latex_code = latex_code.replace("\midrule", " \hline")
latex_code = latex_code.replace("\\toprule", " \hline")
latex_code = latex_code.replace("\$", "$")
latex_code = latex_code.replace("\_", "_")
if TBE_exp:
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_v.txt"
else:
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr_v.txt"

with open(dir_plots+filename, 'w') as f:
    print(latex_code, file=f)
with pd.option_context("max_colwidth", 1000):
    with open(dir_plots+filename, 'w') as f:
        print (latex_code, file=f)
#%%
# =============================================================================
# --------------------------------------  OF   --------------------------------
# =============================================================================    

colors = ["navy","crimson","lightseagreen","lightcoral","maroon","indigo","peru","slateblue","deeppink","navy", "darkorange","pink","blue","green","brown"]
labels = ["Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
                "Indane", "Decane", "1-Methylnaphthalene", "Undecane"]
Compuestos_n = ("Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
                "Indane", "Decane", "1-Methylnaphthalene", "Undecane")
Compuestos_n = Compuestos_n[0:components]
name = "Objective function, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "FO_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

fs =20
plt.figure(figsize=(9,6))
plt.plot(comp,sols_py,'*',color=colors[0],markersize =12, label="Python")
#plt.plot(comp,sols_f,'o',color=colors[1],label="fortran")
plt.plot(comp,sols_ex,'o',color=colors[1],markersize =8,label="Excel")
# for i in range(len(comp)):
#         plt.plot(comp[i],sols_py[i],'*',color=colors[0],label="python")
#         plt.plot(comp[i],sols_f[i],'o',color=colors[1],label="fortran")
#plt.title(name, fontsize=fs+2)
#plt.xlabel("Mixture", fontsize=fs+2)

plt.xticks(comp, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Min OF", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1.0),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-6)
#y_l = [np.min(sol_OF)*1.1,np.max(sol_OF)*0.9]
#plt.ylim(y_l)
plt.xticks(comp)
plt.grid()
plt.tight_layout()

plt.savefig(imname)
print("Plot figure saved in: ",imname)
plt.show()
            
#%%

name = "Critical Pressure, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CP_plot_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CP_plot_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

fs =20
plt.figure(figsize=(9,6))
plt.plot(comp,Crit_val["Pc"][:,0],'*',color=colors[0],markersize =12, label="Python")
#plt.plot(comp,sols_f,'o',color=colors[1],label="fortran")
plt.plot(comp,Crit_val["Pc"][:,1],'o',color=colors[1],markersize =8,label="Excel")

plt.xticks(comp, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Critical Pressure", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.8, 1.0),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-6)
#y_l = [np.min(sol_OF)*1.1,np.max(sol_OF)*0.9]
#plt.ylim(y_l)
plt.xticks(comp)
plt.grid()
plt.tight_layout()

plt.savefig(imname)
print("Plot figure saved in: ",imname)
plt.show()

#%%
name = "Critical Pressure, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Pc"]
Var_E= Crit_val["Pc_err"]/100
Errors = [ Var_E[:,0], Var_E[:,1], Var_E[:,0]*0]
VC_dic = {
    "Python" : Var_C[:,0],
    "Python_err" : Var_E[:,0],
    "Excel"  : Var_C[:,1],
    "Excel_err"  : Var_E[:,1],
    "Data": Var_C[:,2],
    "Data_err": Var_E[:,1]*0
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
plt.figure(figsize=(11,6))
fs =20
i = 0
offset = 0
for attribute, measurement in VC_dic.items():
    
    
    if multiplier%2==0:
        offset += width 
        print("offset",attribute,multiplier,colors[i])
        rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i])
        i+=1   
    else:
        print("no offset",attribute,multiplier,colors[i+3])
        rects = plt.bar(x+offset, measurement, width, label=attribute,color=colors[i+3])

    #i+=1
    
    #ax.bar_label(rects, padding=3)
    multiplier += 1
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Critical Pressure", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(0,5.5)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)
#%%
name = "Critical Pressure, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Pc"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }
Var_C = Crit_val["Pc_err"]/100
Errors = [ Var_C[:,0], Var_C[:,1], Var_C[:,0]*0]

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.3  # the width of the bars
multiplier = 0
plt.figure(figsize=(11,6.5))
fs =20
i = 0
height =10
for attribute, measurement in VC_dic.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i],yerr=Errors[i])
 
    i+=1
    #ax.bar_label(rects, padding=3)
    multiplier += 1

plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Critical Pressure", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(0,5.5)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)            
#%%
name = "Critical Pressure, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Pc"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
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
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Critical Pressure", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(0,5.5)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%
name = "Critical Temperature, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Tc"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }
Var_C = Crit_val["Tc_err"]
Errors = [ Var_C[:,0], Var_C[:,1], Var_C[:,0]*0]

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
plt.figure(figsize=(11,6.5))
fs =20
i = 0
for attribute, measurement in VC_dic.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i],yerr=Errors[i])
    i+=1
    #ax.bar_label(rects, padding=3)
    multiplier += 1
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Critical Temperature", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(450,800)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)
#%%
name = "Critical Temperature, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Tc"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
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
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Critical Temperature", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(450,800)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%
name = "Normal Boiling Temperature, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Tb"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }
Var_C = Crit_val["Tb_err"]
Errors = [ Var_C[:,0], Var_C[:,1], Var_C[:,0]*0]

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
plt.figure(figsize=(11,6.5))
fs =20
i = 0
for attribute, measurement in VC_dic.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i],yerr=Errors[i])
    i+=1
    #ax.bar_label(rects, padding=3)
    multiplier += 1
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Normal Boiling Temperature", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(300,550)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)
#%%
name = "Normal Boiling Temperature, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Tb"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
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
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Normal Boiling Temperature", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(300,550)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)
#%%
name = "Acentric factor, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "AC_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "AC_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Ac"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }
Var_C = Crit_val["Ac_err"]/100
Errors = [ Var_C[:,0], Var_C[:,1], Var_C[:,0]*0]
x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
plt.figure(figsize=(11,6))
fs =20
i = 0
for attribute, measurement in VC_dic.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i],yerr=Errors[i])
    i+=1
    #ax.bar_label(rects, padding=3)
    multiplier += 1
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Acentric factor", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(0,0.7)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%
name = "Acentric factor, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "AC_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "AC_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Ac"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Data": Var_C[:,2]
    }
x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
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
plt.xticks(x, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Acentric factor", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
plt.ylim(0,0.6)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%

name = "Tb error, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "TB_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "TB_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Tb_err"]
VC_dic = {
    "Python_err" : Var_C[:,0],
    "Excel_err"  : Var_C[:,1],
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
i = 0
fig, ax = plt.subplots(figsize=(11,6))

fs =20
for attribute, measurement in VC_dic.items():
    offset = width * i
    rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[i])
    #ax.bar_label(rects, padding=3)
    i += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Tb error %', fontsize=fs+2)
#ax.set_title('Critical pressure')
#ax.legend(loc='upper right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
ax.grid(axis='y')
ax.set_xticks(np.arange(components))
ax.set_xticklabels(Compuestos_n,rotation=75, fontsize=fs-6)
ax.set_ylim(0,7)


plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)


#%%

name = "Pc error, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CP_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CP_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Pc_err"]
VC_dic = {
    "Python_err" : Var_C[:,0],
    "Excel_err"  : Var_C[:,1],
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
i = 0
fig, ax = plt.subplots(figsize=(11,6))

fs =20
for attribute, measurement in VC_dic.items():
    offset = width * i
    rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[i])
    #ax.bar_label(rects, padding=3)
    i += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Pc error %', fontsize=fs+2)
#ax.set_title('Critical pressure')
#ax.legend(loc='upper right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
ax.grid(axis='y')
ax.set_xticks(np.arange(components))
ax.set_xticklabels(Compuestos_n,rotation=75, fontsize=fs-6)
ax.set_ylim(0,20)


plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%


name = "Tc error, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CT_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "CT_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Tc_err"]
VC_dic = {
    "Python_err" : Var_C[:,0],
    "Excel_err"  : Var_C[:,1],
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
i = 0
fig, ax = plt.subplots(figsize=(11,6))

fs =20
for attribute, measurement in VC_dic.items():
    offset = width * i
    rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[i])
    #ax.bar_label(rects, padding=3)
    i += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Tc error %', fontsize=fs+2)
#ax.set_title('Critical pressure')
#ax.legend(loc='upper right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
ax.grid(axis='y')
ax.set_xticks(np.arange(components))
ax.set_xticklabels(Compuestos_n,rotation=75, fontsize=fs-6)
ax.set_ylim(0,5)

plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)



#%%


name = "Ac error, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "AC_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+".png"  
else:
    imname =  dir_plots + "AC_err_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_corr.png"  

Var_C = Crit_val["Ac_err"]
VC_dic = {
    "Python_err" : Var_C[:,0],
    "Excel_err"  : Var_C[:,1],
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.25  # the width of the bars
i = 0
fig, ax = plt.subplots(figsize=(11,6))

fs =20
for attribute, measurement in VC_dic.items():
    offset = width * i
    rects = ax.bar(x + offset, measurement, width, label=attribute,color=colors[i])
    #ax.bar_label(rects, padding=3)
    i += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Ac error %', fontsize=fs+2)
#ax.set_title('Critical pressure')
#ax.legend(loc='upper right')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
ax.grid(axis='y')
ax.set_xticks(np.arange(components))
ax.set_xticklabels(Compuestos_n,rotation=75, fontsize=fs-6)
ax.set_ylim(0,50)


plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

