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
components = 1
TBEv = np.array([341.88, 353.93, 353.24, 371.57, 391.95, 383.79, 398.83, 
                 424.30, 409.35, 423.97, 451.12, 447.30, 517.83, 469.08])
# [RVM, RMW, RCH, RTB]
cases = [[1,1,0,0],[1,0,1,0],[1,0,0,1],[1,1,1,0],[1,1,0,1],[1,0,1,1],[1,1,1,1]]
cases = [[1,1,0,0]]
IFG    = 2
iCEOS  = 2
compuestos = {0:'CH3', 1:'CH2', 2:'CH(CHAIN)', 3:'C(CHAIN)',
       4:'ACH', 5:'AC', 6: 'CH2(CYCLIC)', 7:'CH(CYCLIC)', 8:'C(CYCLIC)',9:'Fobj',
       10:'ln(Fobj)'}
Compuestos_n = ("Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
                "Indane", "Decane", "1-Methylnaphthalene", "Undecane")
dir_df_ex = "Data/"
comp = []
sols_py = []
sols_f  = []
sols_ex = []
v_py = []
v_f  = []
v_ex = []
Crit_val = {"Tb": np.zeros((components,4)),"Pc": np.zeros((components,4))
            ,"Tc":np.zeros((components,4)), "Ac":np.zeros((components,4)), 
            "Tb_err": np.zeros((components,3)),"Pc_err": np.zeros((components,3)),
            "Tc_err":np.zeros((components,3)), "Ac_err":np.zeros((components,3))}
df_all = []
TBE_exp    = True
HCE_exp    = True
excel = "ex"
component = 0

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
    df_datos_f = dir_df_ex+ "soluciones_fortran.xlsx"
    df_f   = pd.read_excel(df_datos_f)
    df_data = pd.read_csv(df_datos)
    if TBE_exp:
        dir_name = "Compuestos_TB_exp_"
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
        

        data_1 =  {'Case':"P",'C':Compuestos_n[component],#int(component+1),
                   'min OF':np.min(sol_OF[run_c,:]),
                   '$v_1$':sol_v[run_c,min_OF_idx,0],'$v_2$':sol_v[run_c,min_OF_idx,1],
                   '$v_3$':sol_v[run_c,min_OF_idx,2],'$v_4$':sol_v[run_c,min_OF_idx,3],
                   '$v_5$':sol_v[run_c,min_OF_idx,4],'$v_6$':sol_v[run_c,min_OF_idx,5],
                   '$v_7$':sol_v[run_c,min_OF_idx,6],'$v_8$':sol_v[run_c,min_OF_idx,7],
                   '$v_9$':sol_v[run_c,min_OF_idx,8]}
        comp.append(component+1)
        sols_py.append(np.min(sol_OF[run_c,:]))
        v_py.append(v_min)
        df = df.append(data_1, ignore_index = True)
        
        data_f2=  {'Case':"F",'C':int(component+1),
                    'min OF':df_f[compuestos[10]],
                    '$v_1$':df_f[compuestos[0]],'$v_2$':df_f[compuestos[1]],
                    '$v_3$':df_f[compuestos[2]],'$v_4$':df_f[compuestos[3]],
                    '$v_5$':df_f[compuestos[4]],'$v_6$':df_f[compuestos[5]],
                    '$v_7$':df_f[compuestos[6]],'$v_8$':df_f[compuestos[7]],
                    '$v_9$':df_f[compuestos[8]],
                    }
        df = df.append(data_f2, ignore_index = True)
        sols_f.append(df_f[compuestos[10]])
        v_f.append(df_f[0:9])
        df_all.append(df)
   
#%%

plt.plot(OF_s,"*g",label="python",alpha=0.09)
x = np.ones(len(df_f))
plt.plot(x[0:200],df_f["ln(Fobj)"][0:200],"*b",label="fortran",alpha=0.09)
plt.xticks([0,1],["Python","Fortran"],fontsize=8)
#plt.legend()
#%%
plt.figure
plt.hist(OF_s[0],alpha=0.5,label="python")
plt.hist(df_f["ln(Fobj)"][0:200],alpha=0.5,label="fortran")
plt.legend()
#%%
plt.figure
plt.hist(OF_s[0],alpha=0.5,label="python")
plt.hist(df_f["ln(Fobj)"],alpha=0.5,label="fortran")
plt.legend()
    #%%    
    
Crit_val["Tb"][:,3] = df_data["Tb"]
Crit_val["Pc"][:,3] = df_data["Pc"]
Crit_val["Tc"][:,3] = df_data["Tc"]  
Crit_val["Ac"][:,3] = df_data["Omega"]     
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
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_v.txt"
else:
    filename = "FG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_v.txt"

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
labels = ["Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
                "Indane", "Decane", "1-Methylnaphthalene", "Undecane"]
Compuestos_n = ("Hexane", "Cyclohexane", "Benzene", "Heptane", "Cycloheptane",
                "Toluene", "Octane", "Cyclooctane", "Ethylbenzene", "Nonane",
                "Indane", "Decane", "1-Methylnaphthalene", "Undecane")
Compuestos_n = Compuestos_n[0:components]
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

fs =20
plt.figure(figsize=(9,6))
plt.plot(comp,sols_ex,'d',color=colors[2],markersize =12,label="Excel")
plt.plot(comp,sols_py,'*',color=colors[0],markersize =12, label="Python")
plt.plot(comp,sols_f,'o',color=colors[1],label="Fortran")

#
# for i in range(len(comp)):
#         plt.plot(comp[i],sols_py[i],'*',color=colors[0],label="python")
#         plt.plot(comp[i],sols_f[i],'o',color=colors[1],label="fortran")
#plt.title(sheet_n, fontsize=fs+2)
#plt.xlabel("Mixture", fontsize=fs+2)

plt.xticks(comp, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Min OF", fontsize=fs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-6)
#y_l = [np.min(sol_OF)*1.1,np.max(sol_OF)*0.9]
#plt.ylim(y_l)
plt.xticks(comp, fontsize=fs-6)
plt.grid()
plt.tight_layout()

plt.savefig(imname)
print("Plot figure saved in: ",imname)
plt.show()
#%%




for c in range(components):
    
    x_max, x_min =  4, -1
    y_max, y_min = 0, -5
    fs = 22
    
    font = {'family' : 'serif', 'weight' : 'normal', 'size' : fs}
    plt.rc('font', **font)
    
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3, figsize=(20,16), sharey=True)
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
    vpy = v_py[c][0]
    vex = v_ex[c]
    vf  = v_f[c]
    for i in range(9):
            exec('ax{}.set_xlim(x_min, x_max)'.format(i+1))
            exec('ax{}.set_ylim(y_min,y_max)'.format(i+1))
            exec('ax{}.set_xlabel("$v_{}$", fontsize=fs)'.format(i+1,i+1))
            exec('ax{}.set_ylabel("$OFN$", fontsize=fs)'.format(i+1))
            exec('ax{}.tick_params( which="major", direction = "in", left=1, right=1, top =1)'.format(i+1))
    #        exec('ax{}.plot(curves["v{}"], curves["OF"],"+",markersize = 20, color = cmap)'.format(i+1,i+1))
            exec('ax{}.plot(vex[{}], sols_ex[c],"d",markersize = 20, color = colors[1],label = "Excel")'.format(i+1,i))
            exec('ax{}.plot(vpy[{}], sols_py[c],"*",markersize = 20, color = colors[0],label ="Python")'.format(i+1,i))
            exec('ax{}.plot(vf[{}], sols_f[c],"o",markersize = 12, color = colors[2],label="fortran")'.format(i+1,i))
            exec('ax{}.grid(linewidth=1.5)'.format(i+1))
    plt.legend(loc='upper center', bbox_to_anchor=(-0.8, 3.7),
              fancybox=True, shadow=True, ncol=3, fontsize=fs-2)

    name = "Variables, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
    if TBE_exp:
        imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_C_"+str(c)+"_v.png"  
    else:
        imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_C_"+str(c)+"_v.png"  
    
    
    VC_dic = {
        "Python" : vpy,
        "Excel"  : vex,
        "Fortran": vf
        }
    
    x = np.arange(len(vpy))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0
    plt.figure(figsize=(11,6))
    #fs =20
    i = 0
    labels_1 = ['CH3','CH2','CH(ch)', 'C(ch)','ACH','AC','CH2(cy)','CH(cy)','C(cy)']#["$v_1$","$v_2$","$v_3$","$v_4$","$v_5$","$v_6$","$v_7$","$v_8$","$v_9$"]
    for attribute, measurement in VC_dic.items():
        offset = width * multiplier
        rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i])
        i+=1
        #ax.bar_label(rects, padding=3)
        multiplier += 1
    plt.xticks(x, labels_1, rotation=75, fontsize=fs-4)
    plt.ylabel("Value", fontsize=fs)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),#bbox_to_anchor=(0.72, 0.98),
              fancybox=True, shadow=True, ncol=3, fontsize=fs-4)
    #plt.ylim(300,550)
    plt.grid(axis='y')
    #plt.title(sheet_n, fontsize=fs+2)
    plt.tight_layout()
    plt.savefig(imname)
    print("Plot figure saved in: ",imname)
#%%

if TBE_exp:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_OF.png" 
else:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_OF.png" 


VC_dic = {
    "Python" : sols_py,
    "Excel"  : sols_ex,
    "Fortran": sols_f
    }

x = np.arange(len(sols_py))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0
plt.figure(figsize=(11,6))
#fs =20
i = 0
for attribute, measurement in VC_dic.items():
    offset = width * multiplier
    rects = plt.bar(x + offset, measurement, width, label=attribute,color=colors[i])
    i+=1
    #ax.bar_label(rects, padding=3)
    multiplier += 1
plt.xticks(x, labels, rotation=75, fontsize=fs-7)
plt.ylabel("OF", fontsize=fs)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),#bbox_to_anchor=(0.3, 0.23),
          fancybox=True, shadow=True, ncol=3, fontsize=fs-8)
#plt.ylim(300,550)
plt.grid(axis='y')
#plt.title(sheet_n, fontsize=fs+2)
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)
           
#%%

name = "Critical Pressure, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CP_plot_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  
else:
    imname =  dir_plots + "CP_plot_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  

fs =20
plt.figure(figsize=(9,6))
plt.plot(comp,Crit_val["Pc"][:,0],'*',color=colors[0],markersize =12, label="Python")
plt.plot(comp,Crit_val["Pc"][:,1],'d',color=colors[1],markersize =10,label="Excel")
plt.plot(comp,Crit_val["Pc"][:,2],'o',color=colors[2],markersize =8,label="fortran")

plt.xticks(comp, labels, rotation=75, fontsize=fs-6)
plt.ylabel("Critical Pressure", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.17),
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
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  
else:
    imname =  dir_plots + "CP_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  

Var_C = Crit_val["Pc"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Fortran"  : Var_C[:,2],
    "Data": Var_C[:,3]
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.20  # the width of the bars
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
          fancybox=True, shadow=True, ncol=4, fontsize=fs-4)
plt.ylim(0,5.5)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%
name = "Critical Temperature, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  
else:
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  

Var_C = Crit_val["Tc"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Fortran": Var_C[:,2]
    }
Var_C = Crit_val["Tc_err"]
Errors = [ Var_C[:,0], Var_C[:,1], Var_C[:,2]]

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
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  
else:
    imname =  dir_plots + "CT_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_corr.png"  

Var_C = Crit_val["Tc"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Fortran": Var_C[:,2],
    "Data": Var_C[:,3]
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.20  # the width of the bars
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
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          fancybox=True, shadow=True, ncol=4, fontsize=fs-4)
plt.ylim(450,800)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(imname)
print("Plot figure saved in: ",imname)

#%%
name = "Normal Boiling Temperature, IFG: " +str(IFG)+ ", iCEOS: " + str(iCEOS)+", c: "+str(component+1)
if TBE_exp:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  
else:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_corr.png"  

Var_C = Crit_val["Tb"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Fortran"  : Var_C[:,2]
    }
Var_C = Crit_val["Tb_err"]
Errors = [ Var_C[:,0], Var_C[:,1], Var_C[:,2]]

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
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+".png"  
else:
    imname =  dir_plots + "TB_IFG_"+str(IFG)+"_iCEOS_"+str(iCEOS)+"_RVM_"+ str(RVM)+"_RMW_"+ str(RMW)+"_RCH_"+str(RCH)+"_RTB_"+str(RTB)+"_"+sheet_n+"_corr.png"  

Var_C = Crit_val["Tb"]
VC_dic = {
    "Python" : Var_C[:,0],
    "Excel"  : Var_C[:,1],
    "Fortran": Var_C[:,2],
    "Data": Var_C[:,3]
    }

x = np.arange(len(Compuestos_n))  # the label locations
width = 0.20  # the width of the bars
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
plt.ylabel("Normal Boiling Temp.", fontsize=fs+2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
          fancybox=True, shadow=True, ncol=4, fontsize=fs-4)
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
    "Fortran": Var_C[:,2]
    }
Var_C = Crit_val["Ac_err"]/100
Errors = [ Var_C[:,0], Var_C[:,1], Var_C[:,2]]
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
    "Fortran"  : Var_C[:,2],
    "Data": Var_C[:,3]
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
    "Fortran_err"  : Var_C[:,2],
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

