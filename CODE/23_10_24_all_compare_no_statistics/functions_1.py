# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:24:09 2022

@author: gbdiaz
"""

import numpy as np
import random


# =============================================================================
# -------------------------  Variables globales  ----------------------------
# =============================================================================
global CNA, NA, RNA, T, TBI_1, TBI_2, VCI_1, VCI_2, MWI, HCI, Palfa
global A_GCVOL, B_GCVOL, C_GCVOL

RNA = 1
CNA = 9
TS = 288.7055556  

NA      = np.empty(9) 
MWI     = np.empty(9)
HCI     = np.empty(9)
TBI_1   = np.empty(9)
TBI_2   = np.empty(9)
TCI_1   = np.empty(9)    
TCI_2   = np.empty(9)    
PCI_1   = np.empty(9)    
PCI_2   = np.empty(9)
VCI_1   = np.empty(9)    
VCI_2   = np.empty(9)   
Palfa   = np.zeros((4,3)) 
A_GCVOL = np.empty(9)
B_GCVOL = np.empty(9)
C_GCVOL = np.empty(9)

TBI_1[0] = 23.58
TBI_1[1] = 22.88
TBI_1[2] = 21.74
TBI_1[3] = 18.25
TBI_1[4] = 26.73
TBI_1[5] = 49.26
TBI_1[6] = 27.15
TBI_1[7] = 21.78
TBI_1[8] = 21.32

TBI_2[0] = 0.8491
TBI_2[1] = 0.7141
TBI_2[2] = 0.2925
TBI_2[3] = -0.0671
TBI_2[4] = 0.8365
TBI_2[5] = 0.5229
TBI_2[6] = 0.8234
TBI_2[7] = 0.5946
TBI_2[8] = 0.0386

MWI[0] = 15.034820  
MWI[1] = 14.026880
MWI[2] = 13.01984
MWI[3] = 12.011
MWI[4] = 13.018940 
MWI[5] = 24.0220  
MWI[6] = 14.02688 
MWI[7] = 13.01894
MWI[8] = 12.011 

HCI[0] = 3
HCI[1] = 2
HCI[2] = 1
HCI[3] = 0
HCI[4] = 1
HCI[5] = 0
HCI[6] = 2
HCI[7] = 1
HCI[8] = 0 

A_GCVOL[0] = 1.643E-05
A_GCVOL[1] = 1.204E-05
A_GCVOL[2] = 7.2990E-06
A_GCVOL[3] = 8.780E-05
A_GCVOL[4] = 9.929E-06
A_GCVOL[5] = 39.370E-06
A_GCVOL[6] = 1.565E-05
A_GCVOL[7] = -5.295E-05
A_GCVOL[8] = 1.158E-04

B_GCVOL[0] = 5.562E-08
B_GCVOL[1] = 1.410E-08
B_GCVOL[2] = -2.606E-08
B_GCVOL[3] = -6.199E-07
B_GCVOL[4] = 1.741E-08
B_GCVOL[5] = -2.721E-07
B_GCVOL[6] = 5.985E-09
B_GCVOL[7] = 2.956E-07
B_GCVOL[8] = -7.670E-07

C_GCVOL[0] = 0.000E+00
C_GCVOL[1] = 0.000E+00
C_GCVOL[2] = 0.000E+00
C_GCVOL[3] = 8.822E-10
C_GCVOL[4] = 0.000E+00
C_GCVOL[5] = 2.492E-10
C_GCVOL[6] = 0.000E+00
C_GCVOL[7] =-3.138E-10
C_GCVOL[8] = 1.226E-09  

TCI_1[0] = 0.0141
TCI_1[1] = 0.0189
TCI_1[2] = 0.0164
TCI_1[3] = 0.0067
TCI_1[4] = 0.0082
TCI_1[5] = 0.021
TCI_1[6] = 0.01
TCI_1[7] = 0.0122
TCI_1[8] = 0.0042

TCI_2[0] = 1.7506
TCI_2[1] = 1.3327
TCI_2[2] = 0.596
TCI_2[3] = 0.0306
TCI_2[4] = 2.0337
TCI_2[5] = 0.8756
TCI_2[6] = 1.8815
TCI_2[7] = 1.102
TCI_2[8] = -0.2399

PCI_1[0] = -0.0012
PCI_1[1] = 0
PCI_1[2] = 0.002
PCI_1[3] = 0.0043
PCI_1[4] = 0.0011
PCI_1[5] = 0.0051
PCI_1[6] = 0.0025
PCI_1[7] = 0.0004
PCI_1[8] = 0.0061

PCI_2[0] = 0.018615
PCI_2[1] = 0.013547
PCI_2[2] = 0.007259
PCI_2[3] = 0.001219
PCI_2[4] = 0.00726
PCI_2[5] = 0.0011298
PCI_2[6] = 0.009884
PCI_2[7] = 0.007596
PCI_2[8] = 0.003268

VCI_1[0] = 65
VCI_1[1] = 56
VCI_1[2] = 41
VCI_1[3] = 27
VCI_1[4] = 41
VCI_1[5] = 59
VCI_1[6] = 48
VCI_1[7] = 38
VCI_1[8] = 27

VCI_2[0] = 68.35
VCI_2[1] = 56.28
VCI_2[2] = 37.5
VCI_2[3] = 16.01
VCI_2[4] = 42.39
VCI_2[5] = 35.71
VCI_2[6] = 49.24
VCI_2[7] = 44.95
VCI_2[8] = 33.32

NA[0] = 4
NA[1] = 3
NA[2] = 2
NA[3] = 1
NA[4] = 2
NA[5] = 1
NA[6] = 3
NA[7] = 2
NA[8] = 1

Palfa[0,0] =  0.48
Palfa[1,0] =  1.574
Palfa[2,0] = -0.176
Palfa[3,0] =  0
Palfa[0,1] =  0.37464
Palfa[1,1] =  1.54226
Palfa[2,1] = -0.26992
Palfa[3,1] =  0
Palfa[0,2] =  0.379642
Palfa[1,2] =  1.48503
Palfa[2,2] = -0.164423
Palfa[3,2] =  0.016666


# =============================================================================
# -------------------------  Funciones requeridas  ----------------------------
# ------- Funciones auxiliares para calcular la función objetivo   ------------
# =============================================================================
def TBS_f(MW,SG):
    TBS = 1071.28-94167*MW**(-0.03522)*SG**(3.266)*np.exp(-0.004922*MW-4.7695*SG+0.003462*MW*SG)
    return TBS

## Función que calcula la temperatura de ebullición
def TB_f(FG,IFG): 
    if (IFG == 1):
        TBI = TBI_1
        STB = np.sum(FG*TBI)
        TB = 198.2 + STB
    else:
        TBI = TBI_2
        STB = np.sum(FG*TBI)
        TB = 222.543 * np.log(STB)      
    return TB

## Función que calcula la relación hidrógeno Carbono de la fracción de petróleo
def CHC_f(HCE,FG): 
    CHC =  np.sum(FG * (HCI - HCE) / HCE)
    return CHC

## Función que calcula el error relativo de la temperatura de ebullición.
def CTB_f(TBE,IFG,v): 
    TBI   = np.empty(CNA)
    if (IFG == 1):
        TBI = TBI_1
        STB = np.dot(TBI,v)
        CTB = (198.2 - TBE + STB) / TBE
    else:
        TBI = TBI_2
        STB = np.dot(TBI,v)
        CTB = (STB - np.exp(TBE / 222.543)) / np.exp(TBE / 222.543)    
    return CTB

## Función que calcula el error relativo del volumen traslacional
def CSG_f(MWE,SGE,DH2O,FG): 
    DV = np.empty(CNA)
    VME = MWE / (100 ** 3 * DH2O * SGE)
    DV  = A_GCVOL + B_GCVOL * TS + C_GCVOL* TS** 2
    CSG = np.sum(np.dot(DV,FG))
    CSG = (CSG - VME)/VME
    return CSG

## Función que calcula el error relativo del peso molecular
def CMW_f(MWE,FG):
    CMW = np.sum(np.dot(FG,MWI))
    CMW = (CMW - MWE) / MWE
    return CMW

## Función auxiliar para la construcción del sistema lineal, en caso de tener 
## restricción de relación Hidrógeno-Carbono
def CHC_var(CNA, HCE):
    RHC = HCI-HCE
    RHS = 0 
    return HCI, HCE, RHC, RHS

## Función auxiliar para la construcción del sistema lineal, en caso de tener 
## restricción de Temperatura de ebullición
def TB_var(IFG, TBE):
    if (IFG == 1):
        TBI = TBI_1
        TBI[8] = 21.32
        RHS_TB = TBE - 198.2
        RTB = TBI
    else:
        TBI = TBI_2
        RHS_TB = np.exp(TBE/222.543)
        RTB = TBI
    return TBI, TBE, RTB, RHS_TB

## Función auxiliar para la construcción del sistema lineal, en caso de tener 
## restricción de peso molecular.
def CMW_var(N, MWE):
    RHS = MWE
    RMW = MWI
    return MWI, MWE, RMW,  RHS

## Función auxiliar para la construcción del sistema lineal, en caso de tener 
## restricción de vólumen traslacional.
def CSG_var(N,MWE,SGE,DH2O): 
    DV = np.empty(N)
    VME = MWE / (100 ** 3 * DH2O * SGE)
    DV  = A_GCVOL + B_GCVOL * TS + C_GCVOL* TS** 2
    RVM = DV
    RHS = VME
    return DV, VME, RVM, RHS

## Función que calcula la temperatura crítica
def TCGC_f(IFG,v,TBE):
    if IFG == 1:
        TCI = TCI_1
        ST = np.dot(v, TCI)
        if (ST <= 0.4825):
            TCGC = TBE / (0.584 + 0.965 * ST - ST ** 2)   
        else:
            TCGC = TBE / (0.96239389 - 0.446314966 / (ST + 2.583110349))
    else: 
        TCI = TCI_2
        ST = np.dot(v, TCI)
        TCGC = 231.239 * np.log(ST)
    return TCGC

## Función que calcula la presión crítica
def PCGC_f(IFG,v):
    if (IFG == 1):
        PCI = PCI_1
        SP = np.dot(v,PCI)
        SNA = np.dot(v,NA) 
        PCGC = (0.1 / (0.113 + 0.0032 * SNA - SP)**2)
    else:
        PCI = PCI_2
        SP = np.dot(v,PCI)
        PCGC = 0.1/(0.108998 + SP)**2 + 0.59827
    return PCGC

## Función que calcula el volumen crítico
def VCGC_f(IFG , FG):
    if IFG ==1:
        VCI = VCI_1
        SV =  np.dot(FG,VCI)
        VCGC = (17.5 + SV) / 100 ** 3
    else:
        VCI = VCI_2
        SV =  np.dot(FG,VCI)
        VCGC = (7.95 + SV) / 100 ** 3
    return VCGC

## Función que calcula el volumen crítico
def SGGC_f(MWE, v, DH2O):  
    DV = A_GCVOL + B_GCVOL * TS + C_GCVOL * TS ** 2
    SGGC = np.dot(DV,v)
    SGGC = MWE / (DH2O * SGGC * 100 ** 3) 
    return SGGC

# Función auxiliar para calcular la Función Objetivo
def Z_CEoS(iCEOS,U,V,ietp,AM,BM):
    # Resolver ecuación de tercer grado
    Z = np.empty(3)
    ETA = np.empty(2)
    C1 = BM*(U - 1) - 1
    C2 = BM**2*(V-U) - U*BM + AM
    C3 = -V*(BM**2 + BM**3) - AM*BM
    Q3 = (3*C2 - C1**2)/9
    U3 = (9*C1*C2 - 27*C3 - 2*C1**3)/54
    D3 = Q3**3 + U3**2
    if (D3 < 0):
        TETA = np.arccos(U3 / np.sqrt(-Q3**3))
        Z[0] = 2*np.sqrt(-Q3)*np.cos(TETA/3) - C1/3
        Z[1] = 2*np.sqrt(-Q3)*np.cos((TETA + 2*np.pi)/3) - C1/3
        Z[2] = 2*np.sqrt(-Q3)*np.cos((TETA + 4*np.pi)/3) - C1/3
        iext = 0 
    else:
        S3 = U3 + np.sqrt(D3)
        if (S3<0):
            S3 = -(np.abs(S3)) ** (1/3)
        else:
            S3 =  (np.abs(S3)) ** (1/3)
        
        T3 = U3 - np.sqrt(D3)
        if (T3 < 0):
            T3 = -(np.abs(T3)) ** (1 / 3)
        else:
            T3 =  (np.abs(T3)) ** (1 / 3)
        Z[0] = S3 + T3 - C1 / 3
        Z[1] = Z[0]
        Z[2] = Z[1]
        iext = 1
    # Identificación de la fase
    ZV = Z[0]
    ZL = Z[2]
    ZL = np.min(Z)
    ZV = np.max(Z)
    ETAL = BM / ZL
    ETAV = BM / ZV
    if (ETAL >= 1 or ETAL <= 0):
        ZL = ZV
        iext = 1
    ETA[0] = 0.25992105
    ETA[1] = 0.25307667
    if (iext == 1):
        MCL = 1 / (1 - ETAL) ** 2 - (AM / BM) * ETAL * (2 + U * ETAL) / (1 + U * ETAL + V * ETAL ** 2) ** 2
        if (ETAL > ETA[iCEOS-1] and MCL > 0.1 and ETAL < 1):
            EXPV = 1
            EXPL = 0
        else:
            EXPV = 0
            EXPL = 1
            ZT = ZV
            ZV = ZL
            ZL = ZT
    EXPV = 0
    EXPL = 0
    # Extrapolación de la fase
    if (EXPV == 1 or EXPL == 1):
        if (ietp == 1):
            if (EXPV == 1):
                ZV = np.sqrt(((S3 + T3) / 2 + C1 / 3) ^ 2 + (np.sqrt(3) * (S3 - T3) / 2) ** 2) 
            
            else:
                ZL = np.sqrt(((S3 + T3) / 2 + C1 / 3) ^ 2 + (np.sqrt(3) * (S3 - T3) / 2) ** 2) 
            if (ZL > ZV):
                ZT = ZL
                ZL = ZV
                ZV = ZT
                EXPT = EXPL
                EXPL = EXPV
                EXPV = EXPT
            
            if (ZL < BM):
                ZL = 1.01 * BM
                EXPV = 0
                EXPL = 1
    return ZL, EXPV, EXPL
     
## Función que calcula la función objetivo
def OFN(N,iCEOS,IFG,FG,TB):
    TCI = np.empty(N)
    PCI = np.empty(N)
    STB = np.empty(N)
    
    P = 101325          # Presión estándar [Pa]
    ST = 0
    SP = 0
    SNA = 0
    TCI   = np.empty(9)
    PCI   = np.empty(9)
    if (IFG == 1):
        TCI = TCI_1
        PCI = PCI_1
    else:
        TCI = TCI_2
        PCI = PCI_2
    ST  = ST + np.dot(FG,TCI)
    SP  = SP + np.dot(FG,PCI)
    SNA = SNA + np.dot(FG,NA)    
    if IFG == 1:
        if ST<= 0.4825:
            TC = TB/(0.584 + 0.965*ST - ST**2)
        else:
            TC = TB/(1 - 0.088208/ST)
        PC = (0.1/(0.113 + 0.0032*SNA - SP)**2)*1e6
    else:
        TC = 231.239 * np.log(ST)
        PC = (0.1/(0.108998 + SP)**2 + 0.59827)*1e6
    Tr  = TB/TC
    TAU = 1-Tr
    PR  = P/PC
    F0  = (-5.97616*TAU + 1.29874*TAU**1.5 - 0.60394*TAU**2.5 -1.06841*TAU**5)/Tr
    F1  = (-5.03365*TAU + 1.11505*TAU**1.5 - 5.41217*TAU**2.5 -7.46628*TAU**5)/Tr
    AF = (np.log(PR) - F0)/F1
    if iCEOS ==2:
        X = (-1 + (6*np.sqrt(2) + 8)**(1/3) - (6*np.sqrt(2) -8)**(1/3))/3
        OMGA = (8*(5*X+1))/(49-37*X)
        OMGB = X/(X+3)
        U =  2
        V = -1
    else:
        X = (2**(1/3)-1)/3
        OMGA = 1/(27*X)
        OMGB = X
        U = 1
        V = 0
    ALF = (np.sqrt(U**2 - 4*V) - U)/2
    BET = (np.sqrt(U**2 - 4*V) + U)/2
    EPS = np.log((1 + BET)/(1 - ALF))/(ALF + BET)
    if iCEOS == 2:
        if AF > 0.491:
            ALFA = (1 + (Palfa[0,2] + Palfa[1,2]*AF + Palfa[2,2]*AF**2 + Palfa[3,2]*AF**3)*(1 - np.sqrt(TS/TC)))**2
        else:
            ALFA = (1 + (Palfa[0,1] + Palfa[1,1]*AF + Palfa[2,1]*AF**2 + Palfa[3,1]*AF**3)*(1 - np.sqrt(TS/TC)))**2
    else:
        ALFA = (1 + (Palfa[0,0] + Palfa[1,0]*AF + Palfa[2,0]*AF**2 + Palfa[3,0]*AF**3)*(1-np.sqrt(TS/TC)))**2
    AM = OMGA*P*TC**2*ALFA/(PC*TS**2)
    BM = OMGB*P*TC/(PC*TS)
    ZL, EXPV, EXPL = Z_CEoS(iCEOS, U, V, 1, AM, BM )
    OFN = -np.log(ZL-BM) +(ZL -1) -AM/(BM *(ALF + BET)) * np.log((ZL + BM*BET)/(ZL-BM*ALF))
    return OFN

## Función que calcula el factor acéntrico
def ACGC_f(TB_py, Pc_py, Tc_py):
    P = .101325 
    Tr  = TB_py/Tc_py
    TAU = 1-Tr
    PR  = P/Pc_py
    F0  = (-5.97616*TAU + 1.29874*TAU**1.5 - 0.60394*TAU**2.5 -1.06841*TAU**5)/Tr
    F1  = (-5.03365*TAU + 1.11505*TAU**1.5 - 5.41217*TAU**2.5 -7.46628*TAU**5)/Tr
    AF = (-np.log(Pc_py/P) - F0)/F1
    return AF

def Ab_2x2 (R1, R2, RE1, RE2, v, idx_array, epMV, epMW, N):
    ks = [0,1,2,3,4,5,6,7,8]
    A = np.empty((2,2))          
    b = np.empty(2)
    k1 = np.random.choice(idx_array)
    k2 = np.random.choice(idx_array)
    detA = 0
    eMW = epsU(epMW)
    eMV = epsU(epMV)
    while k2==k1 or detA==0:
        k2 = np.random.choice(idx_array)
        A[0,0] = R1[k1]
        A[0,1] = R1[k2]
        A[1,0] = R2[k1]
        A[1,1] = R2[k2] 
        detA= np.linalg.det(A)
        
    sum_R1 = 0
    sum_R2 = 0
    for k in ks:
        if (k!=k1) and (k!=k2):
            sum_R1  += R1[k] * v[k]
            sum_R2  += R2[k] * v[k]
    
    b[0] = RE1*(eMW+1) - sum_R1
    b[1] = RE2*(eMV+1) - sum_R2
    v[k1],v[k2]= np.linalg.solve(A,b)  
    return v


#%%
def Ab_3x3 (R1, R2, R3, RE1, RE2, RE3, v, idx_array, epMV,epMW, epHC, N):
    ks = [0,1,2,3,4,5,6,7,8]
    A = np.empty((3,3))          # Coeficients matrix
    b = np.empty(3)
    k1 = np.random.choice(idx_array)
    k2 = np.random.choice(idx_array)  
    k3 = np.random.choice(idx_array)
    detA = 0
    eMW = epsU(epMW)
    eMV = epsU(epMV)
    eHC = epsU(epHC)
    while k2==k1 or k2==k3 or k3==k1 or detA == 0:
        k2 = np.random.choice(idx_array)
        k3 = np.random.choice(idx_array)
        A[0,0] = R1[k1]
        A[0,1] = R1[k2]
        A[0,2] = R1[k3]
        A[1,0] = R2[k1]
        A[1,1] = R2[k2]
        A[1,2] = R2[k3]
        A[2,0] = R3[k1]
        A[2,1] = R3[k2]
        A[2,2] = R3[k3]
        detA = np.linalg.det(A)
    sum_R1 = 0
    sum_R2 = 0
    sum_R3 = 0
    for k in ks:
        if (k!=k1) and (k!=k2) and (k!=k3):
            sum_R1  += R1[k] * v[k]
            sum_R2  += R2[k] * v[k]
            sum_R3  += R3[k] * v[k]
    b[0] = RE1*(eMW+1) - sum_R1
    b[1] = RE2*(eMV+1) - sum_R2
    b[2] = RE3*(eHC+1) - sum_R3
    
    v[k1],v[k2], v[k3] = np.linalg.solve(A,b)
    kst = np.array([k1,k2,k3])
    return v, A, kst

#%%
## Función para contruir el sistema lineal en caso de existir 4 restricciones
def Ab_4x4 (R1, R2, R3, R4, RE1, RE2, RE3, RE4, v, idx_array, epMV,epMW, epHC, epTB, N):
    ks = [0,1,2,3,4,5,6,7,8]
    A  = np.empty((4,4))         
    b  = np.empty(4)
    k1 = np.random.choice(idx_array)
    k2 = np.random.choice(idx_array)  
    k3 = np.random.choice(idx_array)
    k4 = np.random.choice(idx_array)
    detA = 0
    eMW = epsU(epMW)
    eMV = epsU(epMV)
    eHC = epsU(epHC)
    eTB = epsU(epTB)
    while k1==k2 or k1==k3 or k1==k4 or k2==k3 or k2==k4 or k3==k4 or detA == 0:
        k2 = np.random.choice(idx_array)
        k3 = np.random.choice(idx_array)
        k4 = np.random.choice(idx_array)
        A[0,0] = R1[k1]
        A[0,1] = R1[k2]
        A[0,2] = R1[k3]
        A[0,3] = R1[k4]
        A[1,0] = R2[k1]
        A[1,1] = R2[k2]
        A[1,2] = R2[k3]
        A[1,3] = R2[k4]
        A[2,0] = R3[k1]
        A[2,1] = R3[k2]
        A[2,2] = R3[k3]
        A[2,3] = R3[k4]
        A[3,0] = R4[k1]
        A[3,1] = R4[k2]
        A[3,2] = R4[k3]
        A[3,3] = R4[k4]
    sum_R1 = 0
    sum_R2 = 0
    sum_R3 = 0
    sum_R4 = 0
    for k in ks:
        if (k!=k1) and (k!=k2) and (k!=k3) and (k!=k4):
            sum_R1  += R1[k] * v[k]
            sum_R2  += R2[k] * v[k]
            sum_R3  += R3[k] * v[k]
            sum_R4  += R4[k] * v[k]
    b[0] = RE1*(eMW+1) - sum_R1
    b[1] = RE2*(eMV+1) - sum_R2
    b[2] = RE3*(eHC+1) - sum_R3
    b[3] = RE4*(eTB+1) - sum_R4
    v[k1], v[k2], v[k3], v[k4] = np.linalg.solve(A,b)
    return v
def epsU(ep):
    x= random.random()
    ep = 2*ep*x-ep
    return ep