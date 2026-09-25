import numpy as np
import time 
from scipy.ndimage import gaussian_filter
from Libs import Lib_General         as General
from Libs import Lib_Soft            as Soft
from Libs import Lib_OscBnd          as OscBnd
from Libs import Lib_Handle_Top      as Handle_Top
from Libs import Lib_FitRI           as FitRI
from Libs import Lib_StreamCollide   as StrColl
from Libs import Lib_IO              as IO
from Libs import Lib_Plots_from_Main as Plots_from_Main 
from Libs import Lib_Plots_for_GUI  as Plots_from_GUI


def Response_Is_Unstable(ResponseRaw, limit=1e7):
    """Detect divergence from the simulated response, not an early fit."""
    return (
        not np.isfinite(ResponseRaw.real)
        or not np.isfinite(ResponseRaw.imag)
        or np.abs(ResponseRaw) > limit
    )

def RingIn(SPs,FracVolSph,OscBndPars,\
        tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym,rhos,Do_Ref):
    nSph            = np.int64(SPs['nSph']) 
    nx              = np.int64(SPs['nx']) 
    ny              = np.int64(SPs['ny']) 
    nz              = np.int64(SPs['nz']) 
    n               = np.float64(SPs['n']) 
    om              = np.float64(SPs['om']) 
    tauInvBulk      = np.complex128(SPs['tauInvBulk'])
    UpdateMotionFac = np.float64(SPs['UpdateMotionFac'])
    if SPs['ProblemType'] == 'Cylinder2D':
        SphPoss = None
    else:
        SphPoss = np.float64(SPs['SphPoss'])
    dimensions      = SPs['dimensions']
    is_cylinder2d   = SPs['ProblemType'] == 'Cylinder2D'
    Do_Plot_RingIns = SPs['Do_Plot_RingIns']
    if Do_Ref and is_cylinder2d:
        raise ValueError('Cylinder2D does not use the planar QCM reference run')
    if Do_Ref : Increased_Precision_Fac = 0.1
    else      : Increased_Precision_Fac = 1
    if Do_Ref : 
        ZBulk      = SPs['ZBulk']
        Lambda_TRT = SPs['Lambda_TRT']
        tauInv_Ref = 1./tauInvBulk
        if Lambda_TRT != 0 :  
            tauInv_Asym_Ref = (4.-2.*tauInv_Ref)/(2.-tauInv_Ref+4*Lambda_TRT*tauInv_Ref)
        else : tauInv_Asym_Ref = tauInv_Ref
        one_m_tauInv_m_Iom_Ref      = 1 - tauInv_Ref - 1j*om
        one_m_tauInv_m_Iom_Asym_Ref = 1 - tauInv_Ref - 1j*om

        one_m_tauInv_m_Iom_Ref      += 5./2.*om**2
        one_m_tauInv_m_Iom_Asym_Ref += 5./2.*om**2
        
    
    nd,cxs,cys,czs,ibars,wi,i_ups,i_notups,i_downs,i_notdowns = General.ReadStencil(SPs['dimensions'])      

    nuBulk = (1./tauInvBulk-0.5)/3.
    if dimensions == 1: MatricesTop = np.nan
    if dimensions == 2:
        if is_cylinder2d:
            MatricesTop = np.nan
        else:
            MatricesTop = Handle_Top.Calc_MatricesTop_2D(nx,nuBulk,om)
    if dimensions == 3: MatricesTop = Handle_Top.Calc_MatricesTop_3D(nx,nz,nuBulk,om)
    FitInterval = int(SPs.get('RingInFitIntervalSteps',0))
    if FitInterval == 0: FitInterval = np.max([int(ny**2*0.05),10])
    PrintInterval = int(ny**2 * SPs['PrintIntervalFac'])
    PrintInterval = np.max([FitInterval,PrintInterval]) 
    FxCont_tot = 0
    if dimensions == 1: h = np.zeros((   ny,   nd),dtype = np.complex128)
    if dimensions == 2: h = np.zeros((nx,ny,   nd),dtype = np.complex128)
    if dimensions == 3: h = np.zeros((nx,ny,nz,nd),dtype = np.complex128)
    h1 = np.zeros((ny,nd),dtype = np.complex128)

    tbytRI_Extrapols = np.ones( 1000 * ny**2                 )*np.nan
    Response_Extrapols = np.ones( 1000 * ny**2   ,dtype=complex)*np.nan
    tbytRI_RIs       = np.ones( 1000 * ny**2                 )*np.nan
    Response_RIs     = np.ones( 1000 * ny**2   ,dtype=complex)*np.nan
    MotionPars_RI    = np.ones((1000 * ny**2,6),dtype=complex)*np.nan
    AuxPars_RI       = np.ones((1000 * ny**2,6),dtype=complex)*np.nan
    MotionParTitles = ['','','','','','']
    AuxParTitles    = ['','','','','','']

    time0=time.time(); countFits = 0; step = 0; Converged = False;
    if not Do_Ref and SPs['Do_OscBnd'] and is_cylinder2d:
        nLstot,xLs,yLs,OutsideLBMDomains,i_BCs,PoiLs,qs,uxLs,uyLs = \
            OscBnd.Extract_OscBndPars_Cylinder2D(OscBndPars)
    if not Do_Ref and SPs['Do_OscBnd'] and not is_cylinder2d: 
        iSs,nLs,nLstot,xLs,yLs,zLs,OutsideLBMDomains,InParticles,i_BCs,PoiLs,\
            qs,uxLs,uyLs,uzLs,OscBndAmps,SphRespPars,UpdateMotionFac,\
            OscBndLocked,OscBndLockedTo,nSph,RSph,ySphbyR,rhoSph,iSiL_Lists = \
            OscBnd.Extract_OscBndPars(OscBndPars)
    B = None
    if SPs['ProblemType'] == 'SoftParticles':
        B = Soft.compute_B(nx, ny, nz, int(SPs['nSph']), SPs['RSph'], SphPoss[0], SphPoss[1], SphPoss[2])
    while not Converged: 
        if Do_Ref : 
            h1,Fx_on_Wall = StrColl.FD_LBM_Step_Ref(h1,ny,nd,cxs,cys,czs,wi,ibars,\
                    i_ups,i_notups,i_downs,i_notdowns,ZBulk,om,\
                    tauInv_Ref,tauInv_Asym_Ref,one_m_tauInv_m_Iom_Ref,one_m_tauInv_m_Iom_Asym_Ref)
            MotionPars = np.ones(6,dtype = complex)*np.nan            
            AuxPars    = np.ones(6,dtype = complex)*np.nan            
        if not Do_Ref: 
            if SPs['Do_OscBnd']  : 
                if SPs['dimensions'] == 3: 
                    h,Fx_on_Wall,FxLs,FyLs,FzLs = \
                        StrColl.FreqDLBMStep_OscBnd_3D(h,nx,ny,nz,nd,cxs,cys,czs,wi,ibars,MatricesTop,\
                            nLstot,OutsideLBMDomains,i_BCs,PoiLs,qs,uxLs,uyLs,uzLs,\
                            tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym)
                    uxLs,uyLs,uzLs,FxCont_tot,OscBndAmps,MotionPars,MotionParTitles,\
                        AuxPars,AuxParTitles = \
                        OscBnd.Update_Motion_3D(nLs,nLstot,iSs,xLs,yLs,zLs,uxLs,uyLs,uzLs,\
                            FxLs,FyLs,FzLs,nSph,om,\
                            OscBndAmps,SphPoss,SphRespPars,\
                            UpdateMotionFac,OscBndLocked,OscBndLockedTo,iSiL_Lists,n,SPs)
                if SPs['dimensions'] == 2:
                    if is_cylinder2d:
                        h,FxLs,FyLs = \
                            StrColl.FreqDLBMStep_OscBnd_Cylinder2D(h,nx,ny,nd,cxs,cys,wi,ibars,\
                                nLstot,OutsideLBMDomains,i_BCs,PoiLs,qs,uxLs,uyLs,\
                                tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym)
                        uxLs,uyLs,FxCyl,FzCyl,MotionPars,MotionParTitles,AuxPars,AuxParTitles = \
                            OscBnd.Update_Motion_Cylinder2D(nLstot,FxLs,FyLs,uxLs,uyLs)
                        Fx_on_Wall = np.zeros(nx,dtype=np.complex128)
                    else:
                        h,Fx_on_Wall,FxLs,FyLs = \
                            StrColl.FreqDLBMStep_OscBnd_2D(h,nx,ny,nd,cxs,cys,czs,wi,ibars,MatricesTop,\
                                nLstot,OutsideLBMDomains,i_BCs,PoiLs,qs,uxLs,uyLs,uzLs,\
                                tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym)
                        uxLs,uyLs,FxLiq,MotionPars,MotionParTitles,AuxPars,AuxParTitles = \
                            OscBnd.Update_Motion_2D(nLstot,FxLs)
            if not SPs['Do_OscBnd']  : 
                if SPs['ProblemType'] == 'SoftParticles' : 
                    h,Fx_on_Wall = \
                        StrColl.FreqDLBMStep_SoftPt_3D(h,nx,ny,nz,\
                            nd,cxs,cys,czs,wi,ibars,i_ups,i_notups,i_downs,i_notdowns,MatricesTop,\
                            tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym)
                if SPs['ProblemType'] == 'FilmResonance' : 
                    h,Fx_on_Wall = \
                        StrColl.FreqDLBMStep_SoftPt_1D(h,nx,ny,nz,nd,cxs,cys,czs,wi,ibars,\
                                i_ups,i_notups,i_downs,i_notdowns,om,MatricesTop,\
                                tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym)
                MotionPars = np.ones(6,dtype = complex)*np.nan            
                AuxPars    = np.ones(6,dtype = complex)*np.nan            
        if is_cylinder2d and not Do_Ref:
            ResponseRaw = FxCyl
        else:
            ResponseRaw = General.Calc_Dfcbyn(SPs,Fx_on_Wall,FxCont_tot,Do_Ref)
        if step >= len(tbytRI_RIs) - 1: 
            tbytRI_RIs       = np.append(tbytRI_RIs      ,np.ones( 1000 * ny**2                  )*np.nan)
            Response_RIs     = np.append(Response_RIs    ,np.ones( 1000 * ny**2   ,dtype= complex)*np.nan)
            tbytRI_Extrapols = np.append(tbytRI_Extrapols,np.ones( 1000 * ny**2                  )*np.nan)
            Response_Extrapols = np.append(Response_Extrapols,np.ones(1000 * ny**2,dtype=complex)*np.nan)
            MotionPars_RI    = np.append(MotionPars_RI   ,np.ones((1000 * ny**2,6),dtype= complex)*np.nan)
            AuxPars_RI       = np.append(AuxPars_RI      ,np.ones((1000 * ny**2,6),dtype= complex)*np.nan)

        tbytRI_RIs[step] = float(step)/ny**2
        if Do_Ref or is_cylinder2d : Response_RIs[step] = ResponseRaw
        else                       : Response_RIs[step] = ResponseRaw - SPs['Dfcbyn_Ref']
        
        # with open(SPs['folder']+'\\errors1.txt', "a") as f:
        #     print('step               : ', step, file=f) 
        #     print('MotionPars         : ', MotionPars, file=f) 
        #     print('MotionPars_RI[step]: ', MotionPars_RI[step], file=f)             

        MotionPars_RI[step] = MotionPars 
     
        if step%FitInterval == 0 and step >= max(FitInterval,2):
            if SPs['SigSmoothDfcbynsFac'] > 0 and step > 2*ny**2: 
                sig = step*SPs['SigSmoothDfcbynsFac']   
                i_ini =      int(np.max([10*sig,step/3.]))
                i_fin = step-int(10*sig)
                tbytRI_RIs4Fit = tbytRI_RIs[i_ini:i_fin]
                Response_RIs4Fit = gaussian_filter(Response_RIs,sigma=sig)
                Response_RIs4Fit = Response_RIs4Fit[i_ini:i_fin]
            else : 
                i_ini = int(step/3.)
                tbytRI_RIs4Fit = tbytRI_RIs[i_ini:step]
                Response_RIs4Fit = Response_RIs[i_ini:step]     
            try :     
                Response_Extrapol,StdErr_Extrapol,amplitude,om_complex,Response_RI_Fit = \
                    FitRI.Fit_RI(tbytRI_RIs4Fit,Response_RIs4Fit)
            except : 
                Response_Extrapol = np.nan
                amplitude       = np.nan
                om_complex      = np.nan
                Response_RI_Fit = np.nan
            countFits += 1    
            tbytRI_Extrapols[countFits] = step/ny**2
            Response_Extrapols[countFits] = Response_Extrapol
            
        if step%PrintInterval == 0 and step > 10:
            DriftFitResults40perc,DriftFitResults20perc = \
                FitRI.Calc_DriftFitResults(SPs,tbytRI_Extrapols,Response_Extrapols,\
                    amplitude,om_complex,countFits)
            if not Do_Ref : 
                if Do_Plot_RingIns :  
                    if SPs['Do_from_GUI'] :  
                        Plots_from_GUI.Plot_RI(tbytRI_RIs4Fit, Response_RIs4Fit, Response_RI_Fit,tbytRI_Extrapols, Response_Extrapols,countFits, DriftFitResults40perc, DriftFitResults20perc, SPs)      
                    else :  
                        Plots_from_Main.Plot_RI(tbytRI_RIs4Fit,Response_RIs4Fit,Response_RI_Fit,tbytRI_Extrapols,Response_Extrapols,countFits)                    
                        
                if SPs['Do_Plot_MotionPars'] and SPs['ProblemType'] == 'StiffParticles':                                                          
                    if SPs['Do_from_GUI'] :  
                        Plots_from_GUI.Plot_MotionPars_RI( tbytRI_Extrapols[int(countFits/3.):countFits], MotionPars_RI[int(countFits/3.):countFits],MotionParTitles,SPs)
                    else :  
                        Plots_from_Main.Plot_MotionPars_RI(tbytRI_Extrapols[int(countFits/3.):countFits], MotionPars_RI[int(countFits/3.):countFits],MotionParTitles,SPs)

                print('t/t_RI',np.round(step/ny**2,2),\
                      'Drift40%',np.round(np.abs(DriftFitResults40perc),3),\
                      'Drift20%',np.round(np.abs(DriftFitResults20perc),3),)
            if np.abs(DriftFitResults40perc.real) < SPs['TargetSlopeFitResults']*Increased_Precision_Fac and \
               np.abs(DriftFitResults40perc.imag) < SPs['TargetSlopeFitResults']*Increased_Precision_Fac and \
               np.abs(DriftFitResults20perc.real) < SPs['TargetSlopeFitResults']*Increased_Precision_Fac and \
               np.abs(DriftFitResults20perc.imag) < SPs['TargetSlopeFitResults']*Increased_Precision_Fac : 
                   Converged = True; 
                   CompTimeMins = np.round((time.time()-time0)/60,2); 
                   SPs['ProblemFlag']     = 0
                   SPs['steps']           = step
                   SPs['tbytRI']          = step/ny**2
                   if is_cylinder2d:
                       SPs['CylinderForcePerLengthXOnCylinderByLiquid_LBM'] = Response_Extrapol
                       SPs['Dfratio']     = np.nan
                   else:
                       SPs['Dfcbyn_Extrapol'] = Response_Extrapol
                       SPs['Dfratio']     = Response_Extrapol.imag / (-Response_Extrapol.real)
                   SPs['CompTimeMins'] = CompTimeMins
            if Response_Is_Unstable(ResponseRaw) or step/ny**2 > SPs['MaxtbytRI']:
                CompTimeMins = np.round((time.time()-time0)/60,2);
                print('Raw response diverged or tbytRI > MaxtbytRI',SPs['MaxtbytRI'])
                #UPDATED
                if SPs['Do_from_GUI']: Plots_from_GUI.SimError(SPs)
                SPs['ProblemFlag']  = 1
                dr = np.ones(((nx,ny,nz)))*np.nan
                ux = np.ones(((nx,ny,nz)))*np.nan
                uy = np.ones(((nx,ny,nz)))*np.nan
                uz = np.ones(((nx,ny,nz)))*np.nan
                break
        step += 1    
        
    if Converged : 
        if Do_Ref :         
            SPs['Dfcbyn_Ref'] = Response_Extrapol
            SPs['StdErr_Ref'] = StdErr_Extrapol
            SPs['Dfratio_Ref'] = Response_Extrapol.imag/(-Response_Extrapol.real)
            dr = np.ones(ny)*np.nan
            ux = np.sum([h]*cxs,axis=1)
            uy = np.ones(ny)*np.nan
            uz = np.ones(ny)*np.nan
        if not Do_Ref : 
            if is_cylinder2d:
                dr = np.sum(h,axis=2,dtype=np.complex128)
                ux = np.sum(h*cxs,axis=2,dtype=np.complex128)
                uy = np.sum(h*cys,axis=2,dtype=np.complex128)
                uz = np.zeros((nx,ny),dtype=np.complex128)
                SPs['CylinderForcePerLengthZOnCylinderByLiquid_LBM_Last'] = FzCyl
                if SPs['CylUx_LBM'] != 0:
                    # Resistance opposes the imposed cylinder velocity.
                    SPs['CylinderFrictionPerLength_LBM'] = -Response_Extrapol / SPs['CylUx_LBM']
                else:
                    SPs['CylinderFrictionPerLength_LBM'] = np.nan
            if SPs['ProblemType'] == 'SoftParticles' : 
                dr,ux,uy,uz = Soft.Calc_dr_ux_uy_uz_SoftPt_3D(h,cxs,cys,czs,nx,ny,nz)
                # MotionPars,MotionParTitles,AuxPars,AuxParTitles = \
                    # Soft.Calc_MotionPars_3D(nx,ny,nz,nd,cxs,cys,czs,i_ups,i_notups,ibars,wi,h,\
                    #                  tauInvs,FracVolSph,SPs,SphPoss)
                MotionPars,MotionParTitles,AuxPars,AuxParTitles = \
                    Soft.Calc_MotionPars_3D_UU(B,nx,ny,nz,nd,cxs,cys,czs,i_ups,i_notups,ibars,wi,h,\
                                     tauInvs,FracVolSph,SPs,SphPoss)
                MotionPars_RI[step] = MotionPars
                AuxPars_RI[   step] = AuxPars
            if SPs['ProblemType'] in ['StiffParticles','Roughness','SFA','CylinderQCM3D']:
                dr,ux,uy,uz = \
                    OscBnd.Calc_dr_ux_uy_uz_OscBnd(h,cxs,cys,czs,\
                        OutsideLBMDomains,InParticles,OscBndAmps,SPs)
            if SPs['ProblemType'] == 'FilmResonance': 
                dr = np.zeros(ny)
                ux = np.ones(ny,dtype=np.complex128)*np.nan
                for y in range(ny): ux[y] = np.sum(h[y]*cxs)
                uy = np.zeros(ny)
                uz = np.zeros(ny)
        
            if SPs['ProblemType'] in ['SoftParticles','StiffParticles','Roughness','SFA'] or \
               (SPs['ProblemType'] == 'CylinderQCM3D' and SPs['Do_SavePlots']):
                if SPs['Do_from_GUI'] :  
                    Plots_from_GUI.Plot_Fields_Horizontal(dr,ux,uy,uz,'Re($\Delta \\rho$)', 'Re(u$_{\mathrm{x}}$)','Re(u$_{\mathrm{y}}$)', 'Re(u$_{\mathrm{z}}$)',SPs,int(SPs['RSph']))
                    Plots_from_GUI.Plot_Fields_Vertical(dr,ux,uy,uz, 'Re($\Delta \\rho$)', 'Re(u$_{\mathrm{x}}$)','Re(u$_{\mathrm{y}}$)','Re(u$_{\mathrm{z}}$)',SPs)
                else : 
                    Plots_from_Main.Plot_Fields_Horizontal(dr,ux,uy,uz,'Re($\Delta \\rho$)', 'Re(u$_{\mathrm{x}}$)','Re(u$_{\mathrm{y}}$)', 'Re(u$_{\mathrm{z}}$)',SPs,int(SPs['RSph']))
                    Plots_from_Main.Plot_Fields_Vertical(dr,ux,uy,uz, 'Re($\Delta \\rho$)', 'Re(u$_{\mathrm{x}}$)','Re(u$_{\mathrm{y}}$)','Re(u$_{\mathrm{z}}$)',SPs)
                    


                # Plots.Plot_Top(ux,uy,uz,uz,\
                #     'Re(u$_{\mathrm{x}}$)',\
                #     'Re(u$_{\mathrm{y}}$)',\
                #     'Re(u$_{\mathrm{z}}$)','',SPs)


            if is_cylinder2d and SPs['Do_SavePlots']:
                Plots_from_Main.Plot_Fields_Cylinder2D(ux,uy,SPs)

            if SPs['ProblemType'] == 'FilmResonance' :
                pass #Plots.Plot_DisplacementField_1D(h,SPs)

            MotionParsDict,AuxDict = IO.Make_MotionParsDict_AusParsDict(\
                MotionPars_RI[step-1],MotionParTitles,\
                AuxPars_RI[   step-1],AuxParTitles)
            ResultLabel = 'CylinderForcePerLengthXOnCylinderByLiquid_LBM' if is_cylinder2d else 'Dfcbyn'
            print(SPs['iavg'],SPs['iPar1'],SPs['iPar2'],SPs['iPar3'],SPs['iovt'],\
                ResultLabel,np.round(Response_Extrapol,3),'CompTimeMins',SPs['CompTimeMins'])
            IO.Write_Config(SPs); IO.Save(SPs,MotionParsDict,AuxDict)    
    return   
