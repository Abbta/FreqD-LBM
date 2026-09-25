import numpy as np
from Libs import Lib_General   as General
from Libs import Lib_RingIn    as RingIn
from Libs import Lib_Soft      as Soft
from Libs import Lib_OscBnd    as OscBnd
from Libs import Lib_SetPos_3D as SetPos_3D

def Handle_Geometry_CylinderQCM3D(SPs):
    dx_nm = float(SPs['Dx_nm'])
    radius_nm = float(SPs['RCyl_nm'])
    cylinder_height_nm = float(SPs['HCyl_nm'])
    width_nm = float(SPs['CylBoxWidth_nm'])
    box_height_nm = float(SPs['CylBoxHeight_nm'])

    if min(dx_nm, radius_nm, cylinder_height_nm, width_nm, box_height_nm) <= 0:
        raise ValueError('Cylinder dimensions and Dx_nm must be positive')
    if SPs['CylBoundaryCondition'] != 'PeriodicXZ':
        raise ValueError("CylinderQCM3D requires CylBoundaryCondition='PeriodicXZ'")

    width_lattice = width_nm / dx_nm
    height_lattice = box_height_nm / dx_nm
    nx = int(np.round(width_lattice))
    ny = int(np.round(height_lattice))
    if not np.isclose(width_lattice, nx) or not np.isclose(height_lattice, ny):
        raise ValueError('Cylinder box dimensions must be integer multiples of Dx_nm')

    radius = radius_nm / dx_nm
    cylinder_height = cylinder_height_nm / dx_nm
    if 2.0 * radius >= nx:
        raise ValueError('Cylinder diameter must be smaller than the lateral period')
    if cylinder_height >= ny - 1:
        raise ValueError('Cylinder top must leave at least one liquid node below the upper boundary')

    SPs['Width_nm'] = width_nm
    SPs['Height_nm'] = box_height_nm
    SPs['RCyl'] = radius
    SPs['HCyl'] = cylinder_height
    # Compatibility aliases let the established one-particle 3D path carry the cylinder.
    SPs['nCyl'] = SPs['nSph'] = 1
    SPs['RSph'] = radius
    SPs['ySphbyR'] = cylinder_height/(2.*radius)
    SPs['nx'] = SPs['nz'] = nx
    SPs['ny'] = ny
    SPs['CylPoss'] = SetPos_3D.Set_CylPoss(SPs)
    SPs['SphPoss'] = SPs['CylPoss']
    SPs['nNodes'] = nx * ny * nx
    SPs['CoverageTrue'] = np.pi*radius**2/nx**2

def Handle_Geometry_Cylinder2D(SPs):
    dx_nm = float(SPs['Dx_nm'])
    radius_nm = float(SPs['RCyl_nm'])
    width_nm = float(SPs['CylBoxWidth_nm'])
    height_nm = float(SPs['CylBoxHeight_nm'])

    if min(dx_nm, radius_nm, width_nm, height_nm) <= 0:
        raise ValueError('Cylinder dimensions and Dx_nm must be positive')

    nx_float = width_nm / dx_nm
    ny_float = height_nm / dx_nm
    nx = int(np.round(nx_float))
    ny = int(np.round(ny_float))
    if not np.isclose(nx_float, nx) or not np.isclose(ny_float, ny):
        raise ValueError('Cylinder box dimensions must be integer multiples of Dx_nm')

    radius = radius_nm / dx_nm
    if 2.0 * radius >= min(nx, ny):
        raise ValueError('The cylinder diameter must be smaller than both box dimensions')

    SPs['Width_nm'] = width_nm
    SPs['Height_nm'] = height_nm
    SPs['RCyl'] = radius
    SPs['nx'] = nx
    SPs['ny'] = ny
    SPs['nz'] = 1
    SPs['CylPos'] = np.array([(nx - 1) / 2.0, (ny - 1) / 2.0])
    SPs['nNodes'] = nx * ny
    SPs['CoverageTrue'] = np.pi * radius**2 / (nx * ny)

def Handle_Geometry_FilmResonance(SPs): # FilmResonance option not debugged
    SPs['Width_nm']     = np.nan
    SPs['RSph']         = np.nan
    SPs['nx'] = SPs['nz'] = 1
    SPs['ny']           = int(np.round(SPs['FilmThickness']/SPs['Dx_nm'])) + 3
    SPs['SphPoss']      = [[np.nan],[np.nan],[np.nan]]
    SPs['nNodes']       = SPs['ny']
    SPs['CoverageTrue'] = np.nan

def Handle_Geometry_Roughness(SPs):  # Roughness option not debugged
    SPs['Width_nm']     = SPs['Roughn_Width_nm']
    SPs['RSph']         = np.nan
    nx_float            = SPs['Width_nm']/SPs['Dx_nm']
    SPs['nx']           = int(nx_float/2)*2+1  #odd
    SPs['nz']           = 1
    SPs['Roughn_VertScale'] = SPs['Roughn_VertScale_nm'] / SPs['Dx_nm']
    SPs['ny']           = int(np.round(2.*SPs['Roughn_VertScale'])+\
                              2.*SPs['Roughn_VertScale']*SPs['Gap2TopbyR'])
    SPs['SphPoss']      = [[np.nan],[np.nan],[np.nan]]
    SPs['nNodes']       = SPs['nx']*SPs['ny']*SPs['nz']
    SPs['CoverageTrue'] = np.nan

def Handle_Geometry_SFA(SPs): # SFA option not debugged
    SPs['Width_nm']     = SPs['Roughn_Width_nm']
    SPs['RSph']         = np.nan
    nx_float            = SPs['Width_nm']/SPs['Dx_nm']
    SPs['nx']           = int(nx_float/2)*2+1  #odd
    SPs['nz']           = 1
    SPs['Roughn_VertScale'] = SPs['Roughn_VertScale_nm'] / SPs['Dx_nm']
    SPs['ny']           = int(np.round(2.*SPs['Roughn_VertScale'])+\
                              2.*SPs['Roughn_VertScale']*SPs['Gap2TopbyR'])
    SPs['SphPoss']      = [[np.nan],[np.nan],[np.nan]]
    SPs['nNodes']       = SPs['nx']*SPs['ny']*SPs['nz']
    SPs['CoverageTrue'] = np.nan

def Handle_Geometry_Spheres(SPs):
    SPs['Width_nm']     = (SPs['nSph']*np.pi*SPs['RSph_nm']**2/SPs['CovTarget'])**0.5
    SPs['RSph']         = SPs['RSph_nm']/SPs['Dx_nm']
    nx_float            = SPs['Width_nm']/SPs['Dx_nm']
    SPs['nx']=SPs['nz'] = int(nx_float/2)*2+1  #odd
    SPs['ny']           = int(np.round(SPs['RSph']*SPs['ySphbyR']+\
                              SPs['RSph']*(1.+SPs['Gap2TopbyR'])))
    SPs['SphPoss']      = SetPos_3D.Set_SphPoss(SPs)
    SPs['nNodes']       = SPs['nx']*SPs['ny']*SPs['nz']
    SPs['CoverageTrue'] = SPs['nSph']*np.pi*SPs['RSph_nm']**2/(SPs['nx']*SPs['Dx_nm'])**2

def SingleSimulation(SPs):
    # ns=SPs['ns']
    # if SPs['ProblemType'] in ['SoftParticles','StiffParticles'] :  Handle_Geometry_Spheres(SPs)
    # if SPs['ProblemType'] == 'SFA'            : Handle_Geometry_SFA(SPs)
    # if SPs['ProblemType'] == 'Roughness'      : Handle_Geometry_Roughness(SPs)
    # if SPs['ProblemType'] == 'FilmResonance'  : Handle_Geometry_FilmResonance(SPs)

    # OscBndPars = OscBnd.Setup_Boundaries_3D(SPs)

    OscBndPars = SPs['OscBndPars']

    print('nx,ny,nz',SPs['nx'],SPs['ny'],SPs['nz'],'n',SPs['n'])
        # nu_for_om = 1./6.
        # SPs['delta'] = SPs['delta0_nm'] / SPs['Dx_nm'] / SPs['n']**0.5
        # SPs['om']    = 2*nu_for_om / SPs['delta']**2

    General.Calc_tauInvBulk_ZBulk(SPs)
    General.Calc_etaabstandel(SPs)
    if SPs['ProblemType'] == 'StiffParticles' : OscBnd.Calc_SphRespPars_3D(SPs,OscBndPars)
    if SPs['ProblemType'] == 'CylinderQCM3D' : OscBnd.Calc_CylRespPars_3D(SPs,OscBndPars)

    FracVolSph,tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym,rhos = \
        Soft.Set_RelaxPars(SPs)

    if SPs['ProblemType'] != 'Cylinder2D':
        RingIn.RingIn(SPs,FracVolSph,OscBndPars,\
            tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym,rhos,Do_Ref = True)
        if SPs.get('ProblemFlag') != 0:
            raise RuntimeError('Reference ring-in failed before the loaded simulation')
        print('Dfcbyn_Ref' ,np.round(SPs['Dfcbyn_Ref' ],3),\
                 'Dfratio_Ref',np.round(SPs['Dfratio_Ref'],3))
    else:
        SPs['Dfcbyn_Ref'] = 0
        SPs['Dfratio_Ref'] = np.nan
    RingIn.RingIn(SPs,FracVolSph,OscBndPars,\
        tauInvs,tauInvs_Asym,one_m_tauInvs_m_Iom,one_m_tauInvs_m_Iom_Asym,rhos,Do_Ref = False)
