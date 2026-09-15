"""Run FreqD-LBM from a JSON configuration.

Usage:
    python Main_FreqDLBM_config_file.py path/to/config.json

The JSON file is a flat object. Every entry is optional because ``DEFAULTS``
contains a value for every input. Unknown entries are rejected.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from Libs import Lib_IO as IO
from Libs import Lib_OscBnd as OscBnd
from Libs import Lib_SingleSim as Single_Sim

DEFAULTS = {
    "ProblemType": "StiffParticles",
    "Dx_nm": 1.0,
    "folder": "test",
    "fname0": "test",
    "Do_from_GUI": False,
    "VEPars_Choice": "etaabs_tandel",
    "etaabsBulk": 1.0,
    "tandelBulk": 1e33,
    "f0_SI": 5e6,
    "Zq_SI": 8.8e6,
    "delta0_nm": None,
    "Gap_P2P": 1.0,
    "rhoSph": 1.0,
    "Gap2TopbyR": 1.5,
    "betap_Sph": 0.0,
    "betapp_Sph": 0.0,
    "Jp_FacSph": 10.0,
    "Jpp_FacSph": 10.0,
    "MaxwellRelaxRate_MHz": 1.0,
    "UpdateMotionFac": 0.02,
    "Do_SavePlots": False,
    "Do_Plot_MotionPars": True,
    "Do_Plot_RingIns": True,
    "PrintIntervalFac": 1.0,
    "nSph": 3,
    "TargetSlopeFitResults": 20.0,
    "MaxtbytRI": 100.0,
    "SigSmoothDfcbynsFac": 1e-2,
    "Lambda_TRT": 0.25,
    "Do_UseQuadraticTerm": True,
    "Do_Allow_rhoUneq1": False,
    "Roughn_VertScale_nm": 3.0,
    "Roughn_HoriScale_nm": 5.0,
    "Roughn_Width_nm": 100.0,
    "Single_Wave": True,
    "FilmThickness_nm": 5.0,
    "RSph_nm": 5.0,
    "ySphbyR": 0.9,
    "CovTarget": 0.3,
    "etaabscenSphmPas": 1e4,
    "tandelcenSph": 0.1,
    "RCyl_nm": 100.0,
    "CylBoxWidth_nm": 4000.0,
    "CylBoxHeight_nm": 4000.0,
    "CylBoundaryCondition": "PeriodicXY",
    "CylUx_LBM": 1.0,
    "CylUz_LBM": 0.0,
    "n": 7,
    "dimensions": None,
    "Do_OscBnd": None,
    "OscBndLocked": None,
    "OscBndLockedTo": None,
}


PROBLEM_DEFAULTS = {
    "SoftParticles": {
        "dimensions": 3,
        "Do_OscBnd": False,
        "OscBndLocked": False,
        "OscBndLockedTo": "Zero",
    },
    "StiffParticles": {
        "dimensions": 3,
        "Do_OscBnd": True,
        "OscBndLocked": False,
        "OscBndLockedTo": "Zero",
    },
    "SFA": {
        "dimensions": 3,
        "Do_OscBnd": True,
        "OscBndLocked": True,
        "OscBndLockedTo": "Zero",
    },
    "Roughness_3D": {
        "dimensions": 3,
        "Do_OscBnd": True,
        "OscBndLocked": True,
        "OscBndLockedTo": "Substrate",
    },
    "Roughness_2D": {
        "dimensions": 2,
        "Do_OscBnd": True,
        "OscBndLocked": True,
        "OscBndLockedTo": "Substrate",
    },
    "FilmResonance": {
        "dimensions": 1,
        "Do_OscBnd": False,
        "OscBndLocked": False,
        "OscBndLockedTo": "Zero",
    },
    "Cylinder2D": {
        "dimensions": 2,
        "Dx_nm": 5.0,
        "Do_OscBnd": True,
        "OscBndLocked": True,
        "OscBndLockedTo": "Prescribed",
        "Do_SavePlots": False,
        "Do_Plot_MotionPars": False,
        "Do_Plot_RingIns": False,
        "nSph": 1,
    },
}


RESULT_PARAMETERS = {
    "default": ("etaabscenSphmPas", "tandelcenSph", "CovTarget"),
    "Cylinder2D": ("RCyl_nm", "CylBoxWidth_nm", "CylBoxHeight_nm"),
}


def load_config(path):
    """Load JSON overrides and resolve all defaults without running a simulation."""
    path = Path(path)
    with path.open(encoding="utf-8-sig") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError("The configuration root must be a JSON object")
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration entries: {sorted(unknown)}")
    collections = [key for key, value in overrides.items() if isinstance(value, (list, dict))]
    if collections:
        raise ValueError(f"Single-simulation parameters must be scalar: {sorted(collections)}")

    problem_type = overrides.get("ProblemType", DEFAULTS["ProblemType"])
    if problem_type not in PROBLEM_DEFAULTS:
        raise ValueError(
            f"Unknown ProblemType {problem_type!r}; choose from {sorted(PROBLEM_DEFAULTS)}"
        )
    config = copy.deepcopy(DEFAULTS)
    config.update(PROBLEM_DEFAULTS[problem_type])
    config.update(overrides)
    if config["delta0_nm"] is None:
        config["delta0_nm"] = 252.0 * config["etaabsBulk"] ** 0.5
    _validate_config(config)
    return config


def _validate_config(config):
    if isinstance(config["n"], bool) or not isinstance(config["n"], int) or config["n"] < 1:
        raise ValueError("n must be a positive integer")
    for key in ("Dx_nm", "f0_SI", "Zq_SI", "delta0_nm"):
        if not isinstance(config[key], (int, float)) or config[key] <= 0:
            raise ValueError(f"{key} must be positive")


def initialize_sps(config):
    """Create the SP dictionary expected by the existing FreqD-LBM libraries."""
    sps = copy.deepcopy(config)
    # The legacy result writer and material conversion expect these fields.
    # Each contains only the current value; this entrypoint never sweeps them.
    sps["ns"] = np.asarray([sps["n"]])
    parameters = RESULT_PARAMETERS.get(sps["ProblemType"], RESULT_PARAMETERS["default"])
    for index, parameter in enumerate(parameters, start=1):
        sps[f"Par{index}str"] = parameter
        sps[parameter + "s"] = np.asarray([sps[parameter]])
        sps[f"nPar{index}"] = 1
        sps[f"iPar{index}"] = 0
    sps["novt"] = 1
    sps["iovt"] = 0
    sps["navg"] = 1
    sps["iavg"] = 0
    return sps


def _prepare_geometry(sps):
    problem_type = sps["ProblemType"]
    if problem_type in {"SoftParticles", "StiffParticles"}:
        Single_Sim.Handle_Geometry_Spheres(sps)
    elif problem_type == "Cylinder2D":
        Single_Sim.Handle_Geometry_Cylinder2D(sps)
    elif problem_type == "SFA":
        Single_Sim.Handle_Geometry_SFA(sps)
    elif problem_type in {"Roughness_2D", "Roughness_3D"}:
        Single_Sim.Handle_Geometry_Roughness(sps)
    elif problem_type == "FilmResonance":
        Single_Sim.Handle_Geometry_FilmResonance(sps)


def _prepare_boundaries(sps):
    if sps["ProblemType"] == "Cylinder2D":
        sps["OscBndPars"] = OscBnd.Setup_Boundaries_Cylinder2D(sps)
    elif sps["dimensions"] == 3:
        sps["OscBndPars"] = OscBnd.Setup_Boundaries_3D(sps)
    else:
        raise NotImplementedError(
            f"Boundary setup for {sps['ProblemType']} is not wired into the libraries yet"
        )


def run(config):
    """Prepare one configuration and invoke SingleSimulation exactly once."""
    sps = initialize_sps(config)
    if sps["dimensions"] != 3 and sps["ProblemType"] != "Cylinder2D":
        raise NotImplementedError(
            f"The simulation path for {sps['ProblemType']} is not fully wired yet"
        )
    _prepare_geometry(sps)
    _prepare_boundaries(sps)
    sps["delta"] = sps["delta0_nm"] / sps["Dx_nm"] / sps["n"] ** 0.5
    sps["om"] = 2.0 * (1.0 / 6.0) / sps["delta"] ** 2
    IO.Set_fname(sps)
    Single_Sim.SingleSimulation(sps)
    return sps


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run FreqD-LBM from a JSON configuration")
    parser.add_argument("config", type=Path, help="path to the JSON configuration file")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        run(config)
    except (OSError, json.JSONDecodeError, ValueError, NotImplementedError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
