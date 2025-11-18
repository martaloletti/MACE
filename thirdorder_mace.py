# Marta Loletti, Noviembre 2025 
# Institut de Ciència de Materials de Barcelona (ICMAB-CSIC)
# E-mail: mloletti@icmab.es


# It computes the third-order anharmonic forces (IFC3s) reading displaced structure generated from 
# thirdorder.py (part of the ShengBTE package: https://www.shengbte.org/home) and using MACE.
# After calculating the forces, it generates a "fake" vasprun.xml file for all the distorted structure. The vasprun.xml file that is generated is in a 
# reduced form with the minimum information readable by thirdorder.py for post-analysis.

# In order to run the script we need to provide the following input files: 3RD.POSCAR.* (displacement configurations from running thirdorder.py)
# We need to provide the following variables: device, model, prefix of structures' files (optional)
#       device -> cpu or cuda for mace calculations
#       model -> MACE model to use


import numpy as np
import os
import subprocess
import glob
import warnings
import sys
import shutil 

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mace.calculators import mace_mp
import torch

torch.serialization.add_safe_globals([slice])
warnings.simplefilter('ignore')


#initialize MACE model
model = 'mace-mpa-0-medium.model'
device = 'cpu'
prefix = "3RD"

#Counts number of distorted structures
files = sorted(glob.glob('3RD.POSCAR.*'))
num_third_distorted_struc = len(files)
print("Found {0} displacement files:".format(num_third_distorted_struc))

width = len(str(num_third_distorted_struc))
ase_adaptor = AseAtomsAdaptor()

for dist in range(num_third_distorted_struc):
    poscar_filename = f"{prefix}.POSCAR.{dist+1:0{width}d}"
    disp_folder = f"disp-{dist+1:0{width}d}"
    vasprun_filename = os.path.join(disp_folder, "vasprun.xml")
    disp_poscar = os.path.join(disp_folder, "POSCAR")

    #Makes folder with displacement
    os.makedirs(disp_folder, exist_ok=True)
    
    #Copy POSCARs in folders 
    shutil.copy(poscar_filename, disp_poscar)
    
    #Reads the structure 
    structure = Structure.from_file(poscar_filename)
    atoms = ase_adaptor.get_atoms(structure)

    #Get forces with MACE
    atoms.calc = mace_mp(model=model, device=device)
    forces = atoms.get_forces()

    # Compute the forces with MACE for all the distorted structures
    with open (vasprun_filename, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<modeling>\n')
        f.write('  <calculation>\n')
        f.write('    <structure>\n')
        f.write('      <varray name="positions">\n')
        for site in structure.sites:
            x, y, z = site.frac_coords
            f.write(f'        <v>{x:15.8f}{y:15.8f}{z:15.8f}</v>\n')
        f.write('      </varray>\n')
        f.write('    </structure>\n')
        f.write('    <varray name="forces">\n')
        for fx, fy, fz in forces:
            f.write(f'      <v>{fx:15.8f}{fy:15.8f}{fz:15.8f}</v>\n')
        f.write('    </varray>\n')
        f.write('  </calculation>\n')
        f.write('</modeling>\n')

print("All calculations completed")
