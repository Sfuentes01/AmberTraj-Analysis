# General
import os
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import gc
import logging
# Data Handling
import pandas as pd
import numpy as np
# Analysis
import MDAnalysis as mda
from MDAnalysis.analysis import dihedrals
from MDAnalysis import transformations      
from MDAnalysis.analysis import align      
from MDAnalysis.analysis import rms         
from MDAnalysis.analysis import dssp      
import pyKVFinder                        
import prolif as plf                     
import freesasa                             
from mdakit_sasa.analysis.sasaanalysis import SASAAnalysis
from tqdm.auto import tqdm   
freesasa.setVerbosity(freesasa.nowarnings)

# Chi-1 CG atom varies by residue type
CHI1_CG_MAP = {
    'VAL': 'CG1',
    'ILE': 'CG1',
    'THR': 'OG1',
    'SER': 'OG',
    'CYS': 'SG',
}
CHI1_DEFAULT_CG = 'CG'  # Works for ARG, TRP, TYR, PHE, HIS, LYS, etc.

def run_rmsd(u, protein_sel, ligand_sel, system_name, top, traj_list, out_dir):
    logging.info(f"Calculating RMSD for {system_name}...")

    # 1. Create a static reference
    ref = mda.Universe(top, traj_list[0])
    
    # 2. Define the structural masks
    prot_mask = f"{protein_sel} and name CA"
    group_sels = []
    
    if ligand_sel:
        lig_mask = f"{ligand_sel} and not name H*"
        group_sels.append(lig_mask)

    # 3. Pass the concatenated universe into the active calculation
    R = rms.RMSD(u, ref, select=prot_mask, groupselections=group_sels)
    R.run()

    # 4. Extract the data matrix dynamically into a clean DataFrame
    data_dict = {
        'Frame': R.results.rmsd[:, 0],
        'Time_ps': R.results.rmsd[:, 1], 
        'Time_ns': R.results.rmsd[:, 1] / 1000,
        'Protein_RMSD': R.results.rmsd[:, 2]
    }

    if ligand_sel:
        data_dict['Ligand_RMSD'] = R.results.rmsd[:, 3] 

    # 5. Saving
    df_rmsd_all = pd.DataFrame(data_dict)
    df_rmsd_all.to_csv(out_dir / f'{system_name}_RMSD.csv', index=False)

def run_rmsf(u, protein_sel, ligand_sel, system_name, top, traj_list, out_dir):
    logging.info(f"Calculating RMSF for {system_name}...")

    ref = mda.Universe(top, traj_list[0])
    
    # 1. Define masks and align
    align_mask = f'{protein_sel} and name CA'
    align.AlignTraj(u, ref, select=align_mask, in_memory=True).run()

    calc_masks = {
        'Receptor': f'{protein_sel} and name CA'
    }
    if ligand_sel:
        calc_masks['Ligand'] = f'{ligand_sel} and not name H*'

    rmsf_results = {}

    # 2. Calculate RMSF
    for target_name, mask_string in calc_masks.items():
        target_atoms = u.select_atoms(mask_string)
        rmsf_calc = rms.RMSF(target_atoms).run()
        
        df_target = pd.DataFrame({
            'Residue': target_atoms.resids,
            'RMSF_Å': rmsf_calc.rmsf
        })
        df_target.set_index('Residue', inplace=True)
        rmsf_results[target_name] = df_target

    df_rmsf_protein = rmsf_results['Receptor']
    df_rmsf_ligand = rmsf_results.get('Ligand', None) 

    # 3. Save Raw Data
    df_rmsf_protein.to_csv(out_dir / f"{system_name}_Receptor_RMSF_Data.csv")
    if ligand_sel:
        df_rmsf_ligand.to_csv(out_dir / f"{system_name}_Ligand_RMSF_Data.csv")

    # 4. Extract High Mobility Data (Threshold = 1.5 Å)
    threshold = 1.5
    df_high_mob_receptor = df_rmsf_protein[df_rmsf_protein['RMSF_Å'] >= threshold].copy()
    
    if ligand_sel:
        df_high_mob_ligand = df_rmsf_ligand[df_rmsf_ligand['RMSF_Å'] >= threshold].copy()

    # 5. Save High Mobility Data
    if not df_high_mob_receptor.empty:
        df_high_mob_receptor.to_csv(out_dir / f"{system_name}_Receptor_HighMobRes.csv", index=True)

    if ligand_sel:
        if not df_high_mob_ligand.empty:
            df_high_mob_ligand.to_csv(out_dir / f"{system_name}_Ligand_HighMobRes.csv", index=True)

def run_dssp(u, protein_sel, ligand_sel, system_name, out_dir):
    logging.info(f"Calculating Secondary Structure (DSSP) for {system_name}...")

    # 1. Dynamic Selection Configuration
    dssp_masks = {
        'Receptor': f'protein and {protein_sel}' 
    }

    if ligand_sel:
        lig_atoms = u.select_atoms(ligand_sel)
        if len(lig_atoms.select_atoms("name CA")) > 0: # For non-peptide ligands
            dssp_masks['Ligand'] = ligand_sel
        else:
            logging.info(" -> Ligand appears to be a small molecule. Skipping DSSP for ligand.")

    # 2. Execution Loop
    for target_name, mask_string in dssp_masks.items():
        selection = u.select_atoms(mask_string)
        
        if len(selection) == 0:
            logging.info(f" -> Skipping {target_name}: No atoms found for mask '{mask_string}'.")
            continue
            
        dssp_calc = dssp.DSSP(selection).run()
        combined_ss = dssp_calc.results.dssp
        total_frames = combined_ss.shape[0]
        
        # 3. Vectorized Counting
        helix_counts = np.isin(combined_ss, ['H', 'G', 'I']).sum(axis=0)
        sheet_counts = np.isin(combined_ss, ['E', 'B']).sum(axis=0)
        coil_counts = total_frames - (helix_counts + sheet_counts)
        
        # 4. Construct DataFrame (Clean labels from the start)
        df_target = pd.DataFrame({'Residue': selection.residues.resids})
        df_target['Alpha Helix'] = (helix_counts / total_frames) * 100
        df_target['Beta Sheet'] = (sheet_counts / total_frames) * 100
        df_target['Coil'] = (coil_counts / total_frames) * 100
        df_target.set_index('Residue', inplace=True)
        
        # 5. Apply 60% Threshold Classification
        ss_cols = ['Alpha Helix', 'Beta Sheet', 'Coil']
        max_vals = df_target[ss_cols].max(axis=1)
        dominant_ss = df_target[ss_cols].idxmax(axis=1)
        
        df_target['Dominant_SS'] = np.where(max_vals >= 60.0, dominant_ss, 'Mixed')
        df_target['Confidence_%'] = max_vals.round(2)
        
        # 6. Save Final Data
        csv_path = out_dir / f"{system_name}_{target_name}_SS_Data.csv"
        df_target.to_csv(csv_path, index=True)

def run_dihedrals(u, system_name, out_dir, motif_dry, motif_tswitch, motif_npy):
    logging.info(f"Calculating Micro-Switch Dihedrals (Chi-1) for {system_name}...")

    # 1. Build the target dictionary dynamically based on user input
    target_motifs = {}
    if motif_dry is not None:
        target_motifs['Ionic_Lock_R'] = motif_dry
    if motif_tswitch is not None:
        target_motifs['Toggle_Switch_W'] = motif_tswitch
    if motif_npy is not None:
        target_motifs['NPxxY_Y'] = motif_npy

    if not target_motifs:
        logging.info(" -> No motif residues provided. Skipping dihedral analysis.")
        return

    # 2. Initialize the DataFrame with chemical time
    total_frames = len(u.trajectory)
    dt_ps = getattr(u.trajectory, 'dt', 10.0)

    sys_df = pd.DataFrame({
        'Frame': np.arange(total_frames),
        'Time_ns': np.arange(total_frames) * (dt_ps / 1000.0)
    })

    # 3. Calculate Dihedrals
    for motif_name, res_id in target_motifs.items():

        # Identify the residue name to pick the right CG atom
        res_atoms = u.select_atoms(f"resid {res_id}")
        if len(res_atoms) == 0:
            logging.warning(f" -> Warning: Residue {res_id} not found. Skipping {motif_name}.")
            continue

        resname = res_atoms.residues[0].resname.upper()
        cg_atom = CHI1_CG_MAP.get(resname, CHI1_DEFAULT_CG)
        logging.info(f" -> {motif_name} ({resname} {res_id}): Using Chi-1 atoms N-CA-CB-{cg_atom}")

        # Dynamic atom name list
        chi1_atom_names = ['N', 'CA', 'CB', cg_atom]
        atom_groups = [u.select_atoms(f"resid {res_id} and name {atom}") for atom in chi1_atom_names]

        if any(len(ag) == 0 for ag in atom_groups):
            logging.warning(f" -> Warning: Missing atoms for {motif_name} (Residue {res_id}, {resname}). Skipping.")
            continue

        target_dihedral_atoms = atom_groups[0] + atom_groups[1] + atom_groups[2] + atom_groups[3]
        dih_analysis = dihedrals.Dihedral([target_dihedral_atoms]).run()

        sys_df[motif_name] = dih_analysis.results.angles[:, 0]
        logging.info(f" -> {motif_name} (Residue {res_id}): Extracted {total_frames} frames.")

    # 4. Save Data
    csv_path = out_dir / f"{system_name}_Dihedral_Chi1.csv"
    sys_df.to_csv(csv_path, index=False)

def run_sasa(u, system_name, out_dir, sasa_resids):
    logging.info(f"Calculating SASA for {system_name}...")

    # --- CRITICAL SAFETY OVERRIDE ---
    # Mute the C-library's stderr pipe to prevent memory buffer deadlocks
    # caused by missing hydrogens in charged systems.
    # import freesasa
    freesasa.setVerbosity(freesasa.nowarnings)

    res_str = " ".join(map(str, sasa_resids))
    pocket_sel_string = f"protein and resid {res_str}"

    pocket_ag = u.select_atoms(pocket_sel_string)
    
    if len(pocket_ag) == 0:
        logging.warning(" -> WARNING: No atoms found for SASA selection. Skipping.")
        return

    logging.info(f" -> Selected {len(pocket_ag)} atoms across {len(pocket_ag.residues)} residues.")

    sasa_calc = SASAAnalysis(u, select=pocket_sel_string)
    sasa_calc.run(verbose=True)

    total_frames = len(u.trajectory)
    dt_ps = getattr(u.trajectory, 'dt', 10.0)

    sasa_df = pd.DataFrame({
        'Frame': np.arange(total_frames),
        'Time_ns': np.arange(total_frames) * (dt_ps / 1000.0),
        'SASA_Å2': sasa_calc.results.sasa
    })

    csv_path = out_dir / f"{system_name}_Pocket_SASA.csv"
    sasa_df.to_csv(csv_path, index=False)
    logging.info(f" -> SASA data successfully saved.")

def run_volume(u, system_name, top, traj_list, out_dir, vol_resids):
    logging.info(f"Calculating Cavity Volume (pyKVFinder) for {system_name}...")

    # 1. On-the-fly Alignment Reference
    ref = mda.Universe(top, traj_list[0])
    ref.trajectory[0]
    backbone_sel = "protein and backbone"
    
    align_transform = transformations.fit_rot_trans(u.select_atoms(backbone_sel), ref.select_atoms(backbone_sel))
    u.trajectory.add_transformations(align_transform)

    # 2. Build the Master Array
    protein = u.select_atoms("protein")
    vdw_dict = {'C': 1.70, 'O': 1.52, 'N': 1.55, 'S': 1.80, 'H': 1.20, 'P': 1.80}
    radii = np.array([vdw_dict.get(atom.element.upper(), 1.50) for atom in protein])

    try:
        chains = protein.chainIDs
    except AttributeError:
        try:
            chains = protein.segids
        except AttributeError:
            chains = np.array(['A'] * len(protein))

    atomic_data = np.empty((len(protein), 8), dtype=object)
    atomic_data[:, 0] = protein.names
    atomic_data[:, 1] = protein.resnames
    atomic_data[:, 2] = chains
    atomic_data[:, 3] = protein.resids
    atomic_data[:, 7] = radii

    # 3. Define the Bounding Box
    res_str = " ".join(map(str, vol_resids))
    pocket_atoms = u.select_atoms(f"protein and resid {res_str}")
    
    if len(pocket_atoms) == 0:
        logging.warning(" -> WARNING: No atoms found for Volume selection. Skipping.")
        return

    u.trajectory[0]  # Ensure we are at frame 0 to measure the box
    box_buffer = 5.0
    p_min = pocket_atoms.positions.min(axis=0) - box_buffer
    p_max = pocket_atoms.positions.max(axis=0) + box_buffer

    box_vertices = np.array([
        [p_min[0], p_min[1], p_min[2]],  # Origin
        [p_max[0], p_min[1], p_min[2]],  # X-Limit
        [p_min[0], p_max[1], p_min[2]],  # Y-Limit
        [p_min[0], p_min[1], p_max[2]]   # Z-Limit
    ])

    # 4. Execute pyKVFinder
    volumes = []
    step_size = 0.6
    voxel_volume = step_size ** 3  # 0.216 Å³

    for ts in tqdm(u.trajectory, desc=f"Volume {system_name}"):
        # Update ONLY the coordinates for this frame
        atomic_data[:, 4:7] = protein.positions
        
        results = pyKVFinder.detect(
            atomic=atomic_data,
            vertices=box_vertices,
            step=step_size,
            probe_in=1.4,
            probe_out=4.0
        )

        num_cavities = results[0]
        grid = results[1]

        if num_cavities > 0:
            cavity_volumes = [np.sum(grid == cav_id) * voxel_volume for cav_id in range(1, num_cavities + 1)]
            volumes.append(max(cavity_volumes))
        else:
            volumes.append(0.0)

    # 5. Format Time & Export
    total_frames = len(volumes)
    dt_ps = getattr(u.trajectory, 'dt', 10.0)
    
    df_vol = pd.DataFrame({
        'Frame': range(total_frames),
        'Time_ns': [i * (dt_ps / 1000.0) for i in range(total_frames)],
        'Volume_Å3': volumes
    })

    csv_path = out_dir / f"{system_name}_Cavity_Volume.csv"
    df_vol.to_csv(csv_path, index=False)
    logging.info(f" -> Cavity volume data successfully saved.")

def main():

    parser = argparse.ArgumentParser(description="MD Analysis Pipeline for GPCRs")
    parser.add_argument("--system", type=str, required=True, help="System name identifier (e.g., 'AT1R')")
    parser.add_argument("--RMSD", action="store_true", help="Run RMSD analysis")
    parser.add_argument("--RMSF", action="store_true", help="Run RMSF Analysis")
    parser.add_argument("--DSSP", action="store_true", help="Run Secondary structure content analysis")
    parser.add_argument("--length", type=int, required=True, help="Receptor length")
    parser.add_argument("--ligand", type=str, help="Input ligand 'resname LIG and resid NUM'")
    parser.add_argument("--dihedrals", action="store_true", help="Run Motif Dihedral analysis")
    parser.add_argument("--motif-all", type=int, nargs=3, metavar=('DRY', 'TSWITCH', 'NPY'), 
                        help="Provide all 3 motif resids separated by spaces (e.g., --motif-all 126 256 302)")
    parser.add_argument("--motif-dry", type=int, help="Residue ID for DRY motif (TM3)")
    parser.add_argument("--motif-tswitch", type=int, help="Residue ID for Toggle Switch (TM6)")
    parser.add_argument("--motif-npy", type=int, help="Residue ID for NPxxY motif (TM7)")
    parser.add_argument("--sasa", type=int, nargs='+', help="List of residue IDs for pocket SASA (e.g., --sasa 31 34 36 96)")
    parser.add_argument("--volume", nargs='*', type=int, help="Run pyKVFinder cavity volume. Accepts resid list. If flag used without numbers, falls back to --sasa residues.")
    parser.add_argument("--system-complex", type=str, help="Run analysis only on a specific complex/subfolder (e.g., 'A18')")

    args = parser.parse_args()

# --ALL function by default if no arguments added
    if not (args.RMSD or args.RMSF or args.DSSP or args.dihedrals or args.sasa or args.volume):
        args.RMSD = True
        args.RMSF = True
        args.DSSP = True
    motif_dry_res = args.motif_dry
    motif_tswitch_res = args.motif_tswitch
    motif_npy_res = args.motif_npy

    if args.motif_all:
        motif_dry_res = args.motif_all[0]
        motif_tswitch_res = args.motif_all[1]
        motif_npy_res = args.motif_all[2]

# Pathing
    base_path = Path(args.system)
    general_system = Path(base_path).name

    # --- Initialize Global Logging ---
    log_file = base_path / "analysis.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This keeps output logging.infoing to the terminal too
        ]
    )
    logging.info(f"=== Starting GPCR MD Analysis Pipeline for {general_system} ===")
    # ---------------------------------

    ligand_list = []
    for ligand_dir in base_path.iterdir():
        if not ligand_dir.is_dir():
            continue
        if args.system_complex and ligand_dir.name != args.system_complex:
            continue
        topology_search = list(ligand_dir.glob("*.prmtop"))
        if not topology_search:
            continue

        ligand_name = ligand_dir.name
        topology_path = topology_search[0]
        trajectory_files = sorted(list(ligand_dir.rglob("*.nc")))


        output_path = (base_path / "DATA" / ligand_name)
        output_path.mkdir(parents= True, exist_ok=True)

        ligand_list.append(ligand_name)

        logging.info("Initializing Virtual Concatenation...")

        # The * operator unpacks the Python list
        u_concat = mda.Universe(topology_path, *trajectory_files)
        total_frames = len(u_concat.trajectory)

        replicate_boundaries_frames = []
        replicate_boundaries_time_ns = []
        current_frame_count = 0
        current_time_ps = 0.0

        # Extract and logging.info the time step from the first file's metadata
        temp_u_first = mda.Universe(topology_path, trajectory_files[0])
        detected_dt_ps = temp_u_first.trajectory.dt
        logging.info(f"Detected Time Step (dt) from metadata: {detected_dt_ps} ps\n")
        del temp_u_first

        for traj in trajectory_files:
            # Temporarily load just the single trajectory
            temp_u = mda.Universe(topology_path, traj)
            
            frames = len(temp_u.trajectory)
            dt_ps = temp_u.trajectory.dt 
            
            current_frame_count += frames
            current_time_ps += (frames * dt_ps)
            
            replicate_boundaries_frames.append(current_frame_count)
            replicate_boundaries_time_ns.append(float(current_time_ps / 1000.0))
            
            del temp_u

        total_time_ns = current_time_ps / 1000.0

        logging.info(f'Processing {ligand_name} system.')
        logging.info(f"Linked {len(trajectory_files)} replicates.")
        logging.info(f"Total Frames: {total_frames}")
        logging.info(f"Total Simulated Time: {total_time_ns:.2f} ns")
        logging.info(f"Replicate Boundaries (Frames): {replicate_boundaries_frames[:-1]}")
        logging.info(f"Replicate Boundaries (Time in ns): {replicate_boundaries_time_ns[:-1]}")
        
        traj_metadata = output_path / f'{ligand_name}_metadata.txt'

        with open(traj_metadata, "w") as f:
            f.write(f'Linked {len(trajectory_files)} replicates.\n')
            f.write(f'Total Concatenated Frames: {total_frames}\n')
            f.write(f'Total Simulated Time: {total_time_ns:.2f} ns\n')
            f.write(f'Replicate Boundaries (Frames): {replicate_boundaries_frames[:-1]}\n')
            f.write(f'Replicate Boundaries (Time in ns): {replicate_boundaries_time_ns[:-1]}\n')

# Receptor and ligand ASL lines

        all_protein_like = u_concat.select_atoms("protein")
        all_residues = all_protein_like.residues
        
        protein_sel_str = ""
        ligand_sel_str = None
        has_ligand = False

        if len(all_residues) > args.length:
            has_ligand = True
            receptor_residues = all_residues[:args.length]
            resid_list = " ".join(map(str, receptor_residues.resids))
            protein_sel_str = f"resid {resid_list}"
            
            # Use custom ligand string if provided, otherwise deduce mathematically
            if args.ligand:
                ligand_sel_str = args.ligand
            else:
                ligand_residues = all_residues[args.length:]
                ligand_sel_str = f"resid {ligand_residues.resids[0]}-{ligand_residues.resids[-1]}"
                
        else:
            protein_sel_str = f"resid {all_residues.resids[0]}-{all_residues.resids[-1]}"
            
            # Only if sequence length implies APO, but user provided ligand
            if args.ligand:
                has_ligand = True
                ligand_sel_str = args.ligand
            else:
                has_ligand = False
# Execution                
        try:
            if args.RMSD:
                run_rmsd(u_concat, protein_sel_str, ligand_sel_str, ligand_name, topology_path, trajectory_files, output_path)

            if args.RMSF:
                run_rmsf(u_concat, protein_sel_str, ligand_sel_str, ligand_name, topology_path, trajectory_files, output_path)
                
            if args.DSSP:
                run_dssp(u_concat, protein_sel_str, ligand_sel_str, ligand_name, output_path)
            
            if args.dihedrals:
                run_dihedrals(u_concat, ligand_name, output_path, motif_dry_res, motif_tswitch_res, motif_npy_res)
            if args.sasa:
                run_sasa(u_concat, ligand_name, output_path, args.sasa)

            if args.volume is not None:
                # If args.volume is an empty list [], fall back to args.sasa
                vol_resids = args.volume if len(args.volume) > 0 else args.sasa
                
                if not vol_resids:
                    logging.error(f"[{ligand_name}] Volume analysis skipped: No residues provided and --sasa was not used as a fallback.")
                else:
                    run_volume(u_concat, ligand_name, topology_path, trajectory_files, output_path, vol_resids)

            logging.info(f"[{ligand_name}] Analysis completed successfully.")

        except Exception as e:
            # If a system fails, log the exact error and continue the loop
            logging.error(f"[{ligand_name}] FAILED during analysis: {str(e)}", exc_info=True)

        finally:
            # Explicit Memory Management (Prevents RAM Stacking)
            if 'u_concat' in locals():
                del u_concat
            gc.collect()
            logging.info(f"[{ligand_name}] Cleared trajectory from memory.\n")


if __name__ == "__main__":
    main()
