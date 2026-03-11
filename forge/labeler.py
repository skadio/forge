from typing import Any, Dict, Optional, Tuple
from multiprocessing import get_context
from functools import partial
from tqdm import tqdm
import os
import gurobipy as gp

from forge.processor import MIPProcessor, _MIPUtils, _SATUtils
from forge.utils import save_pickle


class GapInfo:
    def __init__(self, lp_obj: float, lp_sol: Any, mip_obj: float, mip_sol: Any, gap_ratio: float):
        self.lp_obj = lp_obj
        self.lp_sol = lp_sol
        self.mip_obj = mip_obj
        self.mip_sol = mip_sol
        self.gap_ratio = gap_ratio


class SATSatisfiabilityInfo:
    """Container for SAT instance satisfiability labels."""
    def __init__(self, is_satisfiable: bool, solving_time: float):
        """
        Parameters
        ----------
        is_satisfiable : bool
            Whether the SAT instance is satisfiable (True) or unsatisfiable (False).
        solving_time : float
            Time taken to solve the instance (in seconds).
        """
        self.is_satisfiable = is_satisfiable
        self.solving_time = solving_time


class MIPLabeler:

    def __init__(self):
        pass

    @staticmethod
    def convert_mip_to_gapinfo(input_mip_folder,
                               input_mip_instances_file: Optional[str],
                               output_mip_to_gapinfo_pkl,
                               gapinfo_time_limit: int = 120,
                               gurobi_num_threads: int = 1,
                               num_parallel_workers: int = 1,
                               has_return=False) -> Dict[str, GapInfo]:

        # Normalize num_parallel_workers
        if num_parallel_workers is None or num_parallel_workers < 1:
            num_parallel_workers = 1

        mip_files = _MIPUtils.get_only_mip_files(input_mip_folder, input_mip_instances_file, is_sort_by_size=False)

        mip_to_gapinfo: Dict[str, GapInfo] = {}

        # Sequential path when using a single worker
        if num_parallel_workers == 1:
            # Start Gurobi environment and set limits
            gurobi_env = _MIPUtils.start_gurobi_env()
            gurobi_env.setParam("TimeLimit", gapinfo_time_limit)
            gurobi_env.setParam("Threads", gurobi_num_threads)

            for mip_file in tqdm(mip_files):
                mip_to_gapinfo[mip_file] = MIPLabeler._mip_file_to_gapinfo(mip_file, gurobi_env)

            gurobi_env.close()
        else:
            # Parallel path using multiprocessing with the requested number of workers
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_parallel_workers) as pool:
                worker = partial(MIPLabeler._run_gapinfo_worker,
                                 gapinfo_time_limit=gapinfo_time_limit,
                                 gurobi_num_threads=gurobi_num_threads)

                for result in tqdm(pool.imap_unordered(worker, mip_files), total=len(mip_files)):
                    if result:
                        mip_file, gapinfo = result
                        mip_to_gapinfo[mip_file] = gapinfo

        save_pickle(mip_to_gapinfo, output_mip_to_gapinfo_pkl)

        return mip_to_gapinfo if has_return else None


    @staticmethod
    def _mip_file_to_gapinfo(mip_file: str, gurobi_env) -> Optional[GapInfo]:
        """
            On success, returns GapInfo object.
            On fail, returns None.
        """

        # Create mip model
        mip_model: gp.Model = gp.read(mip_file, env=gurobi_env)

        # Solve LP relaxation
        lp_model = mip_model.copy().relax()
        lp_model.optimize()

        # Solve MIP within time limit
        mip_model.optimize()

        # Skip instances that are infeasible or unbounded
        if mip_model.status == gp.GRB.status.INF_OR_UNBD or lp_model.status == gp.GRB.status.INF_OR_UNBD:
            print(f"\rSkipped (INF_OR_UNBD) | MIP status={mip_model.status}, "
                  f"LP status={lp_model.status} | {mip_file}", end="")
            gurobi_env.close()
            return None

        # Skip instances without a solution
        if mip_model.SolCount < 1 or lp_model.SolCount < 1:
            print(f"\rSkipped (No Solution) | MIP status={mip_model.status}, "
                  f"LP status={lp_model.status} | {mip_file}", end="")
            gurobi_env.close()
            return None

        # Retrieve lp and mip objective values and solutions
        lp_obj = lp_model.objVal
        lp_sol = [v.x for v in lp_model.getVars()]
        mip_obj = mip_model.objVal
        mip_sol = [v.x for v in mip_model.getVars()]

        # Calculate ratio (handle zero division)
        # For minimization, LP ≤ MIP, so ratio = lp_obj / mip_obj.
        # For maximization, LP ≥ MIP, so ratio = mip_obj / lp_obj.
        if mip_model.ModelSense == gp.GRB.MINIMIZE:
            ratio = 1.0 if mip_obj == 0 else lp_obj / mip_obj
        else:  # maximization
            ratio = 1.0 if lp_obj == 0 else mip_obj / lp_obj

        print("\rRatio:", ratio, "|", mip_file)
        return GapInfo(lp_obj=lp_obj, lp_sol=lp_sol, mip_obj=mip_obj, mip_sol=mip_sol, gap_ratio=ratio)

    @staticmethod
    def _run_gapinfo_worker(mip_file: str,
                            gapinfo_time_limit: int,
                            gurobi_num_threads: int) -> Optional[Tuple[str, GapInfo]]:

        """Worker function to compute GapInfo for a single MIP instance.

        This function is designed to be picklable so it can be used with multiprocessing.
        It creates and tears down its own Gurobi environment inside each worker process.

        On success, returns GapInfo object.
        On fail, returns None.
        """

        try:
            # Start Gurobi environment and set limits for this worker
            gurobi_env = _MIPUtils.start_gurobi_env()
            gurobi_env.setParam("TimeLimit", gapinfo_time_limit)
            gurobi_env.setParam("Threads", gurobi_num_threads)

            # Create gapinfo
            gapinfo = MIPLabeler._mip_file_to_gapinfo(mip_file, gurobi_env)

            gurobi_env.close()

            # Return mip_file also so that we know which gapinfo is returned
            return mip_file, gapinfo

        except Exception as exc:
            print(f"\nError while processing {mip_file}: {exc}")
            try:
                # Best-effort cleanup; in some failure modes env may not exist
                gurobi_env.close()
            except Exception:
                pass
            return None


class SATLabeler:
    """Labeler for SAT satisfiability prediction tasks.
    
    SAT instances should be labeled in their filenames with either "_sat" or "_unsat"
    to indicate satisfiability.
    """

    def __init__(self):
        pass

    @staticmethod
    def convert_sat_to_satisfiability_info(input_sat_folder: str,
                                          input_sat_instances_file: Optional[str],
                                          output_sat_to_satinfo_pkl: str,
                                          has_return: bool = False) -> Optional[Dict[str, SATSatisfiabilityInfo]]:
        """Extract satisfiability labels from SAT instance filenames.

        SAT instances should have "_sat" or "_unsat" in their filename to indicate satisfiability.

        Parameters
        ----------
        input_sat_folder : str
            Path to folder containing SAT files (in LP/MPS format).
        input_sat_instances_file : Optional[str]
            Optional file containing list of SAT instances to process.
        output_sat_to_satinfo_pkl : str
            Path to save the output pickle file.
        has_return : bool, default=False
            Whether to return the dictionary.

        Returns
        -------
        Optional[Dict[str, SATSatisfiabilityInfo]]
            Mapping from SAT instance file path to satisfiability info if has_return=True.
        """
        sat_files = _SATUtils.get_only_sat_files(input_sat_folder, input_sat_instances_file, is_sort_by_size=False)

        sat_to_satinfo: Dict[str, SATSatisfiabilityInfo] = {}

        for sat_file in tqdm(sat_files):
            satinfo = SATLabeler._sat_filename_to_satisfiability_info(sat_file)
            if satinfo is not None:
                sat_to_satinfo[sat_file] = satinfo

        save_pickle(sat_to_satinfo, output_sat_to_satinfo_pkl)

        return sat_to_satinfo if has_return else None

    @staticmethod
    def _sat_filename_to_satisfiability_info(sat_file: str) -> Optional[SATSatisfiabilityInfo]:
        """Extract satisfiability label from SAT instance filename.

        Looks for "_sat" or "_unsat" in the filename to determine satisfiability.

        Parameters
        ----------
        sat_file : str
            Path to the SAT file.

        Returns
        -------
        Optional[SATSatisfiabilityInfo]
            Satisfiability info on success, None if label cannot be determined from filename.
        """
        try:
            filename = os.path.basename(sat_file).lower()
            
            # Check for satisfiability label in filename
            if "_unsat" in filename or "unsat" in filename:
                is_satisfiable = False
                label = "UNSAT"
            elif "_sat" in filename or "sat" in filename:
                is_satisfiable = True
                label = "SAT"
            else:
                print(f"\rSkipped (No label in filename) | {sat_file}", end="")
                return None

            print(f"\r{label} | {sat_file}", end="")
            return SATSatisfiabilityInfo(is_satisfiable=is_satisfiable, solving_time=0.0)

        except Exception as e:
            print(f"\rSkipped (Error) | {sat_file} | {e}", end="")
            return None