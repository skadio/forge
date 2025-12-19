from typing import Any, Dict, Optional, Tuple
from multiprocessing import get_context
from functools import partial
from tqdm import tqdm
import gurobipy as gp
import os

from forge.processor import MIPProcessor
from forge.utils import save_pickle


class GapInfo:
    def __init__(self, lp_obj: float, lp_sol: Any, mip_obj: float, mip_sol: Any, gap_ratio: float):
        self.lp_obj = lp_obj
        self.lp_sol = lp_sol
        self.mip_obj = mip_obj
        self.mip_sol = mip_sol
        self.gap_ratio = gap_ratio


class MIPLabeler:

    def __init__(self):
        pass

    @staticmethod
    def get_mip_to_gapinfo(input_mip_folder,
                           input_mip_instances_file: Optional[str],
                           output_mip_to_gapinfo_pkl,
                           gapinfo_time_limit: int = 120,
                           gurobi_num_threads: int = 1,
                           num_parallel_workers: int = 1,
                           has_return=False) -> Dict[str, GapInfo]:

        # Normalize num_parallel_workers
        if num_parallel_workers is None or num_parallel_workers < 1:
            num_parallel_workers = 1

        mip_files = MIPProcessor.get_only_mip_files(input_mip_folder, input_mip_instances_file, is_sort_by_size=False)

        mip_to_gapinfo: Dict[str, GapInfo] = {}

        # Sequential path (original behavior) when using a single worker
        if num_parallel_workers == 1:
            # Start Gurobi environment and set limits
            gurobi_env = MIPProcessor._start_gurobi_env()
            gurobi_env.setParam("TimeLimit", gapinfo_time_limit)
            gurobi_env.setParam("Threads", gurobi_num_threads)

            for mip_file in tqdm(mip_files):
                mip_to_gapinfo[mip_file] = MIPLabeler._get_gapinfo_from_mip_file(mip_file, gurobi_env)

            gurobi_env.close()
        else:
            # Parallel path using multiprocessing with the requested number of workers
            ctx = get_context("spawn")
            with ctx.Pool(processes=num_parallel_workers) as pool:
                worker = partial(MIPLabeler._run_get_gapinfo,
                                 gapinfo_time_limit=gapinfo_time_limit,
                                 gurobi_num_threads=gurobi_num_threads)

                for result in tqdm(pool.imap_unordered(worker, mip_files), total=len(mip_files)):
                    if result is None:
                        continue
                    mip_file, gapinfo = result
                    mip_to_gapinfo[mip_file] = gapinfo

        save_pickle(mip_to_gapinfo, output_mip_to_gapinfo_pkl)

        return mip_to_gapinfo if has_return else None


    @staticmethod
    def _get_gapinfo_from_mip_file(mip_file: str, gurobi_env) -> Optional[GapInfo]:
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
    def _run_get_gapinfo(mip_file: str,
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
            gurobi_env = MIPProcessor._start_gurobi_env()
            gurobi_env.setParam("TimeLimit", gapinfo_time_limit)
            gurobi_env.setParam("Threads", gurobi_num_threads)

            # Create model and get gapinfo
            gapinfo = MIPLabeler._get_gapinfo_from_mip_file(mip_file, gurobi_env)

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