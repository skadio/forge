from typing import Any, Dict
from tqdm import tqdm
import gurobipy as gp

from forge.processor import MIPProcessor
from forge.utils import save_pickle


class GapInfo:
    def __init__(self, lp_obj: float, lp_sol:Any, mip_obj: float, mip_sol: Any, gap_ratio: float):
        self.lp_obj = lp_obj
        self.lp_sol = lp_sol
        self.mip_obj = mip_obj
        self.mip_sol = mip_sol
        self.gap_ratio = gap_ratio


class MIPLabeler:

    def __init__(self):
        pass

    # TODO: See if I can parallelize this over multiple processes
    @staticmethod
    def get_mip_to_gapinfo(input_mip_folder,
                           output_mip_to_gapinfo_pkl,
                           gapinfo_time_limit: int = 120,
                           gurobi_num_threads: int = 1,
                           has_return=False) -> Dict[str, GapInfo]:

        mip_files = MIPProcessor.get_only_mip_files(input_mip_folder, is_sort_by_size=False)
        print ('''Starting MIP to GapInfo conversion for {} instances...'''.format(len(mip_files)))

        # Start Gurobi environment
        gurobi_env = MIPProcessor._start_gurobi_env()

        # Set time limit
        gurobi_env.setParam("TimeLimit", gapinfo_time_limit)
        gurobi_env.setParam("Threads", gurobi_num_threads)

        mip_to_gapinfo = {}
        for idx, mip_file in tqdm(enumerate(mip_files)):

            # Create mip model
            mip_model: gp.Model = gp.read(mip_file, env=gurobi_env)

            # Solve LP relaxation
            lp_model = mip_model.copy().relax()
            lp_model.optimize()

            # Solve MIP within time limit
            mip_model.optimize()

            # Skip instances that are infeasible or unbounded
            if mip_model.status == gp.GRB.status.INF_OR_UNBD or lp_model.status == gp.GRB.status.INF_OR_UNBD:
                print(f"\rInstance : {idx} | Skipped (status: MIP={mip_model.status}, LP={lp_model.status})", end='')
                continue

            # Skip instances without a solution
            if mip_model.SolCount < 1 or lp_model.SolCount < 1:
                print(f"\rInstance : {idx} | Skipped (status: MIP={mip_model.status}, LP={lp_model.status})", end='')
                continue

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

            print("\rInstance : ", idx, "| Ratio : ", ratio, end='')

            # Store gap information
            mip_to_gapinfo[mip_file] = GapInfo(lp_obj=lp_obj, lp_sol=lp_sol,
                                               mip_obj=mip_obj, mip_sol=mip_sol,
                                               gap_ratio=ratio)

        save_pickle(mip_to_gapinfo, output_mip_to_gapinfo_pkl)

        gurobi_env.close()

        return mip_to_gapinfo if has_return else None
