from typing import Any, Dict

import gurobipy as gp

from forge.processor import MIPProcessor
from forge.utils import save_pickle


class GapInfo:
    def __init__(self, ratio: float, mip_sol: Any, mip_obj: float, lp_obj: float):
        self.ratio = ratio
        self.mip_sol = mip_sol
        self.mip_obj = mip_obj
        self.lp_obj = lp_obj

class MIPLabeler:

    def __init__(self):
        pass

    @staticmethod
    def get_mip_to_integral_gap(input_mip_folder,
                                output_mip_to_gapinfo_pkl,
                                gapinfo_time_limit: int = 120,
                                has_return=False) -> Dict[str, GapInfo]:

        mip_files = MIPProcessor.get_only_mip_files(input_mip_folder, is_sort_by_size=False)

        # Start Gurobi environment
        gurobi_env = MIPProcessor._start_gurobi_env()

        mip_to_gapinfo = {}
        for idx, mip_file in enumerate(mip_files):

            # Set time limit
            gurobi_env.setParam("TimeLimit", gapinfo_time_limit)

            # Create mip model
            mip_model = gp.read(mip_file, env=gurobi_env)

            # Solve LP relaxation
            lp_model = mip_model.copy().relax()
            lp_model.optimize()

            # Solve MIP within time limit
            mip_model.optimize()

            # Skip instances that are infeasible or unbounded
            if mip_model.status == gp.GRB.status.INF_OR_UNBD or lp_model.status == gp.GRB.status.INF_OR_UNBD:
                print(f"\rInstance : {idx} | Skipped (status: MIP={mip_model.status}, LP={lp_model.status})", end='')
                continue

            # Calculate ratio (handle zero division)
            # TODO does this assume minimization?
            min_val = min(mip_model.objVal, lp_model.objVal)
            max_val = max(mip_model.objVal, lp_model.objVal)
            ratio = 1.0 if max_val == 0 else min_val / max_val
            mip_obj = mip_model.objVal
            lp_obj = lp_model.objVal
            mip_sol = mip_model.Xn
            print("\rInstance : ", idx, "| Ratio : ", ratio, end='')

            # Store gap information
            mip_to_gapinfo[mip_file] = GapInfo(ratio=ratio, mip_sol=mip_sol, mip_obj=mip_obj, lp_obj=lp_obj)

        # TODO in original code, this is indented incorrectly? it was inside the for-loop above
        save_pickle(mip_to_gapinfo, output_mip_to_gapinfo_pkl)

        return mip_to_gapinfo if has_return else None
