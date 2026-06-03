"""
HS65: Classic hard problem (3 vars, 1 ineq)
  min  (x1 - x2)^2 + (x1 + x2 - 10)^2/9 + (x3 - 5)^2
  s.t. 48 - x1^2 - x2^2 - x3^2 >= 0
       -4.5 <= x1 <= 4.5
       -4.5 <= x2 <= 4.5
       -5.0 <= x3 <= 5.0
  x0 = (-5, 5, 0), f* = 0.9535288567
"""

import amigo as am
import argparse


class HS65(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("x1", value=-4.0, lower=-4.5, upper=4.5)
        self.add_input("x2", value=4.0, lower=-4.5, upper=4.5)
        self.add_input("x3", value=0.0, lower=-5.0, upper=5.0)
        self.add_objective("obj")
        self.add_constraint("c1", lower=0.0, upper=float("inf"))

    def compute(self):
        x1 = self.inputs["x1"]
        x2 = self.inputs["x2"]
        x3 = self.inputs["x3"]
        self.objective["obj"] = (x1 - x2) ** 2 + (x1 + x2 - 10) ** 2 / 9 + (x3 - 5) ** 2
        self.constraints["c1"] = 48 - x1**2 - x2**2 - x3**2


parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true", default=False)
args = parser.parse_args()

model = am.Model("hs65")
model.add_component("hs65", 1, HS65())
if args.build:
    model.build_module()
model.initialize()

opt = am.Optimizer(model)
opt.optimize(
    {
        "max_iterations": 100,
        "filter_line_search": True,
        "convergence_tolerance": 1e-8,
        "max_line_search_iterations": 30,
    }
)
# f* = 0.9535288567
