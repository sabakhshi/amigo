import amigo as am
import argparse


class Disp1(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("x")
        self.add_input("z", shape=2, value=[1, 1])
        self.add_input("y", shape=2, value=[1, 1])

        self.add_constraint("c1")

    def compute(self):
        x = self.inputs["x"]
        z = self.inputs["z"]
        y = self.inputs["y"]

        self.constraints["c1"] = z[0] ** 2 + z[1] + x - 0.2 * y[1] - y[0]


class Disp2(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("z", shape=2)
        self.add_input("y", shape=2)

        self.add_constraint("c2")

    def compute(self):
        z = self.inputs["z"]
        y = self.inputs["y"]

        self.constraints["c2"] = am.sqrt(y[0]) + z[0] + z[1] - y[1]


class Objective(am.Component):
    def __init__(self):
        super().__init__()
        self.add_objective("obj")

        self.add_input("x")
        self.add_input("z", shape=2)
        self.add_input("y", shape=2)

    def compute(self):
        x = self.inputs["x"]
        z = self.inputs["z"]
        y = self.inputs["y"]

        self.objective["obj"] = x**2 + z[1] + y[0] + am.exp(-y[1])


class Con1(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("y", shape=2)
        self.add_constraint("g1")

    def compute(self):
        self.constraints["g1"] = 3.16 - self.inputs["y"][0]


class Con2(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("y", shape=2)
        self.add_constraint("g2")

    def compute(self):
        self.constraints["g2"] = self.inputs["y"][1] - 24.0


parser = argparse.ArgumentParser()
parser.add_argument(
    "--build", dest="build", action="store_true", default=False, help="Enable building"
)
parser.add_argument(
    "--link",
    choices=["all", "component", "explicit"],
    default="all",
    help="Set the type of linking procedure",
)
args = parser.parse_args()

model = am.Model("sellar")
model.add_component("disp1", 1, Disp1())
model.add_component("disp2", 1, Disp2())
model.add_component("obj", 1, Objective())
model.add_component("con1", 1, Con1())
model.add_component("con2", 1, Con2())

if args.link == "all":
    # Link the variables together with the same names
    model.link_by_name()
elif args.link == "component":
    # Links by component
    model.link_by_name("disp1", "disp2")
    model.link_by_name("disp1", "obj")
    model.link_by_name("disp1", "con1")
    model.link_by_name("disp1", "con2")
elif args.link == "explicit":
    # Explicit links
    model.link("disp1.z", "disp2.z")
    model.link("disp1.y", "disp2.y")

    model.link("disp1.x", "obj.x")
    model.link("disp1.z", "obj.z")
    model.link("disp1.y", "obj.y")

    model.link("disp1.y", "con1.y")
    model.link("disp1.y", "con2.y")

if args.build:
    model.build_module()

model.initialize()

# Create a vector to store the solution
x = model.create_vector()

opt = am.Optimizer(model, x)
data = opt.optimize(
    {
        "initial_barrier_param": 0.1,
        "max_iterations": 500,
    }
)
