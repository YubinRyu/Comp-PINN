# Simultaneous Optimization of Reactor Geometry and Operating Conditions via Geometry-Aware Physics-Informed Neural Networks (PINN)

This repository contains a comprehensive framework for **generalizable Physics-Informed Neural Network (PINN) based CFD simulation**, **compartmental reaction modeling**, and **inverse-design optimization** of autoclave reactors for ethylene polymerization processes. The framework enables blade geometry optimization without multiple CFD runs through parametric learning.

## 🚀 Overview

The framework combines three main components to achieve inverse-design of reactor blade geometry:

1. **Parametric PINN Flow Simulation** - Learns universal flow physics across variable geometries and operating conditions
2. **Compartmental Reaction Analysis** - Couples flow patterns with ethylene polymerization kinetics
3. **Inverse-Design Optimization** - Optimizes blade geometry and operating conditions for targeted molecular weight distribution (MWD)

### Key Features
- **Parametric Learning** for blade geometry generalization
- **Zero-equation turbulence model** with mixing length approach for high-viscosity polymer solutions
- **Moving Reference Frame (MRF)** for accurate blade-fluid interaction modeling
- **Modified Fourier Neural Networks** to capture periodic features of fluid dynamics
- **Method of Moments (MOM)** for efficient polymerization kinetics simulation
- **Joint optimization** of blade parameters and operating conditions

## 📁 Repository Structure

```
├── custom_equation.py          # Governing Equations (Navier-Stokes + zero-equation turbulence)
├── custom_plotter.py           # Visualization and Validation
├── custom_primitives_3d.py     # 3D Geometric Primitives
├── geometry.py                 # Reactor Blade Parameterization
├── flow.py                     # Parametric PINN Flow Simulation (Step 1)
├── reaction.py                 # Compartmental Reaction analysis (Step 2)
├── optimization.py             # Inverse-design optimization for MWD Control (Step 3)
├── conf/
│   ├── config_flow.yaml
│   ├── config_reaction.yaml
│   └── config_optimization.yaml
└── fluent/
    └── 001.csv                 # Reference CFD Data for Validation
```

## 🔧 Prerequisites

### Required Software
This project is implemented using **[NVIDIA PhysicsNeMo](https://developer.nvidia.com/physicsnemo)**. Please follow the official installation guide for your system.

### Installation
```bash
# Clone repository
git clone https://github.com/YubinRyu/Comp-PINN.git
cd Comp-PINN
```

## 🔄 Workflow

The complete workflow consists of three sequential steps:

### Step 1: Parametric PINN Flow Simulation (`flow.py`)
**Purpose**: Train a generalizable PINN that learns universal flow physics across variable blade geometries.
   
**What it does**:
- Sets up autoclave reactor geometry with parameterized blade system
- Defines incompressible Navier-Stokes equations with zero-equation turbulence model
- Applies Moving Reference Frame (MRF) for blade-fluid interactions
- Trains Modified Fourier Neural Network to capture periodic fluid dynamics
- Validates against CFD reference data across multiple blade configurations

**Key Physics**:
- Incompressible Navier-Stokes equations with MRF
- Zero-equation turbulence model
- Parametric boundary conditions for blade geometry generalization
- Ethylene monomer inlet with parabolic velocity profile

**Variables**:
- Inputs: normalized blade parameters **ϕ = (ϕΩ, ϕr, ϕh)** and spatial coordinates (x, y, z).  
- Outputs: velocity components (u, v, w) and pressure p.  
- φᵣ: Blade radius parameter [-1, 1] → radius = 0.05 + 0.02×φᵣ [m]
- φₕ: Blade height parameter [-1, 1] → height = 0.09×φₕ [m]
- ϕΩ: Stirring rate parameter [-1, 1] → Ω = 10×ϕΩ [rad/s]

**Usage**:
```bash
python flow.py
```

**Outputs**:
- `outputs/flow/flow_network.0.pth` - Weights and Biases
- `outputs/flow/optim_checkpoint.0.pth` - Checkpoint
- Validation plots comparing PINN vs CFD across blade configurations

**Runtime**: ~2-6 days (parametric training across geometry space)

---

### Step 2: Compartmental Reaction Analysis (`reaction.py`)
**Purpose**: Extract flow patterns and simulate free radical ethylene polymerization kinetics.

**What it does**:
- Loads trained geometry-aware PINN model from Step 1
- Divides autoclave reactor into 100 compartments (5×5×4 radial×axial×tangential)
- Solves Method of Moments (MOM) and Population Balance Equations (PBE) for free-radical polymerization
- Calculates ethylene conversion and molecular weight distribution (MWD)

**Key Physics**:
- Mass transfer between compartments based on PINN results
- Free radical polymerization kinetics (initiation, propagation, termination, chain transfer)

**Reaction System**:
- Ethylene monomer feed: 18.57 mol/L at main inlet  
- Radical initiator injection through side inlet  
- Temperature: 450 K (isothermal)  
- Kinetic parameters: k<sub>d</sub>, k<sub>p</sub>, k<sub>tc</sub>, k<sub>td</sub>, k<sub>ctm</sub>, k<sub>ctp</sub>, k<sub>β</sub>

**Dependencies**: Requires trained PINN from `flow.py`

**Usage**:
```bash
python reaction.py
```

**Outputs**:
- `kinetics_results.csv` - Species Concentrations and MWD
- Flow Data in `outputs/reaction/monitors/`
- Ethylene Conversion and MWD Plots

**Runtime**: ~3 min

---

### Step 3: Inverse-Design Optimization for MWD Control (`optimization.py`)

**Purpose**: Find optimal blade geometry and operating conditions for targeted molecular weight distribution.

**What it does**:
- Uses dual annealing algorithm for global + local optimization
- Evaluates design candidates using PINN + compartment model
- Optimizes 5 variables: 3 blade parameters + 2 operating conditions
- Minimizes RMSE between target and simulated MWD

**Design Variables**:
- φᵣ: Blade radius parameter [0.03, 0.07] m
- φₕ: Blade height parameter [−0.09, 0.09] m
- ϕΩ: Stirring rate parameter [−10, 10] rad/s
- T: Reaction temperature [400, 450] K
- C: Initiator concentration [0.001, 1.0] mol/L

**Dependencies**: Requires completed `flow.py` execution

**Usage**:
```bash
python optimization.py
```

**Outputs**:
- `outputs/optimization/conversion.npy` - Conversion History
- `outputs/optimization/parameter_history.npy` - Parameter Search History
- `outputs/optimization/time.npy` - Evaluation Time History

**Runtime**: ~5-10 days (rapid evaluation < 1 min per candidate)

## ⚙️ Configuration

Each script uses Hydra configuration files in the `conf/` directory.
**Important**: Do not modify the configuration files unless you understand the implications for model training and evaluation.

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{shin2025optimization,
  title={Optimization of Reactor Geometry Using Physics-Informed Neural Networks: Generalizing Flow Fields Across Variable Geometries},
  author={Shin, Sunkyu and Ryu, Yubin and Na, Jonggeol and Lee, Won Bo},
  journal={[Journal Name]},
  year={2025},
  note={Manuscript submitted}
}
```

**Key Contributors:**
- Sunkyu Shin - Massachusetts Institute of Technology & Seoul National University
- Yubin Ryu - Ewha Womans University 
- Jonggeol Na* - Ewha Womans University
- Won Bo Lee* - Seoul National University

*Corresponding authors
