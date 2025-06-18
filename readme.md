# Parametric PINN for Autoclave Reactor Optimization

This repository contains a comprehensive framework for **generalizable Physics-Informed Neural Network (PINN) based CFD simulation**, **compartmental reaction modeling**, and **inverse-design optimization** of autoclave reactors for ethylene polymerization processes. The framework enables blade geometry optimization without multiple CFD runs through parametric learning.

## 🚀 Overview

The framework combines three main components to achieve inverse-design of reactor blade geometry:

1. **Parametric PINN Flow Simulation** - Learns universal flow physics across variable blade geometries
2. **Compartmental Reaction Analysis** - Couples flow patterns with ethylene polymerization kinetics  
3. **Inverse-Design Optimization** - Optimizes blade geometry and operating conditions for targeted molecular weight distribution (MWD)

### Key Features

- **Parametric Learning** for blade geometry generalization via geometry-integrated loss
- **Zero-equation turbulence model** with mixing length approach for high-viscosity polymer solutions
- **Moving Reference Frame (MRF)** for accurate blade-fluid interaction modeling
- **Modified Fourier Neural Networks** to capture periodic features of fluid dynamics
- **Method of Moments (MOM)** for efficient polymerization kinetics simulation
- **Joint optimization** of blade parameters and operating conditions

## 📁 Repository Structure

```
├── custom_equation.py          # PDE equations (Navier-Stokes + zero-equation turbulence)
├── custom_plotter.py           # Visualization and validation plotting
├── custom_primitives_3d.py     # 3D geometric primitives for autoclave reactor
├── geometry.py                 # Complete reactor geometry with compartmentalization
├── flow.py                     # Parametric PINN flow simulation (Step 1)
├── reaction.py                 # Compartmental ethylene polymerization analysis (Step 2)
├── optimization.py             # Inverse-design optimization for MWD control (Step 3)
├── conf/
│   ├── config_flow.yaml        # Configuration for parametric PINN training
│   ├── config_reaction.yaml    # Configuration for reaction analysis
│   └── config_optimization.yaml # Configuration for inverse-design optimization
└── fluent/
    └── 001.csv                 # Reference CFD data for validation
```

## 🔧 Prerequisites

### Required Software
- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)

### Required Python Packages
```bash
# Core scientific packages
numpy
scipy
pandas
matplotlib

# Deep learning
torch
modulus  # NVIDIA Modulus framework

# Symbolic computation
sympy

# Optimization
scipy (for DIRECT algorithm)

# Configuration management
hydra-core
```

### Installation
```bash
# Clone repository
git clone <repository-url>
cd stirred-reactor-optimization

# Install NVIDIA Modulus (follow official installation guide)
pip install nvidia-modulus

# Install other dependencies
pip install -r requirements.txt
```

## 🔄 Workflow

The complete workflow consists of three sequential steps:

### Step 1: Parametric PINN Flow Simulation (`flow.py`)

**Purpose**: Train a generalizable PINN that learns universal flow physics across variable blade geometries.

**What it does**:
- Sets up autoclave reactor geometry with parameterized blade system
- Implements geometry-integrated loss function for parametric learning
- Defines incompressible Navier-Stokes equations with zero-equation turbulence model
- Applies Moving Reference Frame (MRF) for blade-fluid interactions
- Trains Modified Fourier Neural Network to capture periodic fluid dynamics
- Validates against CFD reference data across multiple blade configurations

**Key Physics**:
- Incompressible Navier-Stokes equations with MRF
- Zero-equation turbulence model (mixing length = 0.0063 m)
- Parametric boundary conditions for blade geometry generalization
- Ethylene monomer inlet with parabolic velocity profile

**Parametric Variables**:
- φᵣ: Blade radius parameter [-1, 1] → radius = 0.05 + 0.02×φᵣ [m]
- φₕ: Blade height parameter [-1, 1] → height = 0.09×φₕ [m]
- φ_Ω: Stirring rate parameter [-1, 1] → Ω = 10×φ_Ω [rad/s]

**Usage**:
```bash
python flow.py
```

**Outputs**:
- `outputs/flow/flow_network.*.pth` - Trained parametric PINN weights
- `outputs/flow/optim_checkpoint.*.pth` - Training checkpoint
- Validation plots comparing PINN vs CFD across blade configurations

**Runtime**: ~2-6 hours (parametric training across geometry space)

---

### Step 2: Compartmental Ethylene Polymerization Analysis (`reaction.py`)

**Purpose**: Extract flow patterns and simulate free radical ethylene polymerization kinetics.

**What it does**:
- Loads trained parametric PINN from Step 1
- Divides autoclave reactor into 100 compartments (5×5×4 radial×axial×tangential)
- Extracts inter-compartmental flow rates using parametric PINN
- Solves Method of Moments (MOM) equations for polymerization kinetics
- Calculates ethylene conversion and molecular weight distribution (MWD)

**Key Physics**:
- Free radical polymerization kinetics (initiation, propagation, termination, chain transfer)
- Method of Moments for efficient polymer chain statistics
- Mass transfer between compartments based on PINN flow fields
- Radical initiator side-stream injection modeling

**Reaction System**:
- Ethylene monomer feed through main inlet (18.57 mol/L)
- Radical initiator injection through side inlet
- Temperature: 450 K (isothermal operation)
- Compartmental CSTR cascade approximation

**Dependencies**: Requires completed `flow.py` execution

**Usage**:
```bash
python reaction.py
```

**Outputs**:
- `kinetics_results.csv` - Final concentrations and MWD in all compartments
- Flow rate data in `outputs/reaction/monitors/`
- Ethylene conversion and polymer MWD analysis

**Runtime**: ~30 minutes

---

### Step 3: Inverse-Design Optimization for MWD Control (`optimization.py`)

**Purpose**: Find optimal blade geometry and operating conditions for targeted molecular weight distribution.

**What it does**:
- Uses dual annealing algorithm for global optimization
- Evaluates design candidates using parametric PINN + compartment model
- Optimizes 5 variables: 3 blade parameters + 2 operating conditions
- Minimizes RMSE between target and predicted MWD

**Design Variables**:
- φᵣ: Blade radius parameter [-1, 1]
- φₕ: Blade height parameter [-1, 1]  
- φ_Ω: Stirring rate parameter [-1, 1]
- T: Reaction temperature [430, 480] K
- C: Initiator concentration [0.001, 1.0] mol/L

**Objective Function**: 
J = Σᵢ 100×(MWDᵢᵗᵃʳᵍᵉᵗ - MWDᵢʳᵉˢᵘˡᵗ)²

**Dependencies**: Requires completed `flow.py` execution

**Usage**:
```bash
python optimization.py
```

**Outputs**:
- `outputs/optimization/optimal_solution.npy` - Best blade geometry and conditions
- `outputs/optimization/conversion.npy` - R² score evolution
- `outputs/optimization/parameter_history.npy` - Parameter search history
- MWD matching results with target distribution

**Runtime**: ~5-10 hours (rapid evaluation < 3 min per candidate)

## ⚙️ Configuration

Each script uses Hydra configuration files in the `conf/` directory:

### `config_flow.yaml`
```yaml
# Training configuration for flow simulation
run_mode: 'train'
optimizer:
  lr: 1e-3
training:
  max_steps: 10000000
  rec_results_freq: 10000
```

### `config_reaction.yaml` & `config_optimization.yaml`  
```yaml
# Evaluation configuration for reaction/optimization
run_mode: 'eval'
training:
  rec_results_freq: 100
```

**Important**: Do not modify the configuration files unless you understand the implications for model training and evaluation.

## 📊 Key Results

### Parametric PINN Validation
The framework generates validation plots showing:
- Velocity field comparisons across 6 blade configurations (PINN vs CFD)
- Relative RMSE below 4% for all velocity components
- Streamline pattern accuracy across parameter space
- Generalization capability without retraining

### MWD Optimization Results
- **Simultaneous optimization**: R² = 0.993 (excellent MWD matching)
- **Fixed blade optimization**: R² = 0.869 (limited without blade design)
- **Computation efficiency**: < 3 min per design evaluation
- **Parameter sensitivity**: Extended blade radius and elevated height amplify stirring effects

### Example Output
```
INVERSE-DESIGN OPTIMIZATION COMPLETED
=====================================
Optimal blade geometry and operating conditions:
  Blade radius: 0.0621 m (φᵣ = 0.242)
  Blade height: -0.0362 m (φₕ = -0.402)  
  Stirring rate: 1.17 rad/s (φ_Ω = 0.117)
  Temperature: 473 K
  Initiator concentration: 0.0031 mol/L

MWD Matching Performance: R² = 0.993

Optimal molecular weight: Mw = 15,230 g/mol
Target achieved with 99.3% accuracy
```

## 🔬 Technical Details

### Autoclave Reactor Geometry
- **Main vessel**: Cylindrical tank (0.1 m radius, 0.2 m height)
- **Blade system**: Dual-blade configuration with variable geometry
- **Parameterization**: Normalized parameters φ ∈ [-1, 1] for blade radius, height, stirring rate
- **Design ranges**: Radius 0.03-0.07 m, Height ±0.09 m, Stirring 0-10 rad/s

### Parametric PINN Model
- **Architecture**: Modified Fourier Neural Network (6 layers, 512 neurons)
- **Inputs**: 3D coordinates (x,y,z) + 3 blade parameters (φᵣ,φₕ,φ_Ω)
- **Outputs**: Velocity components (u,v,w) + pressure (p)
- **Training**: Geometry-integrated loss with Monte Carlo integration
- **Physics**: Incompressible Navier-Stokes + zero-equation turbulence + MRF

### Compartmental Model
- **Discretization**: 100 compartments (5×5×4 radial×axial×tangential)
- **Approach**: CSTR cascade network with inter-compartmental flow rates
- **Coupling**: PINN velocity profiles → compartmental flow rates → reaction kinetics

### Ethylene Polymerization Kinetics
- **Mechanism**: Free radical polymerization with chain transfer and β-scission
- **Method**: Method of Moments (0th, 1st, 2nd moments)
- **Reactions**: Initiation, propagation, termination (combination/disproportionation), chain transfer
- **Output**: Molecular weight distribution (MWD) and conversion

### Optimization Algorithm
- **Method**: Dual annealing (global optimization)
- **Variables**: 5D parameter space (3 blade + 2 operating parameters)  
- **Objective**: Minimize MWD RMSE from target distribution
- **Convergence**: R² score tracking for MWD matching quality

## 🚨 Important Notes

### Memory Management
- Each optimization iteration requires ~4-8 GB GPU memory
- Automatic cleanup routines clear GPU memory between evaluations
- Monitor system resources during long optimization runs

### File Dependencies
- Step 2 requires trained model from Step 1
- Step 3 requires trained model from Step 1
- Missing files will cause runtime errors

### Computational Requirements
- GPU highly recommended for PINN training
- CPU-only execution possible but significantly slower
- Consider cloud computing for large optimization studies

### Validation Data
- Reference data in `fluent/001.csv` must match expected format
- Used for model validation and accuracy assessment
- Can be replaced with experimental data if available

## 🛠️ Troubleshooting

### Common Issues

**GPU Memory Errors**:
```bash
# Reduce batch sizes in configuration files
# Monitor GPU memory usage
nvidia-smi
```

**Training Convergence Issues**:
```bash
# Check loss curves in output logs
# Adjust learning rate or training steps
# Verify boundary condition implementation
```

**File Not Found Errors**:
```bash
# Ensure previous steps completed successfully
# Check file paths in scripts
# Verify output directories exist
```

### Performance Optimization
- Use mixed precision training for faster computation
- Adjust batch sizes based on available GPU memory
- Consider distributed training for large parameter studies

## 📈 Extensions and Future Work

### Potential Enhancements
- **Multi-objective optimization** (conversion + mixing time)
- **Uncertainty quantification** for robust design
- **Experimental validation** with real reactor data
- **3D compartmental models** for improved accuracy
- **Different impeller geometries** (Rushton, pitched blade, etc.)
- **Heat transfer modeling** for non-isothermal operation

### Research Applications
- **Process intensification** studies
- **Scale-up analysis** using dimensionless groups
- **Control system design** for optimal operation
- **Digital twin development** for real-time optimization

## 📄 Citation

If you use this framework in your research, please cite:

```bibtex
@article{shin2024optimization,
  title={Optimization of Reactor Geometry Using Physics-Informed Neural Networks: Generalizing Flow Fields Across Variable Geometries},
  author={Shin, Sunkyu and Ryu, Yubin and Na, Jonggeol and Lee, Won Bo},
  journal={[Journal Name]},
  year={2024},
  note={Manuscript submitted}
}
```

**Key Contributors:**
- Sunkyu Shin¹'² - MIT & Seoul National University
- Yubin Ryu³'⁴ - Ewha Womans University  
- Jonggeol Na³'⁴* - Ewha Womans University
- Won Bo Lee²* - Seoul National University

*Corresponding authors

## 📞 Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check documentation in code comments
- Review NVIDIA Modulus documentation for PINN details

## 📜 License

[Specify your license here - MIT, Apache 2.0, etc.]

---

**Happy optimizing! 🚀**