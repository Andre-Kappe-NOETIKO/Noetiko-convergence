#!/bin/bash
echo "Initializing NOETIKO Environment..."

# 1. Update Pip
python -m pip install --upgrade pip

# 2. Install Scientific Stack
pip install numpy scipy matplotlib pandas

# 3. Install Quantum Stack
pip install qutip torch

echo "✅ Installation complete. Run simulations via: python simulations/qutip_lindblad_model.py"
