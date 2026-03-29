# 🚀 Running MjLab Notebooks Locally on Linux

## Complete Setup Guide for Beginners

### Prerequisites
- Linux distribution (Ubuntu, Debian, Fedora, Arch, etc.)
- Terminal access
- Internet connection

---

## Step 1: Install uv (Python package manager)

### Option 1: Using the official installer (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:
```bash
source $HOME/.cargo/env
```

### Option 2: Using your package manager

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Fedora:**
```bash
sudo dnf install curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Arch Linux:**
```bash
sudo pacman -S uv
```

Verify installation:
```bash
uv --version
```

---

## Step 2: Install Git (if not already installed)

**Ubuntu/Debian:**
```bash
sudo apt install git
```

**Fedora:**
```bash
sudo dnf install git
```

**Arch Linux:**
```bash
sudo pacman -S git
```

---

## Step 3: Clone the Repository

Navigate to where you want to store the project:

```bash
cd ~/Documents
```

Clone the repository from the correct branch:

```bash
git clone -b feature/motor-database-extension https://github.com/robomotic/mjlab.git
```

Enter the project directory:

```bash
cd mjlab
```

---

## Step 4: Install Project Dependencies

The project uses `uv` to manage dependencies. Simply run:

```bash
uv sync
```

This command will:
- Create a virtual environment automatically
- Install Python (if needed)
- Install all required packages from `pyproject.toml`

**Note**: This may take a few minutes on first run.

---

## Step 5: Install Jupyter Notebook Support

Install Jupyter and related packages:

```bash
uv add --dev jupyter matplotlib ipywidgets
```

This adds:
- `jupyter` - Notebook interface
- `matplotlib` - Plotting library (already included, but ensures it's available)
- `ipywidgets` - Interactive widgets for notebooks

---

## Step 6: Run the Tutorial Notebook

Start Jupyter:

```bash
uv run jupyter notebook notebooks/electrical/01_intro.ipynb
```

This will:
1. Open your web browser automatically
2. Display the notebook interface
3. Load the tutorial notebook

**Alternative**: To see all notebooks first:

```bash
uv run jupyter notebook notebooks/
```

Then navigate to `electrical/01_intro.ipynb` in the browser.

---

## 🎯 Quick Start Commands (All-in-One)

If you already have `uv` and `git` installed:

```bash
# Navigate to your preferred directory
cd ~/Documents

# Clone repository
git clone -b feature/motor-database-extension https://github.com/robomotic/mjlab.git
cd mjlab

# Install dependencies
uv sync

# Install Jupyter
uv add --dev jupyter matplotlib ipywidgets

# Run the tutorial
uv run jupyter notebook notebooks/electrical/01_intro.ipynb
```

---

## 📝 Additional Commands

### Run Interactive CartPole Simulation:

Run CartPole with electrical motor and battery physics:

```bash
# Basic sinusoidal torque command
uv run play Mjlab-Cartpole-Constant-Rotation --agent sin --viewer viser

# Custom frequency and amplitude
uv run play Mjlab-Cartpole-Constant-Rotation --agent sin --sin_frequency 2.0 --sin_amplitude 20.0 --viewer viser

# Record video (saved to logs/play/...)
uv run play Mjlab-Cartpole-Constant-Rotation --agent sin --viewer viser --video --video_length 500

# Different agents
uv run play Mjlab-Cartpole-Constant-Rotation --agent zero --viewer viser  # No torque
uv run play Mjlab-Cartpole-Constant-Rotation --agent random --viewer viser  # Random torque
```

**Agent options:**
- `--agent sin` - Sinusoidal torque pattern (simulates walking gaits)
  - `--sin_frequency 1.0` - Frequency in Hz (default: 1 Hz)
  - `--sin_amplitude 15.0` - Amplitude in N⋅m (default: 15 N⋅m)
- `--agent zero` - No torque (gravity only)
- `--agent random` - Random torque commands
- `--agent trained` - Use trained RL policy (requires checkpoint)

**Video recording options:**
- `--video` - Enable video recording
- `--video_length 500` - Record 500 frames (default: 200)
- `--video_width 1280` - Video width in pixels (default: 640)
- `--video_height 720` - Video height in pixels (default: 480)

The viewer shows real-time metrics:
- Motor voltage, current, power
- Battery state of charge (SOC)
- Motor and battery temperature
- Torque commands vs actual torque

### Run all notebooks in a browser interface:
```bash
uv run jupyter notebook
```

### Run JupyterLab (modern interface):
```bash
uv add --dev jupyterlab
uv run jupyter lab
```

### Update dependencies:
```bash
uv sync --upgrade
```

### Run type checking (optional):
```bash
make type
```

### Run tests (optional):
```bash
make test
```

---

## 🔧 Troubleshooting

### Issue: "command not found: uv"
**Solution**: Install uv using the official installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Issue: "command not found: git"
**Solution**: Install git using your package manager:
```bash
# Ubuntu/Debian
sudo apt install git

# Fedora
sudo dnf install git

# Arch Linux
sudo pacman -S git
```

### Issue: Jupyter kernel not found
**Solution**: Restart the Jupyter server:
```bash
# Press Ctrl+C to stop Jupyter, then restart:
uv run jupyter notebook notebooks/electrical/01_intro.ipynb
```

### Issue: Import errors in notebook
**Solution**: Make sure you're running Jupyter via `uv run`:
```bash
uv run jupyter notebook notebooks/electrical/01_intro.ipynb
```

### Issue: Port already in use
**Solution**: Jupyter will automatically try the next available port (8889, 8890, etc.). Check the terminal output for the correct URL.

### Issue: Permission denied when installing packages
**Solution**: Never use `sudo` with `uv`. It manages its own virtual environment and doesn't need system-level permissions.

---

## 🔐 SSL Certificate Issues

### Issue: SSL certificate verification errors
**Solution**: If you encounter SSL certificate errors when installing packages with `uv`, set this environment variable:

```bash
export UV_NATIVE_TLS=1
```

To make this permanent, add it to your shell configuration file:

```bash
# For bash (most common):
echo 'export UV_NATIVE_TLS=1' >> ~/.bashrc
source ~/.bashrc

# For zsh:
echo 'export UV_NATIVE_TLS=1' >> ~/.zshrc
source ~/.zshrc

# For fish:
echo 'set -gx UV_NATIVE_TLS 1' >> ~/.config/fish/config.fish
source ~/.config/fish/config.fish
```

Then retry the installation commands.

---

## 📚 What You Just Installed

- **uv**: Modern Python package manager (faster than pip)
- **mjlab**: The main simulation library
- **jupyter**: Interactive notebook environment
- **matplotlib**: Plotting and visualization
- **torch**: PyTorch for GPU-accelerated computation
- **mujoco**: Physics engine
- Plus many other dependencies automatically managed by `uv`

---

## 🎓 Next Steps

1. **Run the tutorial**: Execute cells in `01_intro.ipynb` by pressing `Shift+Enter`
2. **Explore other notebooks**: Check `notebooks/humanoid_motor_demo.ipynb` and `notebooks/humanoid_motor_demo_easy.ipynb`
3. **Read the docs**: See `docs/` folder for detailed documentation
4. **Read CLAUDE.md**: Development workflow and coding standards

---

## 🆘 Getting Help

- **Issues**: https://github.com/robomotic/mjlab/issues
- **Development Guide**: See `CLAUDE.md` in the repository root
- **Community**: Check the repository discussions

---

**You're all set!** 🎉 The notebook should now be running in your browser.
