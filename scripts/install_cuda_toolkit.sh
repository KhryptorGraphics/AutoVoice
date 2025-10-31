#!/bin/bash
# AutoVoice CUDA Toolkit Installation Script
#
# Automated installation of CUDA toolkit and drivers for Ubuntu/Debian systems.
# Installs the complete CUDA toolkit required for PyTorch CUDA extension building.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Unicode symbols
CHECK="✓"
CROSS="✗"
INFO="ℹ"
WARNING="⚠"
GEAR="⚙"
DOWNLOAD="⬇"
PACKAGE="📦"
GPU="🎮"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default CUDA version (aligned with PyTorch cu121 requirements)
DEFAULT_CUDA_VERSION="12.1"
CUDA_VERSION="${CUDA_VERSION:-$DEFAULT_CUDA_VERSION}"

# Installation flags
INSTALL_NVIDIA_DRIVERS=true
INSTALL_CUDA_TOOLKIT=true
INSTALL_PYTORCH_CUDA=true
FORCE_INSTALL=false

# Header
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        CUDA Toolkit Installation                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${BLUE}[${GEAR}]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[${CHECK}]${NC} $1"
}

print_error() {
    echo -e "${RED}[${CROSS}]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[${WARNING}]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[${INFO}]${NC} $1"
}

print_download() {
    echo -e "${BLUE}[${DOWNLOAD}]${NC} $1"
}

print_package() {
    echo -e "${BLUE}[${PACKAGE}]${NC} $1"
}

print_gpu() {
    echo -e "${BLUE}[${GPU}]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root"
        print_info "It's recommended to avoid running CUDA installation as root when possible"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Function to detect OS
detect_os() {
    print_status "Detecting operating system..."

    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
        print_success "Detected OS: ${PRETTY_NAME}"
    else
        print_error "Unable to detect operating system"
        print_info "This script is designed for Ubuntu/Debian systems"
        exit 1
    fi

    # Verify supported OS
    case $OS in
        ubuntu|debian|linuxmint|elementary|zorin)
            print_info "OS is supported for CUDA installation"
            ;;
        *)
            print_warning "OS '$OS' may not be officially supported by NVIDIA"
            print_info "This script is optimized for Ubuntu/Debian systems"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            ;;
    esac
}

# Function to update package lists
update_package_lists() {
    print_status "Updating package lists..."

    if command -v apt &> /dev/null; then
        sudo apt update
        print_success "Package lists updated"
    else
        print_error "apt package manager not found"
        print_info "This script requires apt (Ubuntu/Debian systems)"
        exit 1
    fi
}

# Function to install NVIDIA drivers
install_nvidia_drivers() {
    if [ "$INSTALL_NVIDIA_DRIVERS" = false ]; then
        print_info "Skipping NVIDIA driver installation (--no-drivers)"
        return 0
    fi

    print_gpu "Installing NVIDIA drivers..."

    # Remove existing NVIDIA packages if force install
    if [ "$FORCE_INSTALL" = true ]; then
        print_warning "Removing existing NVIDIA packages (--force)"
        sudo apt purge -y nvidia-* libnvidia-* 2>/dev/null || true
        sudo apt autoremove -y
        sudo apt autoclean
    fi

    # Install required packages for NVIDIA drivers
    print_package "Installing prerequisites for NVIDIA drivers..."
    sudo apt install -y software-properties-common build-essential gcc-multilib dkms

    # Add graphics drivers PPA for Ubuntu
    if [ "$OS" = "ubuntu" ]; then
        print_info "Adding graphics drivers PPA..."
        sudo add-apt-repository -y ppa:graphics-drivers/ppa
        sudo apt update
    fi

    # Install NVIDIA driver (compatible with CUDA 12.1)
    print_package "Installing NVIDIA driver..."
    sudo apt install -y nvidia-driver-535  # Compatible with CUDA 12.1 (driver >= 535 required)

    print_success "NVIDIA drivers installed"
    print_warning "System restart required for driver activation"
    DRIVER_INSTALLED=true
}

# Function to install CUDA toolkit
install_cuda_toolkit() {
    if [ "$INSTALL_CUDA_TOOLKIT" = false ]; then
        print_info "Skipping CUDA toolkit installation (--no-toolkit)"
        return 0
    fi

    print_download "Installing CUDA Toolkit ${CUDA_VERSION}..."

    # Remove existing CUDA installation if force install
    if [ "$FORCE_INSTALL" = true ]; then
        print_warning "Removing existing CUDA installation (--force)"
        sudo apt purge -y cuda-* libcudnn* 2>/dev/null || true
        sudo apt autoremove -y
        sudo apt autoclean
    fi

    # Add NVIDIA CUDA repository
    print_info "Adding NVIDIA CUDA repository..."

    # Download and install CUDA keyring
    wget -O /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}${VERSION}/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring.deb
    rm /tmp/cuda-keyring.deb

    sudo apt update

    # Install CUDA toolkit
    CUDA_PKG="cuda-toolkit-${CUDA_VERSION//./-}"
    print_package "Installing ${CUDA_PKG}..."
    sudo apt install -y "$CUDA_PKG"

    # Set CUDA environment variables
    CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
    if [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    fi

    # Update .bashrc with CUDA paths
    BASHRC="$HOME/.bashrc"
    if [ ! -f "$BASHRC" ]; then
        touch "$BASHRC"
    fi

    # Add CUDA to PATH and LD_LIBRARY_PATH if not already present
    if ! grep -q "export PATH.*cuda" "$BASHRC"; then
        echo "export CUDA_HOME=\"${CUDA_HOME}\"" >> "$BASHRC"
        echo "export PATH=\"${CUDA_HOME}/bin:\$PATH\"" >> "$BASHRC"
        echo "export LD_LIBRARY_PATH=\"${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH\"" >> "$BASHRC"
        print_info "Added CUDA paths to $BASHRC"
    fi

    # Source the updated bashrc
    source "$BASHRC"

    print_success "CUDA Toolkit ${CUDA_VERSION} installed"
    print_info "CUDA_HOME: ${CUDA_HOME}"
    CUDA_INSTALLED=true
}

# Function to install PyTorch with CUDA
install_pytorch_cuda() {
    if [ "$INSTALL_PYTORCH_CUDA" = false ]; then
        print_info "Skipping PyTorch CUDA installation (--no-pytorch)"
        return 0
    fi

    print_package "Installing PyTorch with CUDA support..."

    # Uninstall existing PyTorch if force install
    if [ "$FORCE_INSTALL" = true ]; then
        print_warning "Removing existing PyTorch installation (--force)"
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    fi

    # Install PyTorch with CUDA support
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121

    # Verify PyTorch CUDA installation
    print_status "Verifying PyTorch CUDA installation..."
    if python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"; then
        print_success "PyTorch with CUDA installed successfully"
        PYTORCH_INSTALLED=true
    else
        print_error "PyTorch CUDA installation failed"
        return 1
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."

    # Check NVIDIA driver
    if [ "$DRIVER_INSTALLED" = true ]; then
        if nvidia-smi &>/dev/null; then
            print_success "NVIDIA driver working"
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
            print_gpu "GPU: $GPU_INFO"
        else
            print_error "NVIDIA driver not working"
            return 1
        fi
    fi

    # Check CUDA toolkit
    if [ "$CUDA_INSTALLED" = true ]; then
        if nvcc --version &>/dev/null; then
            CUDA_VER=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
            print_success "CUDA toolkit working: $CUDA_VER"

            # Check for critical headers
            if [ -f "${CUDA_HOME}/include/nv/target" ]; then
                print_success "CUDA build headers available"
            else
                print_error "Critical CUDA build headers missing (nv/target)"
                return 1
            fi
        else
            print_error "CUDA toolkit not working"
            return 1
        fi
    fi

    # Check PyTorch CUDA
    if [ "$PYTORCH_INSTALLED" = true ]; then
        if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" &>/dev/null; then
            print_success "PyTorch CUDA working"
        else
            print_error "PyTorch CUDA not working"
            return 1
        fi
    fi

    print_success "Installation verification completed"
    return 0
}

# Function to show post-installation instructions
show_post_install_instructions() {
    echo ""
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Post-Installation Instructions${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

    if [ "$DRIVER_INSTALLED" = true ]; then
        echo ""
        print_warning "IMPORTANT: System restart required for NVIDIA drivers"
        echo "  Run: sudo reboot"
        echo "  After restart, run: nvidia-smi (to verify GPU)"
    fi

    echo ""
    print_info "To activate CUDA environment in new terminals:"
    echo "  source ~/.bashrc"
    echo ""
    print_info "To verify CUDA toolkit installation:"
    echo "  $SCRIPT_DIR/check_cuda_toolkit.sh"
    echo ""
    print_info "To build AutoVoice with CUDA extensions:"
    echo "  cd $PROJECT_ROOT"
    echo "  pip install -e ."
    echo ""
    print_info "To run full test suite:"
    echo "  $SCRIPT_DIR/build_and_test.sh"
}

# Function to parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-drivers)
                INSTALL_NVIDIA_DRIVERS=false
                shift
                ;;
            --no-toolkit)
                INSTALL_CUDA_TOOLKIT=false
                shift
                ;;
            --no-pytorch)
                INSTALL_PYTORCH_CUDA=false
                shift
                ;;
            --force)
                FORCE_INSTALL=true
                shift
                ;;
            --cuda-version)
                CUDA_VERSION="$2"
                shift
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Install CUDA toolkit and dependencies for AutoVoice"
                echo ""
                echo "Options:"
                echo "  --no-drivers     Skip NVIDIA driver installation"
                echo "  --no-toolkit     Skip CUDA toolkit installation"
                echo "  --no-pytorch     Skip PyTorch CUDA installation"
                echo "  --force          Force reinstall (removes existing packages)"
                echo "  --cuda-version   Specify CUDA version (default: ${DEFAULT_CUDA_VERSION})"
                echo "  --help          Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                           # Full installation"
                echo "  $0 --no-drivers             # Skip driver install"
                echo "  $0 --cuda-version 12.1      # Install CUDA 12.1"
                echo "  $0 --force                  # Force reinstall everything"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Main installation function
main() {
    # Parse command line arguments
    parse_args "$@"

    # Header with configuration
    echo ""
    print_info "Installation Configuration:"
    echo "  Install NVIDIA Drivers: $INSTALL_NVIDIA_DRIVERS"
    echo "  Install CUDA Toolkit: $INSTALL_CUDA_TOOLKIT ($CUDA_VERSION)"
    echo "  Install PyTorch CUDA: $INSTALL_PYTORCH_CUDA"
    echo "  Force Install: $FORCE_INSTALL"
    echo ""

    # Pre-installation checks
    check_root
    detect_os
    update_package_lists

    # Installation steps
    if [ "$INSTALL_NVIDIA_DRIVERS" = true ]; then
        install_nvidia_drivers
    fi

    if [ "$INSTALL_CUDA_TOOLKIT" = true ]; then
        install_cuda_toolkit
    fi

    if [ "$INSTALL_PYTORCH_CUDA" = true ]; then
        install_pytorch_cuda
    fi

    # Verification
    verify_installation

    # Success message
    echo ""
    print_success "CUDA installation completed successfully!"
    show_post_install_instructions
}

# Run main function
main "$@"
