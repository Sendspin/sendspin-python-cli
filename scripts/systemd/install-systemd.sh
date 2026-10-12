#!/bin/bash
# Sendspin systemd service installation
set -e

# Ensure output is visible even when piped
exec 2>&1

# Colors
C='\033[0;36m'
G='\033[0;32m'
Y='\033[1;33m'
R='\033[0;31m'
B='\033[1m'
D='\033[2m'
N='\033[0m'

# Detect if running interactively
INTERACTIVE=true
if [ ! -t 0 ]; then
    # stdin is not a terminal (piped)
    if [ ! -c /dev/tty ]; then
        # No TTY available - fully non-interactive
        INTERACTIVE=false
        echo "Running in non-interactive mode - using defaults" >&2
    fi
fi

# Prompt for yes/no with configurable default
# Usage: prompt_yn "question" [default]
# default can be "yes" (default) or "no"
prompt_yn() {
    local question="$1"
    local default="${2:-yes}"

    if [ "$INTERACTIVE" = true ]; then
        if [ "$default" = "no" ]; then
            read -p "$question [y/N] " -n1 -r REPLY </dev/tty; echo
            [[ $REPLY =~ ^[Yy]$ ]]
        else
            read -p "$question [Y/n] " -n1 -r REPLY </dev/tty; echo
            [[ ! $REPLY =~ ^[Nn]$ ]]
        fi
    else
        echo "$question [auto: $default]"
        [ "$default" = "yes" ]
    fi
}

# Prompt for input with default value
# Usage: VAR=$(prompt_input "prompt text" "default value")
prompt_input() {
    local prompt="$1"
    local default="$2"
    if [ "$INTERACTIVE" = true ]; then
        echo -en "${C}${prompt}${N} [$default]: " >&2
        read -r REPLY </dev/tty
        echo "${REPLY:-$default}"
    else
        echo "Using default for $prompt: $default"
        echo "$default"
    fi
}

# Install a package using the detected package manager
# Usage: install_package "canonical-package-name"
# Handles package name mapping for different distros
install_package() {
    local canonical_name="$1"
    local pkg_name="$canonical_name"  # default to canonical name

    # Map canonical package names to distro-specific names
    case "$PKG_MGR:$canonical_name" in
        pacman:libportaudio2) pkg_name="portaudio" ;;
        pacman:libopenblas0) pkg_name="openblas" ;;
        dnf:libopenblas0|yum:libopenblas0) pkg_name="openblas" ;;
        # Additional mappings can be added here as needed
    esac

    # Construct install command for the package manager
    local CMD=""
    case "$PKG_MGR" in
        pacman) CMD="pacman -S --noconfirm $pkg_name" ;;
        dnf|yum) CMD="$PKG_MGR install -y $pkg_name" ;;
        apt-get) CMD="$PKG_MGR install -y $pkg_name" ;;
        *) CMD="$PKG_MGR install -y $pkg_name" ;;
    esac

    if prompt_yn "Install now? ($CMD)"; then
        if [ "$PKG_MGR" = "apt-get" ]; then
            apt-get update || true
        fi
        $CMD || { echo -e "${R}Failed${N}"; return 1; }
        return 0
    else
        echo -e "${R}Error:${N} Package required. Install with: ${B}$CMD${N}"
        return 1
    fi
}

# Check for root
[[ $EUID -ne 0 ]] && { echo -e "${R}Error:${N} Please run with sudo or as root"; exit 1; }

echo -e "\n${B}${C}Sendspin Service Installation${N}\n"

# Determine user setup: if run as root directly, use dedicated user automatically
# If run via sudo, offer choice
USE_DEDICATED_USER=true
DAEMON_USER="sendspin"
DAEMON_HOME="/home/sendspin"

if [[ -n "$SUDO_USER" && "$SUDO_USER" != "root" ]]; then
    # Run via sudo - offer choice
    echo -e "${C}User Setup${N}"
    echo -e "${D}You can run sendspin as a dedicated 'sendspin' user (recommended)"
    echo -e "or as your current user ($SUDO_USER).${N}"
    echo ""

    if prompt_yn "Use dedicated 'sendspin' user?" "yes"; then
        DAEMON_USER="sendspin"
        DAEMON_HOME="/home/sendspin"
    else
        USE_DEDICATED_USER=false
        DAEMON_USER="$SUDO_USER"
        DAEMON_HOME="/home/$SUDO_USER"
    fi
else
    # Run as root directly - use dedicated user automatically
    echo -e "${C}User Setup${N}"
    echo -e "${D}Running as root - will use dedicated 'sendspin' user${N}"
fi

# Create sendspin user if using dedicated user and it doesn't exist
if [ "$USE_DEDICATED_USER" = true ] && ! id -u sendspin &>/dev/null; then
    echo -e "${D}Creating sendspin system user...${N}"
    useradd -r -m -d "$DAEMON_HOME" -s /bin/bash -c "Sendspin Daemon" sendspin || \
        { echo -e "${R}Failed to create user${N}"; exit 1; }

    # Add to audio group for audio device access
    usermod -a -G audio sendspin 2>/dev/null || true

    echo -e "${G}✓${N} Created sendspin system user"
elif [ "$USE_DEDICATED_USER" = true ]; then
    echo -e "${D}User 'sendspin' already exists${N}"
fi

# Enable linger so the user's systemd session (and PipeWire/PulseAudio) starts
# at boot without requiring an interactive login
loginctl enable-linger "$DAEMON_USER" 2>/dev/null || true
echo -e "${G}✓${N} Linger enabled for $DAEMON_USER"

echo -e "${D}Daemon will run as: ${B}$DAEMON_USER${N}"

# Detect package manager
PKG_MGR=""
if command -v apt-get &>/dev/null; then PKG_MGR="apt-get"
elif command -v dnf &>/dev/null; then PKG_MGR="dnf"
elif command -v yum &>/dev/null; then PKG_MGR="yum"
elif command -v pacman &>/dev/null; then PKG_MGR="pacman"
fi

echo -e "\n${C}Checking dependencies...${N}"

# Check for and offer to install libportaudio2
if ! ldconfig -p 2>/dev/null | grep -q libportaudio.so; then
    echo -e "${Y}Missing:${N} libportaudio2"
    if [[ -n "$PKG_MGR" ]]; then
        install_package "libportaudio2" || exit 1
    else
        echo -e "${R}Error:${N} libportaudio2 required. Install via your package manager."
        exit 1
    fi
fi

# Check for and offer to install libopenblas0
if ! ldconfig -p 2>/dev/null | grep -q libopenblas.so; then
    echo -e "${Y}Missing:${N} libopenblas0"
    if [[ -n "$PKG_MGR" ]]; then
        install_package "libopenblas0" || exit 1
    else
        echo -e "${R}Error:${N} libopenblas0 required. Install via your package manager."
        exit 1
    fi
fi

# Check for and offer to install uv if needed
if ! sudo -u "$DAEMON_USER" bash -l -c "command -v uv" &>/dev/null && \
   ! sudo -u "$DAEMON_USER" test -f "$DAEMON_HOME/.cargo/bin/uv" && \
   ! sudo -u "$DAEMON_USER" test -f "$DAEMON_HOME/.local/bin/uv"; then
    echo -e "${Y}Missing:${N} uv"
    if prompt_yn "Install now? (curl -LsSf https://astral.sh/uv/install.sh | sh)"; then
        sudo -u "$DAEMON_USER" bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh" || { echo -e "${R}Failed${N}"; exit 1; }
        echo -e "${G}✓${N} uv installed"
    else
        echo -e "${R}Error:${N} uv required. Install with: ${B}curl -LsSf https://astral.sh/uv/install.sh | sh${N}"; exit 1
    fi
fi

# Install or upgrade sendspin
echo -e "\n${C}Installing Sendspin...${N}"
if sudo -u "$DAEMON_USER" bash -l -c "uv tool list" 2>/dev/null | grep -q "^sendspin "; then
    echo -e "${D}Sendspin already installed, upgrading...${N}"
    sudo -u "$DAEMON_USER" bash -l -c "uv tool upgrade sendspin" || { echo -e "${R}Failed${N}"; exit 1; }
    echo -e "  ${C}Release notes:${N} https://github.com/Sendspin/sendspin-cli/releases"
else
    sudo -u "$DAEMON_USER" bash -l -c "uv tool install --python 3.13 sendspin" || { echo -e "${R}Failed${N}"; exit 1; }
fi

# Grab the proper bin path from uv (in case it's non-standard)
SENDSPIN_BIN="$(sudo -u "$DAEMON_USER" bash -l -c "uv tool dir --bin")/sendspin"

# Paths for current JSON config
CONFIG_DIR="$DAEMON_HOME/.config/sendspin"
CONFIG_FILE="$CONFIG_DIR/settings-daemon.json"

# Detect runtime dir/UID for audio device listing and service file
DAEMON_USER_UID=$(id -u "$DAEMON_USER")
DAEMON_RUNTIME_DIR="/run/user/$DAEMON_USER_UID"

if [ -f "$CONFIG_FILE" ]; then
    echo -e "\n${G}✓${N} Existing config detected at ${B}$CONFIG_FILE${N} — keeping it as-is."
else
    # No existing config — prompt and create one
    echo -e "\n${C}Configuration${N}"
    NAME=$(prompt_input "Client friendly name (shown in the UI)" "$(hostname)")

    echo -e "\n${C}Audio Devices${N}"
    # List audio devices - try to detect session environment for accuracy
    DAEMON_DBUS=""
    if [ -d "$DAEMON_RUNTIME_DIR" ]; then
        DAEMON_DBUS=$(ps -u "$DAEMON_USER" e | grep -m1 'DBUS_SESSION_BUS_ADDRESS=' | sed 's/.*DBUS_SESSION_BUS_ADDRESS=\([^ ]*\).*/\1/' || true)
        [ -z "$DAEMON_DBUS" ] && DAEMON_DBUS="unix:path=$DAEMON_RUNTIME_DIR/bus"
    fi

    if [ -n "$DAEMON_DBUS" ]; then
        sudo -u "$DAEMON_USER" env XDG_RUNTIME_DIR="$DAEMON_RUNTIME_DIR" DBUS_SESSION_BUS_ADDRESS="$DAEMON_DBUS" "$SENDSPIN_BIN" audio-devices list 2>&1 | grep -v -e "^To select an audio device" -e "^  sendspin"
    else
        sudo -u "$DAEMON_USER" "$SENDSPIN_BIN" audio-devices list 2>&1 | grep -v -e "^To select an audio device" -e "^  sendspin" || echo -e "${D}(Audio devices will be detected when daemon starts)${N}"
    fi

    DEVICE=$(prompt_input "Audio device" "default")
    [ "$DEVICE" = "default" ] && DEVICE=""

    # Create config directory and write only the prompted settings;
    # all other options are omitted so the daemon uses its built-in defaults.
    sudo -u "$DAEMON_USER" mkdir -p "$CONFIG_DIR"

    if [ -n "$DEVICE" ]; then
        sudo -u "$DAEMON_USER" tee "$CONFIG_FILE" > /dev/null << EOF
{
  "name": "$NAME",
  "audio_device": "$DEVICE"
}
EOF
    else
        sudo -u "$DAEMON_USER" tee "$CONFIG_FILE" > /dev/null << EOF
{
  "name": "$NAME"
}
EOF
    fi

    echo -e "${G}✓${N} Config written to $CONFIG_FILE"
fi

# Check if service is currently running (to determine if we need to restart)
SERVICE_WAS_RUNNING=false
if systemctl is-active --quiet sendspin.service 2>/dev/null; then
    SERVICE_WAS_RUNNING=true
    echo -e "\n${C}Service Update${N}"
    echo -e "${D}Service is currently running, stopping for update...${N}"
    systemctl stop sendspin.service
fi

# Install service
cat > /etc/systemd/system/sendspin.service << EOF
[Unit]
Description=Sendspin Multi-Room Audio Client
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=$DAEMON_USER
Environment=XDG_RUNTIME_DIR=/run/user/$DAEMON_USER_UID
ExecStart=$SENDSPIN_BIN daemon
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
SupplementaryGroups=audio

[Install]
WantedBy=multi-user.target
EOF

chmod 644 /etc/systemd/system/sendspin.service

# Reload systemd to pick up service changes
systemctl daemon-reload

# Enable and start/restart
echo -e "\n${C}Service Setup${N}"

# Check if service is enabled
SERVICE_ENABLED=false
if systemctl is-enabled --quiet sendspin.service 2>/dev/null; then
    SERVICE_ENABLED=true
fi

# Offer to enable on boot if not already enabled
if [ "$SERVICE_ENABLED" = false ]; then
    if prompt_yn "Enable on boot?"; then
        systemctl enable sendspin.service &>/dev/null
        echo -e "${D}Service enabled${N}"
    fi
else
    echo -e "${D}Service already enabled on boot${N}"
fi

# Start or restart the service
if [ "$SERVICE_WAS_RUNNING" = true ]; then
    echo -e "${D}Restarting service...${N}"
    systemctl restart sendspin.service
    echo -e "${G}✓${N} Service restarted"
else
    if prompt_yn "Start now?"; then
        systemctl start sendspin.service
        echo -e "${G}✓${N} Service started"
    fi
fi

# Summary
echo -e "\n${B}${G}Installation Complete!${N}\n"
echo -e "${C}Config:${N}  $CONFIG_FILE"
echo -e "${C}Service:${N} systemctl {start|stop|status} sendspin"
echo -e "${C}Logs:${N}    journalctl -u sendspin -f\n"
