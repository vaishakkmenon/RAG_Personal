#!/bin/bash

# VPS Setup Script (Contabo / Generic Linux)
# Installs Docker, Docker Compose, and sets up firewall basics (UFW) if present.
# Run this on your VPS as root or with sudo.

set -e

echo "🚀 Starting VPS Setup..."

# 1. Update System
echo "🔄 Updating system packages..."
if command -v apt-get >/dev/null; then
    apt-get update && apt-get upgrade -y
    apt-get install -y curl git ufw
elif command -v dnf >/dev/null; then
    dnf update -y
    dnf install -y curl git
    # Firewall usually Firewalld on RHEL/Oracle/CentOS
fi

# 2. Install Docker & Docker Compose
echo "🐳 Installing Docker..."
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    echo "✅ Docker installed."
else
    echo "✅ Docker already installed."
fi

# 3. Setup Firewall (UFW) - Standard for Ubuntu/Debian
if command -v ufw >/dev/null; then
    echo "🛡️ Configuring UFW Firewall..."
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow ssh
    ufw allow 80/tcp
    ufw allow 443/tcp
    ufw allow 443/udp  # HTTP/3

    # Enable if not enabled (non-interactive)
    # ufw --force enable
    echo "⚠️  UFW configured but NOT enabled to prevent locking you out via SSH."
    echo "    Run 'ufw enable' manually after confirming SSH access."
else
    echo "⚠️  UFW not found. If using CentOS/RHEL, ensure ports 80/443 are open in Firewalld."
fi

# 4. Create Workspace Directory
echo "📂 Creating workspace..."
mkdir -p /opt/rag-personal
chown $USER:$USER /opt/rag-personal 2>/dev/null || true

echo "✅ VPS Setup Complete!"
echo "   Next steps:"
echo "   1. Copy your project files to the VPS (e.g. /opt/rag-personal)"
echo "   2. Copy .env.prod to .env"
echo "   3. Run ./scripts/deploy.sh"
