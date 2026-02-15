#!/bin/bash

# Billy Server Security Setup Script
# Tailscale + UFW(Firewall) Configuration

echo "=================================================="
echo "🛡️  Configuring Firewall for Secure API Access"
echo "=================================================="

# Check if UFW is installed
if ! command -v ufw &> /dev/null; then
    echo "❌ UFW (Uncomplicated Firewall) could not be found."
    echo "   Please install it: sudo apt update && sudo apt install ufw"
    exit 1
fi

echo "1️⃣  Resetting UFW to defaults (Deny Incoming, Allow Outgoing)..."
# Note: These commands require sudo. We will echo them for the user to run or try running them.
# The user needs to approve the sudo commands.

# Check current user
if [ "$EUID" -ne 0 ]; then
  echo "⚠️  Please run this script with sudo (or run the commands manually):"
  echo "   sudo ./setup_security.sh"
  exit 1
fi

ufw default deny incoming
ufw default allow outgoing

echo "2️⃣  Allowing SSH (Port 22) to prevent lockout..."
ufw allow 22/tcp

echo "3️⃣  Allowing traffic on Tailscale interface (tailscale0)..."
# This allows ANY traffic coming from the Tailscale VPN
# This is secure because only authenticated Tailscale devices can enter this interface
ufw allow in on tailscale0

echo "4️⃣  (Validating) Denying external access to port 8000..."
# Explicitly ensuring public interface cannot access 8000 (though default deny handles it)
# This is just a safety measure fallback
ufw deny 8000/tcp

echo "5️⃣  Enabling Firewall..."
echo "Type 'y' if prompted to confirm:"
ufw enable

echo ""
echo "=================================================="
echo "✅ Security Configuration Complete!"
echo "--------------------------------------------------"
echo "   - SSH: Allowed (Everywhere)"
echo "   - API (Port 8000): Allowed ONLY via Tailscale"
echo "   - Internet: Blocked"
echo "=================================================="
echo "🔍 Current Status:"
ufw status verbose
