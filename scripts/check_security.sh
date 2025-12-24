#!/bin/bash
# Security status checker

echo "=== Firewall Status (UFW) ==="
if command -v ufw > /dev/null; then
    ufw status verbose
else
    echo "UFW not installed."
fi

echo -e "\n=== Fail2Ban Status ==="
if systemctl is-active --quiet fail2ban; then
    echo "Fail2Ban is RUNNING."
    fail2ban-client status sshd
else
    echo "Fail2Ban is NOT running."
fi

echo -e "\n=== SSH Configuration ==="
grep "^PermitRootLogin" /etc/ssh/sshd_config
grep "^PasswordAuthentication" /etc/ssh/sshd_config
