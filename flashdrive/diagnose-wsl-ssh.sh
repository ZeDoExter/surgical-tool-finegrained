#!/bin/bash
# Run INSIDE WSL on the GPU machine:  bash diagnose-wsl-ssh.sh
echo "== 1) is sshd actually listening? =="
ss -tlnp 2>/dev/null | grep -E ':22\b' || echo "  NOTHING listening on :22"
echo
echo "== 2) try to start sshd manually (foreground errors shown) =="
/usr/sbin/sshd -t 2>&1 || true
echo
echo "== 3) config test result above. If empty = config OK =="
systemctl status ssh --no-pager 2>&1 | head -5
