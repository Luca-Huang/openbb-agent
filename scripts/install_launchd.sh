#!/usr/bin/env bash
# Install macOS launchd agents to (1) refresh watchlist_dashboard.html at 17:00
# and (2) open the HTML at 10:00 each day.
#
# Re-running this script overwrites prior installations — safe to use after
# moving the project, changing Python paths, or pulling updates.
#
# Usage:
#   ./scripts/install_launchd.sh          # install
#   ./scripts/install_launchd.sh uninstall  # remove
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHD_DIR="$HOME/Library/LaunchAgents"
PYTHON_BIN="${PYTHON_BIN:-/Library/Developer/CommandLineTools/usr/bin/python3}"
LONGBRIDGE_BIN_DIR="${LONGBRIDGE_BIN_DIR:-$HOME/.local/bin}"
LOG_DIR="$PROJECT_ROOT/outputs/.launchd_logs"

UPDATE_LABEL="com.user.watchlist-dashboard-update"
OPEN_LABEL="com.user.watchlist-dashboard-open"
UPDATE_PLIST="$LAUNCHD_DIR/$UPDATE_LABEL.plist"
OPEN_PLIST="$LAUNCHD_DIR/$OPEN_LABEL.plist"

uninstall() {
    local uid; uid="$(id -u)"
    for plist in "$UPDATE_PLIST" "$OPEN_PLIST"; do
        [ -f "$plist" ] || continue
        launchctl bootout "gui/$uid" "$plist" 2>/dev/null || true
        rm -f "$plist"
        echo "removed: $plist"
    done
}

install_jobs() {
    local uid; uid="$(id -u)"
    [ -x "$PYTHON_BIN" ] || { echo "ERROR: PYTHON_BIN not executable: $PYTHON_BIN" >&2; exit 1; }
    [ -x "$LONGBRIDGE_BIN_DIR/longbridge" ] || echo "WARN: longbridge CLI not found at $LONGBRIDGE_BIN_DIR/longbridge — update job may fail" >&2
    mkdir -p "$LAUNCHD_DIR" "$LOG_DIR"

    cat > "$UPDATE_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>                <string>$UPDATE_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>$PROJECT_ROOT/scripts/watchlist_dashboard.py</string>
    </array>
    <key>WorkingDirectory</key>     <string>$PROJECT_ROOT</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>             <integer>17</integer>
        <key>Minute</key>           <integer>0</integer>
    </dict>
    <key>RunAtLoad</key>            <false/>
    <key>StandardOutPath</key>      <string>$LOG_DIR/dashboard_update.out.log</string>
    <key>StandardErrorPath</key>    <string>$LOG_DIR/dashboard_update.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>             <string>$LONGBRIDGE_BIN_DIR:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
        <key>LANG</key>             <string>en_US.UTF-8</string>
    </dict>
</dict>
</plist>
EOF

    cat > "$OPEN_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>                <string>$OPEN_LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/open</string>
        <string>$PROJECT_ROOT/outputs/watchlist_dashboard.html</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>             <integer>10</integer>
        <key>Minute</key>           <integer>0</integer>
    </dict>
    <key>RunAtLoad</key>            <false/>
</dict>
</plist>
EOF

    plutil -lint "$UPDATE_PLIST" >/dev/null
    plutil -lint "$OPEN_PLIST" >/dev/null

    for plist in "$UPDATE_PLIST" "$OPEN_PLIST"; do
        launchctl bootout "gui/$uid" "$plist" 2>/dev/null || true
        launchctl bootstrap "gui/$uid" "$plist"
    done

    echo "installed:"
    echo "  $UPDATE_PLIST  (daily 17:00)"
    echo "  $OPEN_PLIST    (daily 10:00)"
    echo
    echo "test now:  launchctl kickstart -k gui/$uid/$UPDATE_LABEL"
    echo "logs:      $LOG_DIR/"
}

case "${1:-install}" in
    install)   install_jobs ;;
    uninstall) uninstall ;;
    *) echo "Usage: $0 [install|uninstall]" >&2; exit 1 ;;
esac
