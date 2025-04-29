#!/bin/bash
######################################################################
## Logging Functions
######################################################################
# ANSI color codes for terminal output:
# \033[0m       - Reset / Normal
# \033[1m       - Bold
# \033[2m       - Dim
# \033[4m       - Underline

# Foreground colors:
# \033[30m      - Black
# \033[31m      - Red
# \033[32m      - Green
# \033[33m      - Yellow
# \033[34m      - Blue
# \033[35m      - Magenta
# \033[36m      - Cyan
# \033[37m      - White

# Bright foreground colors:
# \033[1;30m    - Bright Black (Gray)
# \033[1;31m    - Bright Red
# \033[1;32m    - Bright Green
# \033[1;33m    - Bright Yellow
# \033[1;34m    - Bright Blue
# \033[1;35m    - Bright Magenta
# \033[1;36m    - Bright Cyan
# \033[1;37m    - Bright White

# Background colors (optional, less common):
# \033[40m      - Background Black
# \033[41m      - Background Red
# \033[42m      - Background Green
# \033[43m      - Background Yellow
# \033[44m      - Background Blue
# \033[45m      - Background Magenta
# \033[46m      - Background Cyan
# \033[47m      - Background White

# Colors for Pretty Logs
# BOLD='\033[1m'
# RESET='\033[0m'
# RED='\033[1;31m'
# GREEN='\033[1;32m'
# BLUE='\033[1;34m'
# YELLOW='\033[1;33m'
# MAGENTA='\033[1;35m'
# CYAN='\033[1;36m'
log()     { echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;36m[INFO]\033[0m $1"; }
success() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;32m[SUCCESS]\033[0m $1";}
warn()    { echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;33m[WARNING]\033[0m $1"; }
error()   {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;31m[ERROR]\033[0m $1" >&2;
    exit 1;
}
