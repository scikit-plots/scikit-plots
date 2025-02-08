#!/bin/bash
######################################################################
## Logging Functions
######################################################################
# Colors for Pretty Logs
# RESET="\033[0m"
# CYAN="\033[1;36m"
# GREEN="\033[1;32m"
# YELLOW="\033[1;33m"
# RED="\033[1;31m"
log()     { echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;36m[INFO]\033[0m $1"; }
success() { echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;32m[SUCCESS]\033[0m $1";}
warn()    { echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;33m[WARNING]\033[0m $1"; }
error()   {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') \033[1;31m[ERROR]\033[0m $1" >&2;
    exit 1;
}
