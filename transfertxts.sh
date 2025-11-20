#!/bin/bash
# filepath: /Users/amberedwards/Library/CloudStorage/OneDrive-ClemsonUniversity/Research/ZineOCR/pulltext.sh
LOCAL_DIR="$HOME/desktop/HerstoryArchiveTxt"
REMOTE_USER="are4"
REMOTE_HOST="slogin.palmetto.clemson.edu"
REMOTE_DIR="~/ZineOCR/HerstoryArchiveTxt"

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Use -q flag with ssh to suppress the banner
rsync -avz --progress -e "ssh -q" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/" \
  "$LOCAL_DIR"

echo "Transfer complete!"