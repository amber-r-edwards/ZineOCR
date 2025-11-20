#!/bin/bash
LOCAL_DIR="$HOME/desktop/HerstoryArchive_FeministNewspapers"
REMOTE_USER="are4"
REMOTE_HOST="slogin.palmetto.clemson.edu"
REMOTE_DIR="~/ZineOCR/pdf"
# Use -q flag with ssh to suppress the banner
rsync -avz --progress -e "ssh -q" \
  "$LOCAL_DIR" \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
echo "Transfer complete!"