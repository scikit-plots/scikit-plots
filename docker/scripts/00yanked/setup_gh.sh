# 1. Make sure curl is installed
sudo apt update
sudo apt install curl -y

# 2. Add the GitHub CLI GPG key
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg

sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg

# 3. Add the GitHub CLI repository
echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] \
  https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null

# 4. Install GitHub CLI
sudo apt update
sudo apt install gh -y

# 5. Authenticate with GitHub CLI .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  gh auth login --with-token <<EOF
$GITHUB_TOKEN
EOF
else
  echo "No .env file found. Please create one with GITHUB_TOKEN."
  exit 0
fi

# 6. Verify installation
gh --version

if [ $? -eq 0 ]; then
  echo "GitHub CLI installed and authenticated successfully."
else
  echo "Failed to install or authenticate GitHub CLI."
  exit 0
fi

# 7. Clean up
# sudo rm /usr/share/keyrings/githubcli-archive-keyring.gpg
# sudo rm /etc/apt/sources.list.d/github-cli.list

sudo apt update
sudo apt autoremove -y
echo "Cleanup completed."

# 8. Final message
echo "GitHub CLI setup completed successfully."
echo "You can now use the GitHub CLI with your authenticated account."
echo "Run 'gh auth status' to check your authentication status."
echo "For more information, visit: https://cli.github.com/manual/"
echo "Thank you for using the GitHub CLI setup script!"
