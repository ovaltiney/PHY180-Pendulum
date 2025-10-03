# Switch to bash shell
chsh -s /bin/bash

# Verify BASH shell
echo $SHELL

# Create Virtual Environment for bash (If not created)
python3 -m venv venv

# Activate Virutal Environment for bash
source venv/bin/activate

# Install Python Libraries
pip install -r requirements.txt

# Run Tracking Object App
python trackingVideo.py

# Run StreamLit App
streamlit run streamlitVideo.py