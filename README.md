open cmd
python -m venv cactusDL
cactusDL\Scripts\activate.bat

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python
pip install ultralytics
pip install tensorflow
pip install h5py
pip install scikit-learn

python realtime.py
