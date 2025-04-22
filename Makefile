python: 
	- E:\MyStudy\DataHidingandApplications\deep-fake-detection-on-social-media\Python3106\python.exe --version
env:
	- venv/Script/activate
install-torch:
	- E:\MyStudy\DataHidingandApplications\deep-fake-detection-on-social-media\Python3106\python.exe -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
install:
	- E:\MyStudy\DataHidingandApplications\deep-fake-detection-on-social-media\Python3106\python.exe -m pip install -r requirements.txt
	
train:
	- E:\MyStudy\DataHidingandApplications\deep-fake-detection-on-social-media\Python3106\python.exe main.py