# Primatrix
Run the following (in the virtual env):  
tar -xvzf micro-filtered.tar.gz  
tar -xvzf nano-filtered.tar.gz  
mv micro-filtered/ micro/  
mv nano-filtered/ nano/  
Remove most recent keras version and install keras  
sudo pip uninstall keras  
sudo pip install keras==2.0.8  
