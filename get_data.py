import os

os.system("wget https://dspace.mit.edu/bitstream/handle/1721.1/105492/fermi_data.tar.gz?sequence=5");
os.system("tar -xvf fermi_data.tar.gz?sequence=5");
os.system("rm -r fermi_data.tar.gz*");