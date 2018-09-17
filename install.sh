wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create -n ml python=3
rm Miniconda3-latest-Linux-x86_64.sh
source activate ml
conda install -c anaconda tensorflow-gpu 
conda install -c pytorch pytorch torchvision


