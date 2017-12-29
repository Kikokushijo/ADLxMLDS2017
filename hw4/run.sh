rm -rf generator.hdf5
wget https://gitlab.com/Kikokushijo/ADLxMLDS_models/raw/master/generator.hdf5 -P ./
python3 generate.py $1