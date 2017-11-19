wget https://gitlab.com/Kikokushijo/ADLxMLDS_models/raw/master/1110_random2.h5 -P ./models/
python hw2_test.py $1 $2
python hw2_peerview.py $1 $3
