#The code was written using python3.6 


#You might need to install the folllowing libraries: 
--pytorch
--networkx
--numpy
--tqdm

#or you can have everything set up by running: 

pip install -r requirements.txt


#Note: The initiation of training and testing might take a while as the code creates a cache to store the spatio-temporal graphs.

#To train the models with the best configurations as mentioned in the paper just execute:
./train.sh 

#This will train a model for each data set.

#To test the models and obtain the average ADE/FDE run:
python3 test.py
#Running this code will generate the results reported in table#2 in the paper.
