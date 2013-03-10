#! /bin/bash
# Simple shell script to train, test and score the comp3130 research project
# chriscl, ANU 2013

# Rebuild the program to make sure we are using the latest version
echo -e "\n \n Rebuilding... \n \n"
make clean
make all

# Train the program
echo -e "\n \n Training... \n \n"
../../bin/trainCOMP3130Model -verbose -o model.xml data/images/train/ data/labels/

# Test the program
echo -e "\n \n Testing... \n \n"

../../bin/testCOMP3130Model -verbose -x -o results model.xml data/images/test/

# Score the program
echo -e "\n \n Scoring... \n \n"
# Get today's date and time
thetime=`date +%Y-%m-%d--%H:%M:%S`
../../bin/scoreCOMP3130Results results/ data/labels/ > $thetime.results.txt
echo -e "\n \n Results matrix saved to "$thetime".results.txt"