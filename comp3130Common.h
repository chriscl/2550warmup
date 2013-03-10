/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    comp3130Common.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**
*****************************************************************************/
#include <cmath>
using namespace std;
using namespace Eigen;

// getSuperpixelFeatures -----------------------------------------------------
// This function extracts a feature vector for the superpixel with given id.

vector<double> getSuperpixelFeatures(const cv::Mat& img, const cv::Mat& seg, int id)
{
    // TODO: Define the number of features.
    //const unsigned numFeatures = 1;
    //vector<double> features(numFeatures, 0.0);

    const unsigned numFeatures = 41;
    vector<double> features(numFeatures, 0.0);


    // TODO: Compute some features that you think will be useful. See the
    // getSuperpixelLabel function for an example of iterating over all
    // pixels within a superpixel.
    features.resize(numFeatures ,0);
    int START,END;
    int numOfpixel = 0;
    Vec3b intensity;
    unsigned char blue;
    unsigned char green;
    unsigned char red;
    double bluetemp=0.0,greentemp=0.0,redtemp=0.0;
    double diffToNeighbour_RED,diffToNeighbour_BLUE,diffToNeighbour_GREEN,diffToNeighbour;
    double diffToNeighbour_abs;
    int counter = 0;
    int maxX = 0 ,minX = seg.cols;
    int maxY = 0, minY = seg.rows;

    /* Mean value of R G B */
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            features[0] += (int) red;
            features[1] += (int) green;
            features[2] += (int) blue;
            features[3] += y;
            features[4] += x;
            
            counter = 0;
            bluetemp = 0;
            redtemp = 0;
            greentemp = 0;

            if ( y > 0 ){
                intensity = img.at<Vec3b>(y-1,x);
                bluetemp += intensity.val[0];
                greentemp += intensity.val[1];
                redtemp += intensity.val[2];
                counter ++;
            }
            if (y < seg.rows - 1 ) {
            intensity = img.at<Vec3b>(y+1,x);
            bluetemp += intensity.val[0];
            greentemp += intensity.val[1];
            redtemp += intensity.val[2];
                counter++;
            }
            if (x > 0){
            intensity = img.at<Vec3b>(y,x-1);
            bluetemp += intensity.val[0];
            greentemp += intensity.val[1];
            redtemp += intensity.val[2];
            counter ++;
            }
            if(x < seg.cols - 1){
            intensity = img.at<Vec3b>(y,x+1);
            bluetemp += intensity.val[0];
            greentemp += intensity.val[1];
            redtemp += intensity.val[2];
            counter ++;
            }

            bluetemp /= counter;
            redtemp /= counter;
            greentemp /= counter; 

            diffToNeighbour_RED = abs(redtemp - (int)red);
            diffToNeighbour_GREEN = abs(greentemp - (int)green);
            diffToNeighbour_BLUE = abs(bluetemp - (int)blue);
            diffToNeighbour = (bluetemp + redtemp + greentemp) / 3.0 - ((int) (red + green + blue)) / 3.0;
            diffToNeighbour_abs = abs((bluetemp + redtemp + greentemp) / 3.0 - ((int) (red + green + blue)) / 3.0);
            features[5] += diffToNeighbour;
            features[6] += diffToNeighbour_abs;
            features[7] += diffToNeighbour_RED;
            features[8] += diffToNeighbour_GREEN;
            features[9] += diffToNeighbour_BLUE;
            //cout <<"("<<id << "; "<< x << "," << y << "; "<< (int)red << "," << (int) green <<"," << (int) blue << "; ";
            //cout << diffToNeighbour << ","<< diffToNeighbour_RED << "," << diffToNeighbour_GREEN << "," << diffToNeighbour_BLUE <<")\n";

            if (x > maxX) maxX = x;
            if (x < minX) minX = x;
            if (y > maxY) maxY = y;
            if (y < minY) minY = y;
         
            numOfpixel += 1;
        }
    }
    for ( int i = 0; i < 10; i++){
        features[i] /= (1.0 * numOfpixel);
    }

    /* Count for variance and standard deviation of R G B */
    START = 10;
    END = 12;
    numOfpixel = 0; 
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity += img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];        
            /* Marginal Standard deviation */
            features[START] += abs(((int) red) - features[0]);
            features[START+1] += abs(((int) green) - features[1]);
            features[START+2] += abs(((int) blue) - features[2]);
            
            numOfpixel += 1;
        }
    }
    for ( int i = START; i < END + 1; i++){
        features[i] /= (1.0 * numOfpixel); //work out average
    }

    /* Now process with black-white image
     *  TODO: raw black-white image, filtered black-white image
     * */
    START = 13;
    END = 40;
    double threshold = 0.1;
    double gray;
    double space = 0.05;
    int repetition = 14,time;
    double normX,normY;
    numOfpixel = 0;
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity += img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            gray = (blue + green + red) / 3.0;
            normX = (x-minX) ;
            normY = (y-minY) ;
            for (time = 0 ; time < repetition; time++){
                if (gray >= threshold + space*time) {
                    features[START+time] += 1*normX;
                    features[START+repetition+time] += 1*normY;
                } else {
                    features[START+time] += 0;
                    features[START+repetition+time] += 0;
                }
            }
            numOfpixel += 1;
        }
    }
    
    for (int i = START ; i < START + 2*repetition ; i ++){
        features[i] /= numOfpixel;
    }

    return features;
}

// getSuperpixelLabel --------------------------------------------------------
// This function returns the most frequently occurring pixel label within
// a superpixel. Negative labels are ignored.

int getSuperpixelLabel(const MatrixXi& labels, const cv::Mat& seg, int id)
{
    DRWN_ASSERT((labels.rows() == seg.rows) && (labels.cols() == seg.cols));

    vector<int> counts;
    for (int y = 0; y < seg.rows; y++) {
        for (int x = 0; x < seg.cols; x++) {
            // check correct superpixel id
            if (seg.at<int>(y, x) != id)
                continue;

            // skip negative labels
            if (labels(y, x) < 0)
                continue;

            // accumulate count
            if (counts.size() <= (unsigned)labels(y, x)) {
                counts.resize(labels(y, x) + 1, 0);
            }
            counts[labels(y, x)] += 1;
        }
    }

    // return "unknown" if no non-negative labels
    if (counts.empty())
        return -1;

    return drwn::argmax(counts);
}
