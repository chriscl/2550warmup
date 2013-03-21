/*****************************************************************************
** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
** Distributed under the terms of the BSD license (see the LICENSE file)
** Copyright (c) 2007-2013, Stephen Gould
** All rights reserved.
**
******************************************************************************
** FILENAME:    comp3130Common.h
** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
**              Jimmy Lin <u5223173@uds.anu.edu.anu>
**              Chris Claoue-Long <u5183532@anu.edu.au>
*****************************************************************************/
#include <cmath>
#include <set>
using namespace std;
using namespace Eigen;

// feature extraction algorithms -----------------------------------------------

/* Calculate the Red/Green/Blue average luminance values in the superpixel as 
 * well as their standard deviation, adding them to the features vector starting
 * at the index value. 
 * REQUIRES 6 VECTOR INDEX POSITIONS */
vector<double> getRGBLuminance(vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
     double blue, green, red;
     Vec3b intensity;
     for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0]; // converts from int into doubles
            green = intensity.val[1];
            red = intensity.val[2];
            features[index] += red;
            features[index+1] += green;
            features[index+2] += blue;
        }
     }
     // Set the luminance values to the mean over the superpixel
     for (int i = index; i < index + 3; i ++) {
        features[i] /= (1.0 * pixels);
     }
	// Calculate the deviation of each pixel from the mean in the superpixel
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            features[index+3] += pow((red - features[index]),2.0); //((((int) red) - features[index]) * (((int) red) - features[index]));
            features[index+4] += pow((green - features[index+1]),2.0); //((((int) green) - features[index+1]) * (((int) green) - features[index+1]));
            features[index+5] += pow((blue - features[index+2]),2.0); //((((int) blue) - features[index+2]) * (((int) blue) - features[index+2]));
        }
    }
    // Convert to standard deviation
     for (int i = index + 3; i < index + 6; i ++) {
        features[i] = sqrt(features[i]/ (1.0 * pixels));
     }
     return features;
}

/* Calculate the average difference and average absolute difference between the
 * red, green and blue values in the superpixel, adding them to the features
 * vector starting at the index value.
 * REQUIRES 6 VECTOR INDEX POSITIONS */
vector<double> getRGBDiff(vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
     double blue, green, red;
     Vec3b intensity;
     for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            // absolute difference between superpixel colours
            features[index] += abs(red - green);
            features[index+1] += abs(green - blue);
            features[index+2] += abs(blue - red);
            // standard difference between superpixel colours
            features[index+3] += (red - green);
            features[index+4] += (green - blue);
            features[index+5] += (blue - red);
        }
     }
     // convert to average differences
     for (int i = index; i < index + 6; i ++) {
        features[i] /= (1.0 * pixels);
     }
     return features;
}

// Calculate the number of pixels in the given superpixel
int getPixelCount(const cv::Mat& seg, int id){
     int pixels = 0;
     for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            pixels ++;
        }
     }
	return pixels;
}

/* Calculate the average xy location (the centre) of the superpixel, adding it 
 * to the features vector at the index value as well as the standard deviation
 * within the superpixel
 * REQUIRES 4 VECTOR INDEX POSITIONS */
vector<double> getLocations (vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            features[index] += y;
            features[index+1] += x;
        }
     }
    // Convert to  average x and y positions (the centre)
    for (int i = index; i < index + 2; i ++) {
        features[i] /= (1.0 * pixels);
     }
    // Calculate the deviation of pixels from the average over the superpixel
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            features[index+2] += (y - features[index]) * (y - features[index]) ;
            features[index+3] += (x - features[index+1]) * (x - features[index+1]);
        }
     }
    // Convert to standard deviation
    for (int i = index + 2; i < index + 4; i ++) {
        features[i] = sqrt(features[i]/ (1.0 * pixels));
	}
	return features;
}

/* Calculate the average colour difference across the superpixel, somewhat
 * approximating the average gradient of the superpixel
 * REQUIRES 5 VECTOR INDEX POSITIONS */
vector<double> getColourDifference(vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
    Vec3b intensity;
    double blue, green, red, bluetemp=0.0, greentemp=0.0, redtemp=0.0;
    double diffPixelsRed,diffPixelsBlue,diffPixelsGreen;
    double diffPixelsAbs,diffPixels;
    int counter = 0;
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue += intensity.val[0];
            green += intensity.val[1];
            red += intensity.val[2];
            // gather values from neighbouring pixels
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

            diffPixelsRed = abs(redtemp - red) / 3.0;
            diffPixelsGreen = abs(greentemp - green) / 3.0;
            diffPixelsBlue = abs(bluetemp - blue) / 3.0;
            diffPixels = (diffPixelsRed + diffPixelsGreen + diffPixelsBlue);
            diffPixelsAbs = abs(diffPixels);
            features[index] += diffPixels;
            features[index+1] += diffPixelsAbs;
            features[index+2] += diffPixelsRed;
            features[index+3] += diffPixelsGreen;
            features[index+4] += diffPixelsBlue;
        }
    }
    // normalise the fffeatures
    for ( int i = index; i < index + 5; i++){
        features[i] /= (1.0 * pixels);
    }
    return features;
}

// Update a set with new detected neighbour if one is found 
set<int> updateNeighbours(int y, int x,set<int> neighbours, const cv::Mat& img, const cv::Mat& seg, int expected_id){
	if (!(x >= 0 && x < seg.cols && y >= 0 && y < seg.rows)){ 
        return neighbours; // x or y coordinate value out of range of the picture! Return input.
    } // else carry on
    int id = seg.at<int>(y,x);
    if (id != expected_id) {
        // found a distinct superpixel, could be a new neighbour
        set<int>::const_iterator got = neighbours.find(id);
        if (got == neighbours.end()){ // not found in set
            neighbours.insert(id);
        } 
    }
    return neighbours;
}

// Return a set of neighbouring superpixels
set<int> findNeighbours(const cv::Mat& img, const cv::Mat& seg, int id) {
    set<int> neighbours;
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            } 
            // Naively check around each pixel in the superpixel to see if there is a neighbouring superpixel nearby
            for (int i = -1; i < 2; i++){
            	for (int j = -1; j < 2; j++){
            		neighbours = updateNeighbours(y+j, x-i, neighbours, img, seg, id);
            	}
            }
        }
    }
    return neighbours;
}

/* Calculate the similarity between two feature vectors
 * NOTE: the vectors need to be >= to len in length, and there are inherent
 * errors in using the double type but this works faster than introducing a 
 * library that does infinite precision (very slow) */
double getSimilarity(vector<double> vec1, vector<double> vec2, int index1, int index2, int len){

    // calculate via cosine similarity, euclidian distance provides the inverse effect (smaller = closer instead of larger = closer)
    double dotProd = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0 ; i < len; i ++){
    	dotProd += vec1[i+index1] * vec2[i+index2];
    	norm1 += pow(vec1[i+index1],2.0);
    	norm2 += pow(vec2[i+index2],2.0);
    }
    return (dotProd/(sqrt(norm1)*sqrt(norm2)));
}

// Returns the optimum similar neighbour index (maximum/minimum similarity)
int findNotableNeighbour(set<int> neighbours, list<double> simList, bool findMaxSim){
    int index = 0;
    int optimum;
    int i = 0;
    if (findMaxSim) optimum = -1; // find maximum
    else optimum = 100; // find minimum

    for ( list<double>::iterator itor = simList.begin() ; itor != simList.end(); ++ itor, i++) {
        if (findMaxSim && *itor > optimum) {
            optimum = *itor;
            index = i;
        } else if (!findMaxSim && *itor < optimum) {
            optimum = *itor;
            index = i;
        }
        else continue;
    }
    
    i = 0;
    for (set<int>::iterator itor = neighbours.begin() ; itor != neighbours.end(); ++ itor, i++) {
        if (i == index) return *itor;
        else continue;
    }
// warning that it ends without output on a non-void function but we see no way of returning the correct result otherwise...
}

// feature selection based on neighbours --------------------------------
/* Defines a hashset with all neighbouring superpixels, finds the most and least
 * similar and adds their luminance and RGB difference values to the feature
 * vector 
 * REQUIRES 24 VECTOR INDEX VALUES*/
vector<double> checkNeighbours (vector<double> features,int index, const cv::Mat& img, const cv::Mat& seg, int id){ // 24 attributes
    set<int> neighbours;
    list<double> similarity;
    vector<double> temp( 6, 0.0); // This will always be smaller than features
    int leastsim_neighbour_id , mostsim_neighbour_id;
    int neighbourPixels;

    neighbours = findNeighbours( img, seg, id);
    
   	// Calculate the similarity of neighbours to the reference superpixel
    for (set<int>::iterator itor = neighbours.begin() ; itor != neighbours.end(); ++ itor) {
    	neighbourPixels = getPixelCount(seg, id);
        temp =  getRGBLuminance(temp, 0, neighbourPixels, img, seg, *itor);//getMarginalDistribution( temp, 0, neighbourPixels, img, seg, *itor);
        similarity.push_back(getSimilarity(features, temp, 0, 0, temp.size())); // ASSUMES THAT THE FEATURES IN TEMP ARE IN THE SIMILAR INDEX SPOT IN THE GIVEN FEATURE VECTOR.    
    }
    // Find the most and least similar neighbours and add in their features
    mostsim_neighbour_id = findNotableNeighbour(neighbours, similarity, true);
    leastsim_neighbour_id = findNotableNeighbour(neighbours, similarity, false);
 
    int pixelcount = getPixelCount(seg, mostsim_neighbour_id);
    features = getRGBLuminance( features, index, pixelcount, img, seg, mostsim_neighbour_id);
    features = getRGBDiff( features, index + 6, pixelcount, img, seg, mostsim_neighbour_id);

    pixelcount = getPixelCount(seg, leastsim_neighbour_id);
    features = getRGBLuminance( features, index + 12, pixelcount, img, seg, leastsim_neighbour_id);
    features = getRGBDiff( features, index + 18, pixelcount, img, seg, leastsim_neighbour_id);
    
    return features;
}

// getSuperpixelFeatures -----------------------------------------------------
// This function extracts a feature vector for the superpixel with given id.
vector<double> getSuperpixelFeatures(const cv::Mat& img, const cv::Mat& seg, int id){

    // Length of the feature vector and vector initialised to 0.0 for all values
    const unsigned numFeatures = 21; // using only colour and location data, remove unnecessary length
	vector<double> features(numFeatures, 0.0);
	int pixelcount = getPixelCount(seg, id); // pixelcount of the current superpixel

	/* FULLY FEATURED ALGORITHM
    features = getRGBLuminance(features, 0, pixelcount, img, seg, id);
    features = getRGBDiff(features, 6, pixelcount, img, seg, id); 
    features = getLocations(features, 12 , pixelcount, img, seg, id);
    features = getColourDifference(features, 16, pixelcount, img, seg, id);
    features = checkNeighbours(features, 21, img, seg, id);
    */
    
    /* COLOUR DATA ONLY
    features = getRGBLuminance(features, 0, pixelcount, img, seg, id);
    features = getRGBDiff(features, 6, pixelcount, img, seg, id); 
    features = getColourDifference(features, 12 , pixelcount, img, seg, id);
    */
    
    // COLOUR AND LOCATION DATA
    features = getRGBLuminance(features, 0, pixelcount, img, seg, id);
    features = getRGBDiff(features, 6, pixelcount, img, seg, id); 
    features = getColourDifference(features, 12, pixelcount, img, seg, id);
    features = getLocations(features, 17 , pixelcount, img, seg, id);

    
    return features;
} 

// -----------------------------------------------------------------------------
/* getSuperpixelLabel ----------------------------------------------------------
 * This function returns the most frequently occurring pixel label within
 * a superpixel. Negative labels are ignored. */
int getSuperpixelLabel(const MatrixXi& labels, const cv::Mat& seg, int id){
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
