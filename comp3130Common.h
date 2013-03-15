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
#include <list>
using namespace std;
using namespace Eigen;

// feature extraction algorithms -----------------------------------------------

/* Calculate the Red/Green/Blue average luminance values in the superpixel as 
 * well as their standard deviation, adding them to the features vector starting
 * at the index value. 
 * REQUIRES 6 VECTOR VALUES */
void getRGBLuminance(vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
     unsigned char blue, green, red;
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
            features[index] += (int) red;
            features[index+1] += (int) green;
            features[index+2] += (int) blue;
        }
     }
     // Set the luminance values to the average over the superpixel
     for (int i = index; i < index + 3; i ++) {
        features[i] /= (1.0 * pixels);
     }
	// Calculate the average deviation
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            features[index+3] += ((int) red) - features[index]) * ((int) red) - features[index]);
            features[index+4] += ((int) green) - features[index+1]) * ((int) green) - features[index+1]);
            features[index+5] += ((int) blue) - features[index+2]) * ((int) blue) - features[index+2]);
        }
    }
    // Convert to standard deviation
     for (int i = index + 3; i < index + 6; i ++) {
        features[i] = sqrt(features[i]/ (1.0 * pixels));
     }
}

/* Calculates the average difference and average absolute difference between the
 * red, green and blue values in the superpixel, adding them to the features
 * vector starting at the index value.
 * REQUIRES 6 VECTOR VALUES */
void getRGBDiff(vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
     unsigned char blue, green, red;
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
            features[index] += abs((int) (red - green));
            features[index+1] += abs((int) (green - blue));
            features[index+2] += abs((int) (blue - red));
            features[index+3] += ((int) (red - green));
            features[index+4] += ((int) (green - blue));
            features[index+5] += ((int) (blue - red));
        }
     }
     for (int i = index; i < index + 6; i ++) {
        features[i] /= (1.0 * pixels);
     }
}

/*void getRGBDiffPlus(vector<double> features, int index, const cv::Mat& img, const cv::Mat& seg, int id){ // 6 attributes
     int numOfpixel = 0;
     unsigned char blue, green, red;
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
            features[index] += abs((int) (red - (green + blue) / 2 ));
            features[index+1] += abs((int) (green - (blue + red) / 2));
            features[index+2] += abs((int) (blue - (red + green)/2));
            features[index+3] += ((int) (red - (green + blue) / 2 ));
            features[index+4] += ((int) (green - (blue + red) / 2));
            features[index+5] += ((int) (blue - (red + green)/2));
            numOfpixel ++;
        }
     }
     for (int i = index; i < index + 6; i ++) {
        features[i] /= 1.0 * numOfpixel;
     }
}*/

/* Calculate the number of pixels in the superpixel and add it to the features
 * vector starting at the index value.
 * REQUIRES 1 VECTOR VALUE */
int getPixelCount(const cv::Mat& seg, int id){ // 6 attributes
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

/* Calculate the average xy location (the centre) of the superpixel, as well as 
 * the standard deviation of the pixels, adding them to the features vector 
 * starting at the index value.
 * REQUIRES 4 VECTOR VALUES */
void getLocations (vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            features[index] += y;
            features[index+1] += x;
        }
     }
    // Calculate the average x and y positions (the centre)
    for (int i = index; i < index + 2; i ++) {
        features[i] /= (1.0 * pixels);
     }
    // Calculate the average deviation
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
}

/*void getMarginalDistribution(vector<double> features, int index, const cv::Mat& img, const cv::Mat& seg, int id){
    int numOfpixel;
    unsigned char blue, green, red;
    Vec3b intensity;
    numOfpixel = 0;
    int step = 9;
    int spaceClass = 255 / step + 1;
    int cblue,cgreen,cred;
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            cblue = ((int)blue) / step;
            cgreen = ((int)green) / step;
            cred = ((int)red) / step;
            //cout << "\n(" << (int)blue << "->" << cblue << ", " << (int)green << "->" << cgreen << ", " << (int)red << "->" << cred << ")\n";
            // Category assignment
            features[index+cred] += 1.0 ;
            features[index+spaceClass+cgreen] += 1.0 ;
            features[index+2*spaceClass+cblue] += 1.0 ;
            numOfpixel += 1;
        }
    }
    for (int i = index ; i < index + 3 * spaceClass; i ++){
        features[i] /= 1.0 * numOfpixel;
    }
}*/

/*vector<double> getMarginalDistributionWithLocation(vector<double> features, int index, const cv::Mat& img, const cv::Mat& seg, int id){
    int numOfpixel;
    unsigned char blue, green, red;
    Vec3b intensity;
    numOfpixel = 0;
    int step = 9;
    int spaceClass = 255 / step + 1;
    int cblue,cgreen,cred;

    vector<double> temp(4,0.0);
    temp = getLocations( temp, 0, img, seg, id);

    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            cblue = ((int)blue) / step;
            cgreen = ((int)green) / step;
            cred = ((int)red) / step;
            //cout << "\n(" << (int)blue << "->" << cblue << ", " << (int)green << "->" << cgreen << ", " << (int)red << "->" << cred << ")\n";
            // Category assignment 
            features[index+cred] += 1.0 * ( (x-temp[1])*(x-temp[1])  + (y-temp[0])* (y-temp[0])  ) ;
            features[index+spaceClass+cgreen] += 1.0 * ( (x-temp[1])*(x-temp[1])  + (y-temp[0])* (y-temp[0])  ) ;
            features[index+2*spaceClass+cblue] += 1.0 *  ( (x-temp[1])*(x-temp[1])  + (y-temp[0])* (y-temp[0])  );
            numOfpixel += 1;
        }
    }
    for (int i = index ; i < index + 3 * spaceClass; i ++){
        features[i] /= 1.0 * numOfpixel;
    }
}*/

/* Calculate the diff
 */
void getSmoothness(vector<double> features, int index, int pixels, const cv::Mat& img, const cv::Mat& seg, int id){
    Vec3b intensity;
    unsigned char blue;
    unsigned char green;
    unsigned char red;
    double bluetemp=0.0,greentemp=0.0,redtemp=0.0;
    double diffToNeighbour_RED,diffToNeighbour_BLUE,diffToNeighbour_GREEN;
    double diffToNeighbour_abs,diffToNeighbour;
    int counter = 0;
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            }

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
            features[index] += diffToNeighbour;
            features[index+1] += diffToNeighbour_abs;
            features[index+2] += diffToNeighbour_RED;
            features[index+3] += diffToNeighbour_GREEN;
            features[index+4] += diffToNeighbour_BLUE;
        }
    }
    for ( int i = index; i < index + 5; i++){
        features[i] /= (1.0 * pixels);
    }
}

list<int> detectNeighbours(int y, int x, list<int> neighbours, const cv::Mat& img, const cv::Mat& seg, int expected_id){
/*{{{*/
    if (x > 0 && x < seg.cols && y > 0 && y < seg.rows) {
        ;
    } else {
        return neighbours;
    }

    int id = seg.at<int>(y,x);
    if (id != expected_id) {
        /* a distinct superpixel */
        for (list<int>::iterator itor = neighbours.begin(); itor != neighbours.end(); ++itor){
            if (id == *itor) {
                // neigbhours pre-existing in list
                return neighbours; 
            }
            else continue;
        }
        // superpixel not pre-existing in list, so add it to list
        neighbours.push_back (id);
        return neighbours;
    }
    else return neighbours;
/*}}}*/
}

list<int> getNeighbours(const cv::Mat& img, const cv::Mat& seg, int id) {
/*{{{*/
    list<int> neighbours;
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (seg.at<int>(y,x) != id){
                continue;
            } 
            neighbours = detectNeighbours( y-1, x-1, neighbours, img, seg, id);
            neighbours = detectNeighbours( y-1, x, neighbours, img, seg, id);
            neighbours = detectNeighbours( y-1, x+1, neighbours, img, seg, id);
            neighbours = detectNeighbours( y, x-1, neighbours, img, seg, id);
            neighbours = detectNeighbours( y, x, neighbours, img, seg, id);
            neighbours = detectNeighbours( y, x+1, neighbours, img, seg, id);
            neighbours = detectNeighbours( y+1, x-1, neighbours, img, seg, id);
            neighbours = detectNeighbours( y+1, x, neighbours, img, seg, id);
            neighbours = detectNeighbours( y+1, x+1, neighbours, img, seg, id);
        }
    }
    if (0) {
        cout << "(" << id << ":"<< neighbours.size() <<":";
        for (list<int>::iterator itor = neighbours.begin(); itor != neighbours.end(); ++itor){
            cout << *itor << ",";
        }
        cout  << ")\n";
    }
/*}}}*/
    return neighbours;
}

double getSimilarity(vector<double> vec1, vector<double> vec2, int index1, int index2, int len){
    double sim = 0.0;
    for (int i = 0 ; i < len; i ++){
        sim += vec1[i+index1] * vec2[i+index2];
    }

    return sim;
}

int findSimilarNeighbour(list<int> neighbours, list<double> simList, bool maximum){
/*{{{*/
    int index = 0;
    int optimum;
    int i;
    if (maximum) optimum = -1; // find maximum
    else optimum = 100; // find minimum

    i = 0;
    for ( list<double>::iterator itor = simList.begin() ; itor != simList.end(); ++ itor, i++) {
        if (maximum && *itor > optimum) {
            optimum = *itor;
            index = i;
        } else if (!maximum && *itor < optimum) {
            optimum = *itor;
            index = i;
        }
        else continue;
    }
    
    i = 0;
    for (list<int>::iterator itor = neighbours.begin() ; itor != neighbours.end(); ++ itor, i++) {
        if (i == index) return *itor;
        else continue;
    }
/*}}}*/
}

// feature selection based on neighbours --------------------------------
void NeighbourScheme1 (vector<double> features,int index, const cv::Mat& img, const cv::Mat& seg, int id){ // 24 attributes
/*{{{*/
    list<int> neighbours;
    list<double> similarity;
    vector<double> temp( 87, 0.0);
    int leastsim_neighbour_id , mostsim_neighbour_id;

    neighbours = getNeighbours( img, seg, id);
    for (list<int>::iterator itor = neighbours.begin() ; itor != neighbours.end(); ++ itor) {
        temp = getMarginalDistribution( temp, 0, img, seg, *itor);
        similarity.push_back(getSimilarity(features, temp, 0, 0, temp.size()));
    }
    mostsim_neighbour_id = findSimilarNeighbour(neighbours, similarity, true);
    leastsim_neighbour_id = findSimilarNeighbour(neighbours, similarity, false);
    // cout << id << ":"<<mostsim_neighbour_id << ":" << leastsim_neighbour_id << "\n";
 
    features = getRGBLightness( features, index, img, seg, mostsim_neighbour_id);
    features = getRGBDiff( features, index + 6, img, seg, mostsim_neighbour_id);
    features = getRGBLightness( features, index + 12, img, seg, leastsim_neighbour_id);
    features = getRGBDiff( features, index + 18, img, seg, leastsim_neighbour_id);
    //features = getMarginalDistribution( features, index , img, seg, leastsim_neighbour_id);
/*}}}*/
    return features;
}

bool isInList (int id, list<int> L){
    for (list<int>::iterator itor = L.begin() ; itor != L.end(); ++itor) {
        if (id == *itor) return true;
        else continue;
    }
    return false;
}

/*void getNeighboursMarginalDistribution(vector<double> features, list<int> neighbours, int index, const cv::Mat& img, const cv::Mat& seg){
    int numOfpixel;
    unsigned char blue, green, red;
    Vec3b intensity;
    numOfpixel = 0;
    int step = 9;
    int spaceClass = 255 / step + 1;
    int cblue,cgreen,cred;
    for (int y = 0; y < seg.rows; y++){
        for (int x = 0; x < seg.cols; x++){
            if (!isInList(seg.at<int>(y,x) , neighbours)){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            cblue = ((int)blue) / step;
            cgreen = ((int)green) / step;
            cred = ((int)red) / step;
            //cout << "\n(" << (int)blue << "->" << cblue << ", " << (int)green << "->" << cgreen << ", " << (int)red << "->" << cred << ")\n";
            // Category assignment
            features[index+cred] += 1.0 ;
            features[index+spaceClass+cgreen] += 1.0 ;
            features[index+2*spaceClass+cblue] += 1.0 ;
            numOfpixel += 1;
        }
    }
    for (int i = index ; i < index + 3 * spaceClass; i ++){
        features[i] /= 1.0 * numOfpixel;
    }
}*/

/*vector<double> NeighbourScheme2 (vector<double> features,int index, const cv::Mat& img, const cv::Mat& seg, int id){
    list<int> neighbours;
    neighbours = getNeighbours( img, seg, id);
    features = getNeighboursMarginalDistribution (features,neighbours, index, img, seg);

    return features;
}*/

/*void NeighbourScheme3 (vector<double> features, int index,int meanY, int meanX, const cv::Mat& img, const cv::Mat& seg, int id) {
     // Parameters.
    int sideLength = 70; // unit:pixels

    // -----------------------------------
     int numOfpixel = 0;
     unsigned char blue, green, red;
     Vec3b intensity;
     
     int left, right, bottom, top;
    if ( meanX - sideLength < 0) left = 0;
    if ( meanX + sideLength > seg.cols - 1) right = seg.cols - 1;
    if ( meanY - sideLength < 0) bottom = 0;
    if ( meanY + sideLength > seg.rows - 1) top = seg.rows - 1;

     for (int y = bottom; y < top; y++){
        for (int x = left; x < right; x++){
            // Here we ignore the pixel in the same super pixel
            if (seg.at<int>(y,x) == id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            features[index] += (int) red;
            features[index+1] += (int) green;
            features[index+2] += (int) blue;
            features[index+3] += abs((int) (red - green));
            features[index+4] += abs((int) (green - blue));
            features[index+5] += abs((int) (blue - red));
            features[index+6] += ((int) (red - green));
            features[index+7] += ((int) (green - blue));
            features[index+8] += ((int) (blue - red));
            numOfpixel ++;
        }
     }
     for (int i = index; i < index + 9; i ++) {
        features[i] /= 1.0 * numOfpixel;
     }
     numOfpixel = 0;
     for (int y = bottom; y < top; y++){
        for (int x = left; x < right; x++){
            // Here we ignore the pixel in the same super pixel
            if (seg.at<int>(y,x) == id){
                continue;
            }
            intensity = img.at<Vec3b>(y,x);
            blue = intensity.val[0];
            green = intensity.val[1];
            red = intensity.val[2];
            features[index+9] += abs(((int) red) - features[0]) * abs(((int) red) - features[0]);
            features[index+10] += abs(((int) green) - features[1]) * abs(((int) green) - features[1]);
            features[index+11] += abs(((int) blue) - features[2]) * abs(((int) blue) - features[2]);
            numOfpixel ++;
        }
    }
     for (int i = index + 9; i < index + 12; i ++) {
        features[i] /= 1.0 * numOfpixel;
     }
}*/

// getSuperpixelFeatures -----------------------------------------------------
// This function extracts a feature vector for the superpixel with given id.
vector<double> getSuperpixelFeatures(const cv::Mat& img, const cv::Mat& seg, int id){

    // Length of the feature vector and vector initialised to 0.0 for all values
    const unsigned numFeatures = 63;
	vector<double> features(numFeatures, 0.0);
	int pixelCount = getPixelCount(seg, id);

	/*/ feature: 16+87  A: 0.433891
    if (false){ 
        features = getRGBLightness( features, 0, img, seg, id);
        features = getMarginalDistribution( features, 6, img, seg, id);
        features = getLocations( features, 6 + 87, img, seg, id); 
        features = getSmoothness(features, 6 + 87 + 4, img, seg, id);
    }

	// #feature: 14  A: 0.346711
    if (false){ 
        features = getRGBLightness( features, 0, img, seg, id);
        features = getLocations( features, 6 , img, seg, id);
        features = getSmoothness(features, 6 + 4, img, seg, id);
    }
    
    // #feature: 17  A: 0.460461
    if (false){
        // RGB's difference to each other substantially increases the accuracy of decision tree
        features = getRGBLightness( features, 0, img, seg, id);
        features = getRGBDiff( features, 6, img, seg, id); // here has three attributes with abs only
        features = getLocations( features, 6 + 3 , img, seg, id);
        features = getSmoothness( features, 6 + 3 + 4, img, seg, id);
    }

    // #feature: 21  A: 0.512019
    if (false){
        features = getRGBLightness( features, 0, img, seg, id);
        features = getRGBDiff( features, 6, img, seg, id); // here has six attributes with abs and original values
        features = getLocations( features, 6 + 6 , img, seg, id);
        features = getSmoothness( features, 6 + 6 + 4, img, seg, id);
    }

    // #feature: 22   A: 0.511862
    if (false){
        // number of pixel weakens the outcome of classifier
        features = getRGBLightness( features, 0, img, seg, id);
        features = getRGBDiff( features, 6, img, seg, id); 
        features = getLocations( features, 6 + 6 , img, seg, id);
        features = getSmoothness( features, 6 + 6 + 4, img, seg, id);
        features = getNumOfPixel( features, 6 + 6 + 4 + 5, img, seg, id);
    }

    // #feature: 27  A: 0.51378
    if (false){
        features = getRGBLightness( features, 0, img, seg, id);
        features = getRGBDiff( features, 6, img, seg, id); // here has six attributes with abs and original values
        features = getLocations( features, 6 + 6 , img, seg, id);
        features = getSmoothness( features, 6 + 6 + 4, img, seg, id);
        features = getRGBDiffPlus( features, 21, img, seg, id);
    }

    // #feature: 33  A: 0.517115
    if (false){ 
        features = getRGBLightness( features, 0, img, seg, id);
        features = getRGBDiff( features, 6, img, seg, id); 
        features = getLocations( features, 6 + 6 , img, seg, id);
        features = getSmoothness( features, 6 + 6 + 4, img, seg, id);
        features = NeighbourScheme3( features, 6 + 6 + 4 + 5, features[12],features[13], img, seg, id);
    }*/

	// #features: 45  A: 0.519499
    getRGBLuminance(features, 0, pixelcount, img, seg, id);
    getRGBDiff(features, 6, pixelcount, img, seg, id); 
    getLocations(features, 12 , pixelcount, img, seg, id);
    getSmoothness(features, 16, pixelcount, img, seg, id);
    NeighbourScheme1(features, 21, img, seg, id);

    /*// #features: 63  A: 0.511859
    if (false){
		features = getRGBLightness( features, 0, img, seg, id);
    	features = getRGBDiff( features, 6, img, seg, id); 
    	features = getLocations( features, 6 + 6 , img, seg, id);
    	features = getSmoothness( features, 6 + 6 + 4, img, seg, id);
    	features = getRGBDiffPlus( features, 21, img, seg, id);
    	features = NeighbourScheme1( features, 27 , img, seg, id);
    	features = NeighbourScheme3( features, 27 + 24, features[12],features[13], img, seg, id);
	}

/* Output all features to screen (this doesn't print out what the features are!)
        cout << "(";
        for (int i = 0; i < numFeatures ; i ++){
            cout << features[i] << "," ;
        }
        cout << ")\n"; */
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
