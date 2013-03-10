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

using namespace std;
using namespace Eigen;

// getSuperpixelFeatures -----------------------------------------------------
// This function extracts a feature vector for the superpixel with given id.

vector<double> getSuperpixelFeatures(const cv::Mat& img, const cv::Mat& seg, int id)
{
    // TODO: Define the number of features.
    const unsigned numFeatures = 1;
    vector<double> features(numFeatures, 0.0);

    // TODO: Compute some features that you think will be useful. See the
    // getSuperpixelLabel function for an example of iterating over all
    // pixels within a superpixel.

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
