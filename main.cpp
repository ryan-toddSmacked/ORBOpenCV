
// Opencv includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

// Boost
#include <boost/program_options.hpp>

// C++ includes
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

struct OrbInput
{
    std::string image1;
    std::string image2;
    std::string homography;
};

// Orb parameters
static int nfeatures;
static float scaleFactor;
static int nlevels;
static int edgeThreshold;
static int firstLevel;
static int WTA_K;
static cv::ORB::ScoreType scoreType;
static int patchSize;
static int fastThreshold;

// Split a string into a vector of strings
std::vector<std::string> string_split(const std::string& str, const std::string& delim);

// ORB function
void orb(const OrbInput& input);

// imadjust function
void imadjust(cv::Mat& im1, cv::Mat& im2);

void loadProcesses(std::vector<OrbInput>& inputs, const std::string& input_file);

int main(int argc, char** argv)
{
    // Usage ./ORBDetector [REQUIRED] -i <input_file> [OPTIONAL] --nfeatures <number_of_features> --scaleFactor <scale_factor> --nlevels <number_of_levels> --edgeThreshold <edge_threshold> --firstLevel <first_level> --WTA_K <WTA_K> --scoreType <score_type> --patchSize <patch_size> --fastThreshold <fast_threshold>
    // The only required argument is the input file
    // The rest of the arguments are optional and have default values
    // The default values are the same as the default values for the ORB detector in OpenCV

    // Create a program options object
    boost::program_options::options_description desc("Allowed options");

    // Add the options
    desc.add_options()
        ("help", "Produce help message")
        ("inputFile", boost::program_options::value<std::string>(), "Input file")
        ("nthreads", boost::program_options::value<int>()->default_value(1), "Number of threads")
        ("nfeatures", boost::program_options::value<int>()->default_value(500), "Number of features")
        ("scaleFactor", boost::program_options::value<float>()->default_value(1.2f), "Scale factor")
        ("nlevels", boost::program_options::value<int>()->default_value(8), "Number of levels")
        ("edgeThreshold", boost::program_options::value<int>()->default_value(31), "Edge threshold")
        ("firstLevel", boost::program_options::value<int>()->default_value(0), "First level")
        ("WTA_K", boost::program_options::value<int>()->default_value(2), "WTA_K, 2 or 3")
        ("scoreType", boost::program_options::value<int>()->default_value(cv::ORB::HARRIS_SCORE), "Score type, 0 = HARRIS_SCORE, 1 = FAST_SCORE")
        ("patchSize", boost::program_options::value<int>()->default_value(31), "Patch size")
        ("fastThreshold", boost::program_options::value<int>()->default_value(20), "Fast threshold");

    // Create a variables map
    boost::program_options::variables_map vm;

    // Parse the command line arguments
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

    // Notify the variables map
    boost::program_options::notify(vm);

    // Check if the help option was given
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        // Describe what should be in the input file
        // The input file needs to be a text file with the following format
        // image1.jpg, image2.jpg, image1_warped.jpg, image2_warped.jpg (optional) homograpy.txt
        std::cout << "The input file needs to be a text file with the following format\n";
        std::cout << "image1.jpg, image2.jpg, homograpy.txt\n";
        std::cout << "The file can contain as many lines as you want, they all\n";
        std::cout << "must contain 3 values that are comma seperated\n\n";
        return 1;
    }

    // Check if the input file was given
    if (vm.count("inputFile") != 1)
    {
        std::cout << desc << "\n";
        std::cout << "There must be exactly one provided input file\n";
        return 1;
    }

    // Get the input file
    std::string inputFile = vm["inputFile"].as<std::string>();

    // Open the input file
    std::ifstream ifs(inputFile);

    // Check if the file was opened
    if (!ifs.is_open())
    {
        std::cout << "Could not open the input file\n";
        return 1;
    }

    // Set the static variables
    nfeatures = vm["nfeatures"].as<int>();
    scaleFactor = vm["scaleFactor"].as<float>();
    nlevels = vm["nlevels"].as<int>();
    edgeThreshold = vm["edgeThreshold"].as<int>();
    firstLevel = vm["firstLevel"].as<int>();
    WTA_K = vm["WTA_K"].as<int>();
    scoreType = (cv::ORB::ScoreType)vm["scoreType"].as<int>();
    patchSize = vm["patchSize"].as<int>();
    fastThreshold = vm["fastThreshold"].as<int>();
    int nthreads = vm["nthreads"].as<int>();

    std::vector<OrbInput> inputs;

    loadProcesses(inputs, inputFile);

    if (nthreads > 1)
    {
        size_t threads_pushed_back = 0;
        std::vector<std::thread> threads(nthreads);
        size_t num_inputs = inputs.size();

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < num_inputs; i++)
        {
            if (threads_pushed_back < (size_t)nthreads)
            {
                threads[threads_pushed_back] = std::thread(orb, inputs[i]);
                threads_pushed_back++;
            }
            else
            {
                for (size_t j = 0; j < threads_pushed_back; j++)
                {
                    threads[j].join();
                }

                threads_pushed_back = 0;
                i--;
            }
        }

        for (size_t j = 0; j < threads_pushed_back; j++)
        {
            threads[j].join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
        std::cout << "Average time: " << elapsed.count() / (double)num_inputs << " seconds\n";
    }
    else
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < inputs.size(); i++)
            orb(inputs[i]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds\n";
        std::cout << "Average time: " << elapsed.count() / (double)inputs.size() << " seconds\n";
    }

    return 0;
}


void loadProcesses(std::vector<OrbInput>& inputs, const std::string& input_file)
{
    OrbInput input;

    // Open the input file
    std::ifstream ifs(input_file);

    if (!ifs.is_open())
    {
        std::cout << "Could not open the input file\n";
        return;
    }

    // Create a string to hold each line
    std::string line;

    // Create a vector of strings to hold the split line
    std::vector<std::string> split_line;

    // Read the file line by line
    while (std::getline(ifs, line))
    {
        // Split the line
        split_line = string_split(line, ",");

        // Check if the line has at least 4 values
        if (split_line.size() != 3)
        {
            continue;
        }

        // Set the input values
        input.image1 = split_line[0];
        input.image2 = split_line[1];
        input.homography = split_line[2];

        // Add the input to the vector
        inputs.push_back(input);
    }
}

std::vector<std::string> string_split(const std::string& str, const std::string& delim)
{
    // Create a vector of strings
    std::vector<std::string> tokens;

    size_t pos = 0;
    size_t prev = 0;
    std::string token;

    // Loop through the string
    while ((pos = str.find(delim, prev)) != std::string::npos)
    {
        token = str.substr(prev, pos - prev);
        // Remove leading and trailing whitespace
        token.erase(0, token.find_first_not_of(' '));
        token.erase(token.find_last_not_of(' ') + 1);
        tokens.push_back(token);

        prev = pos + delim.length();
    }

    // If the prev is not the end of the string
    // then add the last token
    if (prev < str.length())
    {
        token = str.substr(prev, std::string::npos);
        // Remove leading and trailing whitespace
        token.erase(0, token.find_first_not_of(' '));
        token.erase(token.find_last_not_of(' ') + 1);
        tokens.push_back(token);
    }

    return tokens;
}

void orb(const OrbInput& input)
{
    // Read the images as 32 bit floats and single channel
    cv::Mat image1 = cv::imread(input.image1, cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(input.image2, cv::IMREAD_GRAYSCALE);

    // Check if the images were read
    if (image1.empty() || image2.empty())
    {
        std::cout << "Could not read the images\n";
        return;
    }

    // Adjust the images
    imadjust(image1, image2);

    // Create the ORB detector
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);

    //cv::Ptr<cv::AKAZE> orb = cv::AKAZE::create();

    //cv::Ptr<cv::SIFT> orb = cv::SIFT::create();

    // Create the keypoint vectors
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    keypoints1.reserve(nfeatures);
    keypoints2.reserve(nfeatures);

    // Create the descriptors
    cv::Mat descriptors1, descriptors2;
    descriptors1.reserve(nfeatures);
    descriptors2.reserve(nfeatures);

    // Create the matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Calculate the keypoints and descriptors
    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // Match the descriptors
    std::vector<cv::DMatch> matches;
    matches.reserve(nfeatures);
    matcher.match(descriptors1, descriptors2, matches);

    // std::sort
    std::sort(matches.begin(), matches.end());

    // Extract the top 20% of the matches
    const size_t best_20_cutoff = (size_t)((float)matches.size() * 0.2f);
    std::vector<cv::Point2f> src_pts(best_20_cutoff), dst_pts(best_20_cutoff);

    for (size_t i = 0; i < best_20_cutoff; ++i)
    {
        src_pts[i] = keypoints1[matches[i].queryIdx].pt;
        dst_pts[i] = keypoints2[matches[i].trainIdx].pt;
    }

    // Calculate the homography
    const cv::Mat H = cv::findHomography(src_pts, dst_pts, cv::RANSAC);

    // Check if the homography file was given
    if (!input.homography.empty())
    {
        // Write the homography to a file
        std::ofstream ofs(input.homography);

        if (ofs.is_open())
        {
            // Write image1 string to file
            ofs << input.image1 << "\n";

            // Write image2 string to file
            ofs << input.image2 << "\n";

            // Write the homography to the file
            // Precision of 16 digits, row major, space seperated
            ofs << std::setprecision(16) << H.at<double>(0, 0) << " " << H.at<double>(0, 1) << " " << H.at<double>(0, 2) << "\n";
            ofs << std::setprecision(16) << H.at<double>(1, 0) << " " << H.at<double>(1, 1) << " " << H.at<double>(1, 2) << "\n";
            ofs << std::setprecision(16) << H.at<double>(2, 0) << " " << H.at<double>(2, 1) << " " << H.at<double>(2, 2);

            // Close the file
            ofs.close();
        }
        else
        {
            std::cout << "Could not open the homography file: " << input.homography << "\n";
        }
    }
}

void imadjust(cv::Mat& im1, cv::Mat& im2)
{
    // Convert images to 32 bit single channel floats
    im1.convertTo(im1, CV_32FC1);
    im2.convertTo(im2, CV_32FC1);

    // Get the mean of the first image
    const float mean1 = cv::mean(im1)[0];

    im1 = 32.0f * im1 / mean1;
    im2 = 32.0f * im2 / mean1;

    // Set all values greater than 255 to 255
    im1.setTo(255.0f, im1 > 255.0f);
    im2.setTo(255.0f, im2 > 255.0f);

    // Convert to single channel 8 bit unsigned integers
    im1.convertTo(im1, CV_8UC1);
    im2.convertTo(im2, CV_8UC1);
}
