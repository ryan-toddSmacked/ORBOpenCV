
#include <chrono>
#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <boost/program_options.hpp>

#include <hdf5.h>

static bool verbose = false;
static std::string input_file;
static int num_threads;
static int nfeatures;
static double scale_factor;
static int nlevels;
static int edge_threshold;
static int first_level;
static int score_type;
static int patch_size;
static int fast_threshold;
static double reproj_threshold;
static int max_iters;
static double confidence;
static bool sort_matches;
static double percent_matches;

static cv::Ptr<cv::ORB> detector;
static const cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);

static std::mutex mutex;

struct ORBWorker
{
    std::string image1;
    std::string image2;
    std::string output;
};

std::vector<ORBWorker> loadWorkers(const std::string& filename);
std::vector<std::string> splitcommas(std::string& str);
void ORBWorkerThread(const ORBWorker& worker);
void writeResults(const std::string& im1, const std::string& im2, const cv::Mat& H, const std::string& output);



const char about[] =
    "This program uses Oriented FAST and Rotated BRIEF (ORB) to detect\n"
    "keypoints and compute descriptors in two images, then matches the\n"
    "descriptors using a Brute-Force matcher and finds a homography\n"
    "matrix using RANSAC to align the images.\n"
    "The homography matrix is written to an HDF5 file.\n";


int main(int argc, char** argv)
{

    // Parse command line arguments
    namespace po = boost::program_options;
    po::options_description desc("Options");

    desc.add_options()
        ("help", "Print help message")
        ("verbose", "Print verbose messages")
        ("input-file", po::value<std::string>(), "Input file containing images to process")
        ("num-threads", po::value<int>()->default_value(1), "Number of parallel processes")
        ("nfeatures", po::value<int>()->default_value(500), "Number of features to detect")
        ("scale-factor", po::value<double>()->default_value(1.2), "Scale factor between levels in the scale pyramid")
        ("nlevels", po::value<int>()->default_value(8), "Number of levels in the scale pyramid")
        ("edge-threshold", po::value<int>()->default_value(31), "Size of the border where the features are not detected")
        ("first-level", po::value<int>()->default_value(0), "First level to start the scale pyramid")
        ("score-type", po::value<int>()->default_value(0), "Type of the score, 0=Harris, 1=FAST")
        ("patch-size", po::value<int>()->default_value(31), "Size of the patch used by the oriented BRIEF descriptor")
        ("fast-threshold", po::value<int>()->default_value(20), "Threshold for the FAST keypoint detector")
        ("reproj-threshold", po::value<double>()->default_value(3.0), "Maximum allowed reprojection error to treat a point pair as an inlier for RANSAC")
        ("max-iters", po::value<int>()->default_value(2000), "Maximum number of iterations to use for Homography generation")
        ("confidence", po::value<double>()->default_value(0.995), "Confidence level, between 0 and 1, for the estimated homography")
        ("sort-matches", po::value<bool>()->default_value(true), "Sort matches by distance")
        ("percent-matches", po::value<double>()->default_value(10.0), "Percent of matches to use for homography generation, only used if sort-matches is true");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    verbose = vm.count("verbose") > 0;
    if (vm.count("help"))
    {
        std::cout << about << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }
    if (!vm.count("input-file"))
    {
        std::cout << "Input file not specified" << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    input_file = vm["input-file"].as<std::string>();
    num_threads = vm["num-threads"].as<int>();
    nfeatures = vm["nfeatures"].as<int>();
    scale_factor = vm["scale-factor"].as<double>();
    nlevels = vm["nlevels"].as<int>();
    edge_threshold = vm["edge-threshold"].as<int>();
    first_level = vm["first-level"].as<int>();
    score_type = vm["score-type"].as<int>();
    patch_size = vm["patch-size"].as<int>();
    fast_threshold = vm["fast-threshold"].as<int>();
    reproj_threshold = vm["reproj-threshold"].as<double>();
    max_iters = vm["max-iters"].as<int>();
    confidence = vm["confidence"].as<double>();
    sort_matches = vm["sort-matches"].as<bool>();
    percent_matches = vm["percent-matches"].as<double>();

    // Load images
    std::vector<ORBWorker> workers = loadWorkers(input_file);

    if (workers.empty())
    {
        std::cerr << "No images to process" << std::endl;
        return 1;
    }

    std::vector<std::thread> threads(num_threads);
    int threads_working = 0;
    detector = cv::ORB::create(nfeatures, scale_factor, nlevels, edge_threshold, first_level, 2, (cv::ORB::ScoreType)score_type, patch_size, fast_threshold);
    
    for (size_t i = 0; i < workers.size(); ++i)
    {
        if (threads_working == num_threads)
        {
            for (size_t j = 0; j < threads.size(); ++j)
            {
                threads[j].join();
            }
            threads_working = 0;
        }

        threads[threads_working] = std::thread(ORBWorkerThread, workers[i]);
        threads_working++;
    }

    for (int i = 0; i < threads_working; ++i)
    {
        threads[i].join();
    }

    return 0;
}

std::vector<std::string> splitcommas(std::string& str)
{
    std::vector<std::string> result;
    std::string token;

    while (str.size() > 0)
    {
        size_t pos = str.find_first_of(',');
        if (pos == std::string::npos)
        {
            token = str;
            str.clear();
        }
        else
        {
            token = str.substr(0, pos);
            str.erase(0, pos + 1);
        }
        // Remove leading and trailing whitespace
        token.erase(0, token.find_first_not_of(" \t\n\r\f\v"));
        token.erase(token.find_last_not_of(" \t\n\r\f\v") + 1);
        result.push_back(token);
    }

    return result;
}

std::vector<ORBWorker> loadWorkers(const std::string& filename)
{
    // Open the file for reading
    std::ifstream file(filename);
    std::vector<ORBWorker> workers;

    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return workers;
    }

    std::string line;
    while (std::getline(file, line))
    {
        // Skip empty lines
        if (line.empty())
            continue;

        // Skip comment lines
        if (line[0] == '#')
            continue;

        // Split the line into tokens
        std::vector<std::string> tokens = splitcommas(line);

        // Skip lines with less than 3 tokens
        if (tokens.size() < 3)
            continue;

        // Create a worker
        ORBWorker worker;
        worker.image1 = tokens[0];
        worker.image2 = tokens[1];
        worker.output = tokens[2];

        // Add the worker to the list
        workers.push_back(worker);
    }

    return workers;
}

void writeResults(const std::string& im1, const std::string& im2, const cv::Mat& H, const std::string& output)
{
    const char* output_file = output.c_str();
    const char* image1 = im1.c_str();
    const char* image2 = im2.c_str();

    // Write results to HDF5 file
    hid_t file_id = H5Fcreate(output_file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write image names as variable length strings
    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, H5T_VARIABLE);

    hid_t dataspace = H5Screate(H5S_SCALAR);
    hid_t dataset = H5Dcreate(file_id, "image1", string_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &image1);
    H5Dclose(dataset);

    dataset = H5Dcreate(file_id, "image2", string_type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &image2);
    H5Dclose(dataset);
    H5Sclose(dataspace);

    // Write homography matrix, 3x3 double
    hsize_t dims[2] = {3, 3};
    dataspace = H5Screate_simple(2, dims, NULL);
    dataset = H5Dcreate(file_id, "homography", H5T_NATIVE_DOUBLE, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, H.data);
    H5Dclose(dataset);
    H5Sclose(dataspace);

    H5Fclose(file_id);
}

void ORBWorkerThread(const ORBWorker& worker)
{
    // Load images
    if (verbose)
    {
        std::chrono::high_resolution_clock::time_point t1, t2;
        double elapsed_time;
        std::stringstream ss;

        ss << "\n";
        ss << "============================================================\n";
        ss << "ORB Processing images: \n";
        ss << "    Input  -> " << worker.image1 << "\n";
        ss << "    Input  -> " << worker.image2 << "\n";
        ss << "    Output -> " << worker.output << "\n\n";

        cv::Mat image1, image2;

        t1 = std::chrono::high_resolution_clock::now();
        cv::equalizeHist(cv::imread(worker.image1, cv::IMREAD_GRAYSCALE), image1);
        cv::equalizeHist(cv::imread(worker.image2, cv::IMREAD_GRAYSCALE), image2);
        t2 = std::chrono::high_resolution_clock::now();

        if (image1.empty())
        {
            ss << "Failed to load image: " << worker.image1 << "\n";
            ss << "Skipping...\n";
            ss << "============================================================\n";
            mutex.lock();
            std::cout << ss.str();
            mutex.unlock();
            return;
        }
        if (image2.empty())
        {
            ss << "Failed to load image: " << worker.image2 << "\n";
            ss << "Skipping...\n";
            ss << "============================================================\n";
            mutex.lock();
            std::cout << ss.str();
            mutex.unlock();
            return;
        }

        // Get elapsed time as a double in seconds
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ss << "Loaded images:         " << elapsed_time << " [s]\n";

        // Detect keypoints and compute descriptors
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;

        // Detect keypoints
        t1 = std::chrono::high_resolution_clock::now();
        detector->detect(image1, keypoints1);
        detector->detect(image2, keypoints2);
        t2 = std::chrono::high_resolution_clock::now();

        // Get elapsed time as a double in seconds
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ss << "Detected keypoints:    " << elapsed_time << " [s]\n";

        // Compute descriptors
        t1 = std::chrono::high_resolution_clock::now();
        detector->compute(image1, keypoints1, descriptors1);
        detector->compute(image2, keypoints2, descriptors2);
        t2 = std::chrono::high_resolution_clock::now();

        // Get elapsed time as a double in seconds
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ss << "Computed descriptors:  " << elapsed_time << " [s]\n";

        // Match descriptors
        std::vector<cv::DMatch> matches;
        
        t1 = std::chrono::high_resolution_clock::now();
        matcher->match(descriptors1, descriptors2, matches);
        t2 = std::chrono::high_resolution_clock::now();

        // Get elapsed time as a double in seconds
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ss << "Matched descriptors:   " << elapsed_time << " [s]\n";

        // Sort matches by distance
        size_t num_matches = matches.size();
        if (sort_matches)
        {
            t1 = std::chrono::high_resolution_clock::now();
            std::sort(matches.begin(), matches.end());

            // keep only the top matches
            num_matches = matches.size() * (1.0 - percent_matches/100.0);
            t2 = std::chrono::high_resolution_clock::now();

            // Get elapsed time as a double in seconds
            elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
            ss << "Sorted matches:        " << elapsed_time << " [s]\n";
        }

        // Convert keypoints to points
        std::vector<cv::Point2f> points1(num_matches);
        std::vector<cv::Point2f> points2(num_matches);

        for (size_t i = 0; i < num_matches; ++i)
        {
            points1[i] = keypoints1[matches[i].queryIdx].pt;
            points2[i] = keypoints2[matches[i].trainIdx].pt;
        }

        // Find homography
        cv::Mat H;

        t1 = std::chrono::high_resolution_clock::now();
        H = cv::findHomography(points1, points2, cv::RANSAC, reproj_threshold, cv::noArray(), max_iters, confidence);
        t2 = std::chrono::high_resolution_clock::now();

        // Get elapsed time as a double in seconds
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ss << "Found homography:      " << elapsed_time << " [s]\n";

        // Write results
        // hdf5 does not like multiple threads
        mutex.lock();
        t1 = std::chrono::high_resolution_clock::now();
        writeResults(worker.image1, worker.image2, H, worker.output);
        t2 = std::chrono::high_resolution_clock::now();
        mutex.unlock();

        // Get elapsed time as a double in seconds
        elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        ss << "Wrote results:         " << elapsed_time << " [s]\n";

        ss << "============================================================\n";
        mutex.lock();
        std::cout << ss.str();
        mutex.unlock();
    }
    else
    {
        cv::Mat image1, image2;

        cv::equalizeHist(cv::imread(worker.image1, cv::IMREAD_GRAYSCALE), image1);
        cv::equalizeHist(cv::imread(worker.image2, cv::IMREAD_GRAYSCALE), image2);

        // Detect keypoints and compute descriptors
        std::vector<cv::KeyPoint> keypoints1, keypoints2;
        cv::Mat descriptors1, descriptors2;

        detector->detectAndCompute(image1, cv::Mat(), keypoints1, descriptors1);
        detector->detectAndCompute(image2, cv::Mat(), keypoints2, descriptors2);

        // Match descriptors
        std::vector<cv::DMatch> matches;

        matcher->match(descriptors1, descriptors2, matches);

        // Sort matches by distance
        size_t num_matches = matches.size();
        if (sort_matches)
        {
            std::sort(matches.begin(), matches.end());

            // keep only the top matches
            num_matches = matches.size() * (1.0 - percent_matches/100.0);
        }

        // Convert keypoints to points
        std::vector<cv::Point2f> points1(num_matches);
        std::vector<cv::Point2f> points2(num_matches);

        for (size_t i = 0; i < num_matches; ++i)
        {
            points1[i] = keypoints1[matches[i].queryIdx].pt;
            points2[i] = keypoints2[matches[i].trainIdx].pt;
        }

        // Find homography
        cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, reproj_threshold, cv::noArray(), max_iters, confidence);

        // Write results
        // hdf5 does not like multiple threads
        mutex.lock();
        writeResults(worker.image1, worker.image2, H, worker.output);
        mutex.unlock();
    }
    
}
