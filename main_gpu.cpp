

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <stack>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp> // findHomography

// gpu ORB from opencv
#include <opencv2/cudafeatures2d.hpp>

// cudaMemGetInfo header
#include <cuda_runtime_api.h>

// boost program options
#include <boost/program_options.hpp>

// hdf5 header
#include <hdf5.h>


//========================================================
// Static ORB parameters

static int s_nfeatures;
static double s_scaleFactor;
static int s_nlevels;
static int s_edgeThreshold;
static const int s_firstLevel = 0;
static const int s_WTA_K = 2;
static const int s_scoreType = (int)cv::ORB::HARRIS_SCORE;
static int s_patchSize;
static const int s_fastThreshold = 20;
static double s_ratio_thresh;

//========================================================
// Static general parameters
static bool s_verbose = false;
static std::vector<std::string> s_images(2);
static std::string s_output_file;
static std::string s_images_identifier;

//========================================================
// Static opencv Ptrs

static cv::Ptr<cv::cuda::ORB> s_gpu_orb = cv::cuda::ORB::create();
static const cv::Ptr<cv::cuda::DescriptorMatcher> s_gpu_matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);


//========================================================
// Static smart search parameters

static bool s_smart_search = false;
static int s_smart_search_ite;
static const std::vector<int> s_smart_search_parameter_ranges = { 50, 10 };
static double s_smart_search_x_shear;
static double s_smart_search_y_shear;
static double s_smart_search_x_scale;
static double s_smart_search_y_scale;
static bool s_smart_search_x_shear_enable = false;
static bool s_smart_search_y_shear_enable = false;
static bool s_smart_search_x_scale_enable = false;
static bool s_smart_search_y_scale_enable = false;

//========================================================
// Static timing variables

static std::stack<std::chrono::high_resolution_clock::time_point> s_timers;


//========================================================
// Static functions

/**
 * @brief Parse the command line arguments
 *
 * @param argc Number of arguments
 * @param argv String of arguments
 */
void parseArgs(int argc, char** argv);

/**
 * @brief Return the current UTC time as a string
 *
 * @return std::string UTC time string
 */
inline std::string UTC_time();

/**
 * @brief Log a message to the console
 *
 * @param msg Msg to log
 */
inline void log(const std::string& msg=std::string());

/**
 * @brief Start a timer
 */
inline void tic();

/**
 * @brief Stop a timer and return the elapsed time in seconds as a string
 *
 * @return std::string Elapsed time in seconds
 */
inline std::string toc();

/**
 * @brief Return the GPU info as a string
 *
 * @return std::string Information on the GPU
 */
inline std::string gpuInfo(int device=0);

/**
 * @brief Return a percentage of the GPU memory used as a string
 *
 * @return std::string Percentage of GPU memory used
 */
inline std::string gpuUsed();

/**
 * @brief Create a vector of doubles from start to end with n elements
 *
 * @param vec Vector to fill
 * @param start Start value
 * @param end End value
 */
inline void arange(std::vector<int>& vec, int start, int end);

/**
 * @brief Print the homography matrix to the console
 *
 * @param H Homography matrix
 */
inline void prettyPrint_Homography(const cv::Mat& H);

/**
 * @brief Update the ORB parameters
 * 
 */
inline void updateORBParams();

/**
 * @brief Load the images from disk to GPU
 *
 * @param img1 GPU matrix for image1
 * @param img2 GPU matrix for image2
 */
inline void loadImages_toGPU(cv::cuda::GpuMat& img1, cv::cuda::GpuMat& img2);

/**
 * @brief Calculate the homography once, given the command line arguments
 *
 */
void singleProcess_quiet(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography);

/**
 * @brief Calculate the homography once, given the command line arguments
 *
 */
void singleProcess_verbose(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography);

/**
 * @brief Smart search for particular shearing and or scaling of the image
 *
 * @param img1 GPU matrix for image1
 * @param img2 GPU matrix for image2
 * @param homography homography matrix
 */
void smartSearch_quiet(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography);

/**
 * @brief Smart search for particular shearing and or scaling of the image
 *
 * @param img1 GPU matrix for image1
 * @param img2 GPU matrix for image2
 * @param homography homography matrix
 */
void smartSearch_verbose(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography);

/**
 * @brief Score the homography
 *
 * @param homography Homography matrix
 * @return double Score
 */
inline double scoreHomography(const cv::Mat& homography);

/**
 * @brief Score the homographies and return the best one
 *
 * @param homographies Vector of homographies
 * @param scores Vector of scores
 * @param best_index Index of the best homography
 */
inline void scoreHomographies(const std::vector<cv::Mat>& homographies, std::vector<double>& scores, size_t& best_index);

/**
 * @brief Write the final homography to the output file
 *
 * @param homography Final homography
 */
inline void writeResults(const cv::Mat& H);

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    tic();
    parseArgs(argc, argv);

    cv::cuda::GpuMat img1, img2;

    loadImages_toGPU(img1, img2);

    cv::Mat homography;
    if (!s_verbose && !s_smart_search)
    {
        singleProcess_quiet(img1, img2, homography);
    }
    else if (s_verbose && !s_smart_search)
    {
        singleProcess_verbose(img1, img2, homography);
    }
    else if (!s_verbose && s_smart_search)
    {
        smartSearch_quiet(img1, img2, homography);
    }
    else
    {
        smartSearch_verbose(img1, img2, homography);
    }

    writeResults(homography);

    log("Final homography\n");
    prettyPrint_Homography(homography);
    std::cout << "\n";
    log("Total time: " + toc() + " [s]\n");

    return 0;
}

void parseArgs(int argc, char** argv)
{
    po::options_description desc("General options", 160, 80);
    po::options_description desc_gpu("GPU ORB options", 160, 80);
    po::options_description desc_smart_search("Smart search options", 160, 80);

    desc.add_options()
        ("help,h", "produce help message")
        ("verbose,v", "verbose output")
        ("input,i", po::value<std::vector<std::string>>(&s_images)->multitoken(), "input images on disk, 2 images required")
        ("output,o", po::value<std::string>(&s_output_file), "output file")
        ("identifier", po::value<std::string>(&s_images_identifier), "identifier for the images");

    desc_gpu.add_options()
        ("nfeatures", po::value<int>(&s_nfeatures)->default_value(500), "The maximum number of features to retain.")
        ("scale-factor", po::value<double>(&s_scaleFactor)->default_value(1.2), "The scale factor for building the image pyramid.")
        ("nlevels", po::value<int>(&s_nlevels)->default_value(5), "The number of pyramid levels.")
        ("edge-threshold", po::value<int>(&s_edgeThreshold)->default_value(31), "This is size of the border where the features are not detected.")
        ("patch-size", po::value<int>(&s_patchSize)->default_value(31), "Size of the patch used by the oriented BRIEF descriptor.")
        ("ratio-thresh", po::value<double>(&s_ratio_thresh)->default_value(0.75), "The ratio between the best match and the second best match to decide whether a match is good or not.");

    desc_smart_search.add_options()
        ("smart-search", "Perform a smart search for particular shearing and or scaling of the image")
        ("smart-search-x-shear", po::value<double>(&s_smart_search_x_shear)->default_value(0.0), "The target shearing for the x direction")
        ("smart-search-y-shear", po::value<double>(&s_smart_search_y_shear)->default_value(0.0), "The target shearing for the y direction")
        ("smart-search-x-scale", po::value<double>(&s_smart_search_x_scale)->default_value(1.0), "The target scaling for the x direction")
        ("smart-search-y-scale", po::value<double>(&s_smart_search_y_scale)->default_value(1.0), "The target scaling for the y direction");

    desc.add(desc_gpu).add(desc_smart_search);


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(EXIT_FAILURE);
    }

    if (s_images.size() != 2)
    {
        std::cout << "Error: 2 images required" << std::endl;
        std::cout << desc << std::endl;
        exit(EXIT_FAILURE);
    }

    if (vm.count("output") == 0)
    {
        std::cout << "Error: output file required" << std::endl;
        std::cout << desc << std::endl;
        exit(EXIT_FAILURE);
    }

    s_verbose = vm.count("verbose") > 0;

    if (s_verbose)
    {
        log("CUDA Device Info\n");
        std::cout << gpuInfo() << "\n" << std::endl;
    }

    if (vm.count("smart-search"))
    {
        s_smart_search = true;
        s_smart_search_x_shear_enable = vm.count("smart-search-x-shear") > 0;
        s_smart_search_y_shear_enable = vm.count("smart-search-y-shear") > 0;
        s_smart_search_x_scale_enable = vm.count("smart-search-x-scale") > 0;
        s_smart_search_y_scale_enable = vm.count("smart-search-y-scale") > 0;
    }

    updateORBParams();

    if (s_verbose && s_smart_search)
    {
        log("Smart search parameters\n");
        std::cout << "    iterations: " << s_smart_search_ite << std::endl;
        std::cout << "    x-shear:    " << s_smart_search_x_shear << std::endl;
        std::cout << "    y-shear:    " << s_smart_search_y_shear << std::endl;
        std::cout << "    x-scale:    " << s_smart_search_x_scale << std::endl;
        std::cout << "    y-scale:    " << s_smart_search_y_scale << "\n" << std::endl;
    }
}


inline std::string UTC_time()
{
    // Get current time, accurate to milliseconds
    // Format string as follows
    // "YYYY-MM-DD HH:MM:SS.mmm"
    // Make sure to add leading zeros to the milliseconds, so the string is always the same length
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::chrono::milliseconds ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    char buf[100];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));

    // If the milliseconds are less than 100, add leading zeros so the length is always 3 characters
    std::string ms_str = std::to_string(ms.count());

    if (ms_str.size() == 1)
        ms_str = "00" + ms_str;
    else if (ms_str.size() == 2)
        ms_str = "0" + ms_str;
    
    std::string time_str = std::string(buf) + "." + ms_str;

    return time_str;
}

inline void log(const std::string& msg)
{
    std::cout << "[" << UTC_time() << "]: " << msg;
}

inline void tic()
{
    s_timers.push(std::chrono::high_resolution_clock::now());
}

inline std::string toc()
{
    std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
    if (s_timers.empty())
    {
        return std::string("0");
    }

    std::chrono::duration<double> elapsed = now - s_timers.top();
    s_timers.pop();
    return std::to_string(elapsed.count());
};

inline std::string gpuInfo(int device)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Format string as follows
    //  Device name - Compute capability X.Y
    //     Memory Clock Rate (KHz): 3004000
    //     Total Memory (Gb): 12.000000
    //     Memory Bus Width (bits): 384
    //     Peak Memory Bandwidth (GB/s): 288.384

    std::string info = std::string("  ") + prop.name + std::string(" - Compute capability ") + std::to_string(prop.major) + std::string(".") + std::to_string(prop.minor);
    info += "\n    Memory Clock Rate (KHz):      " + std::to_string(prop.memoryClockRate);
    info += "\n    Total Memory (GB):            " + std::to_string(prop.totalGlobalMem / 1.0e9);
    info += "\n    Memory Bus Width (bits):      " + std::to_string(prop.memoryBusWidth);
    info += "\n    Peak Memory Bandwidth (GB/s): " + std::to_string(2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    return info;
}

inline std::string gpuUsed()
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return std::to_string(100.0 * (1.0 - (double)free / (double)total));
}

inline void arange(std::vector<int>& vec, int start, int end)
{
    vec.resize(end - start + 1);
    for (int i = 0; i < vec.size(); i++)
        vec[i] = start + i;
}

inline void prettyPrint_Homography(const cv::Mat& H)
{
    fprintf(stdout, "    %15.8lf  %15.8lf  %15.8lf\n", H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2));
    fprintf(stdout, "    %15.8lf  %15.8lf  %15.8lf\n", H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2));
    fprintf(stdout, "    %15.8lf  %15.8lf  %15.8lf\n", H.at<double>(2, 0), H.at<double>(2, 1), H.at<double>(2, 2));
}

inline void updateORBParams()
{
    if (s_verbose)
    {
        log("Updating ORB parameters\n");
    }
    s_gpu_orb->setMaxFeatures(s_nfeatures);
    s_gpu_orb->setScaleFactor(s_scaleFactor);
    s_gpu_orb->setNLevels(s_nlevels);
    s_gpu_orb->setEdgeThreshold(s_edgeThreshold);
    s_gpu_orb->setFirstLevel(s_firstLevel);
    s_gpu_orb->setWTA_K(s_WTA_K);
    s_gpu_orb->setScoreType(s_scoreType);
    s_gpu_orb->setPatchSize(s_patchSize);
    s_gpu_orb->setFastThreshold(s_fastThreshold);
}

inline void loadImages_toGPU(cv::cuda::GpuMat& img1, cv::cuda::GpuMat& img2)
{
    cv::Mat img1_cpu, img2_cpu;

    if (s_verbose)
    {
        tic();
        log("Reading image from disk \"" + s_images[0] + "\"");
        cv::imread(s_images[0], cv::IMREAD_GRAYSCALE).convertTo(img1_cpu, CV_8UC1);
        std::cout << " -- Done " << toc() << " [s]\n";

        tic();
        log("Reading image from disk \"" + s_images[1] + "\"");
        cv::imread(s_images[1], cv::IMREAD_GRAYSCALE).convertTo(img2_cpu, CV_8UC1);
        std::cout << " -- Done " << toc() << " [s]\n";

        tic();
        log("Uploading image to GPU \"" + s_images[0] + "\"");
        img1.upload(img1_cpu);
        std::cout << " -- Done " << toc() << " [s]\n";

        tic();
        log("Uploading image to GPU \"" + s_images[1] + "\"");
        img2.upload(img2_cpu);
        std::cout << " -- Done " << toc() << " [s]\n";
    }
    else
    {
        cv::imread(s_images[0], cv::IMREAD_GRAYSCALE).convertTo(img1_cpu, CV_8UC1);
        cv::imread(s_images[1], cv::IMREAD_GRAYSCALE).convertTo(img2_cpu, CV_8UC1);
        img1.upload(img1_cpu);
        img2.upload(img2_cpu);
    }
}

void singleProcess_quiet(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography)
{
    std::vector<cv::Point2f> points1, points2;

    {
        std::vector<cv::KeyPoint> keypoints1_cpu, keypoints2_cpu;
        std::vector<std::vector<cv::DMatch>> matches_cpu;

        {
            cv::cuda::GpuMat keypoints1, keypoints2;
            cv::cuda::GpuMat descriptors1, descriptors2;
            cv::cuda::Stream stream1, stream2;
            cv::cuda::GpuMat matches;

            s_gpu_orb->detectAndComputeAsync(img1, cv::cuda::GpuMat(), keypoints1, descriptors1, false, stream1);
            s_gpu_orb->detectAndComputeAsync(img2, cv::cuda::GpuMat(), keypoints2, descriptors2, false, stream2);

            stream1.waitForCompletion();
            stream2.waitForCompletion();

            s_gpu_matcher->knnMatchAsync(descriptors1, descriptors2, matches, 2, cv::cuda::GpuMat(), stream1);

            // Download keypoints from GPU
            s_gpu_orb->convert(keypoints1, keypoints1_cpu);
            s_gpu_orb->convert(keypoints2, keypoints2_cpu);

            stream1.waitForCompletion();

            // Download matches from GPU
            s_gpu_matcher->knnMatchConvert(matches, matches_cpu);
        }

        // Extract good matches straight to points1 and points2
        points1.reserve(matches_cpu.size());
        points2.reserve(matches_cpu.size());
        for (size_t i = 0; i < matches_cpu.size(); i++)
        {
            if (matches_cpu[i][0].distance < s_ratio_thresh * matches_cpu[i][1].distance)
            {
                points1.push_back(keypoints1_cpu[matches_cpu[i][0].queryIdx].pt);
                points2.push_back(keypoints2_cpu[matches_cpu[i][0].trainIdx].pt);
            }
        }
    }

    homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, cv::noArray(), 2000, 0.995);
}

void singleProcess_verbose(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography)
{
    std::vector<cv::Point2f> points1, points2;

    {
        std::vector<cv::KeyPoint> keypoints1_cpu, keypoints2_cpu;
        std::vector<std::vector<cv::DMatch>> matches_cpu;

        {
            cv::cuda::GpuMat keypoints1, keypoints2;
            cv::cuda::GpuMat descriptors1, descriptors2;
            cv::cuda::Stream stream1, stream2;
            cv::cuda::GpuMat matches;

            log("Detecting keypoints and computing descriptors");
            tic();
            s_gpu_orb->detectAndComputeAsync(img1, cv::cuda::GpuMat(), keypoints1, descriptors1, false, stream1);
            s_gpu_orb->detectAndComputeAsync(img2, cv::cuda::GpuMat(), keypoints2, descriptors2, false, stream2);

            stream1.waitForCompletion();
            stream2.waitForCompletion();

            std::cout << " -- Done " << toc() << " [s]\n";

            tic();
            log("Matching descriptors");
            s_gpu_matcher->knnMatchAsync(descriptors1, descriptors2, matches, 2, cv::cuda::GpuMat(), stream1);

            stream1.waitForCompletion();
            std::cout << " -- Done " << toc() << " [s]\n";

            // Download keypoints from GPU
            tic();
            log("Downloading keypoints");
            s_gpu_orb->convert(keypoints1, keypoints1_cpu);
            s_gpu_orb->convert(keypoints2, keypoints2_cpu);

            std::cout << " -- Done " << toc() << " [s]\n";

            // Download matches from GPU
            tic();
            log("Downloading matches");
            s_gpu_matcher->knnMatchConvert(matches, matches_cpu);
            std::cout << " -- Done " << toc() << " [s]\n";
        }

        // Extract good matches straight to points1 and points2
        log("Extracting good matches");
        tic();
        points1.reserve(matches_cpu.size());
        points2.reserve(matches_cpu.size());

        for (size_t i = 0; i < matches_cpu.size(); i++)
        {
            if (matches_cpu[i][0].distance < s_ratio_thresh * matches_cpu[i][1].distance)
            {
                points1.push_back(keypoints1_cpu[matches_cpu[i][0].queryIdx].pt);
                points2.push_back(keypoints2_cpu[matches_cpu[i][0].trainIdx].pt);
            }
        }
        std::cout << " -- Done " << toc() << " [s]\n";
    }

    log("Calculating homography");
    tic();
    homography = cv::findHomography(points1, points2, cv::RANSAC, 3.0, cv::noArray(), 2000, 0.995);
    std::cout << " -- Done " << toc() << " [s]\n";
}

void smartSearch_quiet(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography)
{
    std::vector<int> edgeThresholds;
    std::vector<int> patchSizes;
    std::vector<cv::Mat> homographies;
    std::vector<double> scores;
    cv::Mat best_homography;
    double best_score = std::numeric_limits<double>::max();
    size_t best_index = 0;
    int ite = 0;

    s_smart_search_ite = s_smart_search_parameter_ranges.size();

    const int start_center_parameter = s_smart_search_parameter_ranges[0] + 3;
    const int min_param_size = 3;
    const int max_param_size = start_center_parameter + s_smart_search_parameter_ranges[0];

    arange(edgeThresholds, start_center_parameter - s_smart_search_parameter_ranges[0], start_center_parameter + s_smart_search_parameter_ranges[0]);
    arange(patchSizes, start_center_parameter - s_smart_search_parameter_ranges[0], start_center_parameter + s_smart_search_parameter_ranges[0]);

    std::vector<cv::Point2f> points1, points2;

    std::vector<cv::KeyPoint> keypoints1_cpu, keypoints2_cpu;
    std::vector<std::vector<cv::DMatch>> matches_cpu;

    cv::cuda::GpuMat keypoints1, keypoints2;
    cv::cuda::GpuMat descriptors1, descriptors2;
    cv::cuda::Stream stream1, stream2;
    cv::cuda::GpuMat matches;

    do {
        const size_t n = edgeThresholds.size();
        homographies.resize(n);
        for (int i = 0; i < n; i++)
        {
            s_edgeThreshold = edgeThresholds[i];
            s_patchSize = patchSizes[i];

            updateORBParams();

            s_gpu_orb->detectAndComputeAsync(img1, cv::cuda::GpuMat(), keypoints1, descriptors1, false, stream1);
            s_gpu_orb->detectAndComputeAsync(img2, cv::cuda::GpuMat(), keypoints2, descriptors2, false, stream2);
            
            stream1.waitForCompletion();
            stream2.waitForCompletion();

            s_gpu_matcher->knnMatchAsync(descriptors1, descriptors2, matches, 2, cv::cuda::GpuMat(), stream1);

            // Download keypoints from GPU
            s_gpu_orb->convert(keypoints1, keypoints1_cpu);
            s_gpu_orb->convert(keypoints2, keypoints2_cpu);

            stream1.waitForCompletion();

            // Download matches from GPU
            s_gpu_matcher->knnMatchConvert(matches, matches_cpu);

            // Extract good matches straight to points1 and points2
            points1.reserve(matches_cpu.size());
            points2.reserve(matches_cpu.size());
            for (size_t i = 0; i < matches_cpu.size(); i++)
            {
                if (matches_cpu[i][0].distance < s_ratio_thresh * matches_cpu[i][1].distance)
                {
                    points1.push_back(keypoints1_cpu[matches_cpu[i][0].queryIdx].pt);
                    points2.push_back(keypoints2_cpu[matches_cpu[i][0].trainIdx].pt);
                }
            }

            homographies[i] = std::move(cv::findHomography(points1, points2, cv::RANSAC, 3.0, cv::noArray(), 2000, 0.995));
        }

        scoreHomographies(homographies, scores, best_index);

        // Update the best homography
        if (scores[best_index] < best_score)
        {
            best_score = scores[best_index];
            best_homography = homographies[best_index];
        }

        if (ite + 1 == s_smart_search_ite)
            break;

        // Update the search space
        int min_edgeThreshold = std::max(min_param_size, edgeThresholds[best_index] - s_smart_search_parameter_ranges[ite+1]);
        int max_edgeThreshold = std::min(max_param_size, edgeThresholds[best_index] + s_smart_search_parameter_ranges[ite+1]);
        int min_patchSize = std::max(min_param_size, patchSizes[best_index] - s_smart_search_parameter_ranges[ite+1]);
        int max_patchSize = std::min(max_param_size, patchSizes[best_index] + s_smart_search_parameter_ranges[ite+1]);

        arange(edgeThresholds, min_edgeThreshold, max_edgeThreshold);
        arange(patchSizes, min_patchSize, max_patchSize);

        ite++;

    } while (1);

    homography = best_homography;
}

void smartSearch_verbose(const cv::cuda::GpuMat& img1, const cv::cuda::GpuMat& img2, cv::Mat& homography)
{
    std::vector<int> edgeThresholds;
    std::vector<int> patchSizes;
    std::vector<cv::Mat> homographies;
    std::vector<double> scores;
    cv::Mat best_homography;
    double best_score = std::numeric_limits<double>::max();
    size_t best_index = 0;
    int ite = 0;

    s_smart_search_ite = s_smart_search_parameter_ranges.size();

    const int start_center_parameter = s_smart_search_parameter_ranges[0] + 3;
    const int min_param_size = 3;
    const int max_param_size = start_center_parameter + s_smart_search_parameter_ranges[0];

    arange(edgeThresholds, start_center_parameter - s_smart_search_parameter_ranges[0], start_center_parameter + s_smart_search_parameter_ranges[0]);
    arange(patchSizes, start_center_parameter - s_smart_search_parameter_ranges[0], start_center_parameter + s_smart_search_parameter_ranges[0]);

    std::vector<cv::Point2f> points1, points2;

    std::vector<cv::KeyPoint> keypoints1_cpu, keypoints2_cpu;
    std::vector<std::vector<cv::DMatch>> matches_cpu;

    cv::cuda::GpuMat keypoints1, keypoints2;
    cv::cuda::GpuMat descriptors1, descriptors2;
    cv::cuda::Stream stream1, stream2;
    cv::cuda::GpuMat matches;

    do {

        log("Smart Search Iteration {" + std::to_string(ite+1) + " | " + std::to_string(s_smart_search_ite) + "}\n");
        const size_t n = edgeThresholds.size();
        homographies.resize(n);
        for (int i = 0; i < n; i++)
        { // This scope should be placede in a try catch block
            s_edgeThreshold = edgeThresholds[i];
            s_patchSize = patchSizes[i];

            updateORBParams();

            tic();
            log("Detecting keypoints and computing descriptors");

            s_gpu_orb->detectAndComputeAsync(img1, cv::cuda::GpuMat(), keypoints1, descriptors1, false, stream1);
            s_gpu_orb->detectAndComputeAsync(img2, cv::cuda::GpuMat(), keypoints2, descriptors2, false, stream2);

            stream1.waitForCompletion();
            stream2.waitForCompletion();
            std::cout << " -- Done " << toc() << " [s]\n";

            tic();
            log("Matching descriptors");
            s_gpu_matcher->knnMatchAsync(descriptors1, descriptors2, matches, 2, cv::cuda::GpuMat(), stream1);

            stream1.waitForCompletion();
            std::cout << " -- Done " << toc() << " [s]\n";

            // Download keypoints from GPU
            tic();
            log("Downloading keypoints");
            s_gpu_orb->convert(keypoints1, keypoints1_cpu);
            s_gpu_orb->convert(keypoints2, keypoints2_cpu);

            std::cout << " -- Done " << toc() << " [s]\n";

            // Download matches from GPU
            tic();
            log("Downloading matches");
            s_gpu_matcher->knnMatchConvert(matches, matches_cpu);

            std::cout << " -- Done " << toc() << " [s]\n";

            // Extract good matches straight to points1 and points2
            tic();
            log("Extracting good matches");
            points1.reserve(matches_cpu.size());
            points2.reserve(matches_cpu.size());
            for (size_t i = 0; i < matches_cpu.size(); i++)
            {
                if (matches_cpu[i][0].distance < s_ratio_thresh * matches_cpu[i][1].distance)
                {
                    points1.push_back(keypoints1_cpu[matches_cpu[i][0].queryIdx].pt);
                    points2.push_back(keypoints2_cpu[matches_cpu[i][0].trainIdx].pt);
                }
            }
            std::cout << " -- Done " << toc() << " [s]\n";

            tic();
            log("Calculating homography");
            homographies[i] = std::move(cv::findHomography(points1, points2, cv::RANSAC, 3.0, cv::noArray(), 2000, 0.995));
            std::cout << " -- Done " << toc() << " [s]\n";

        }

        scoreHomographies(homographies, scores, best_index);

        // Update the best homography
        if (scores[best_index] < best_score)
        {
            best_score = scores[best_index];
            best_homography = homographies[best_index];
        }

        if (ite + 1 == s_smart_search_ite)
            break;

        // Update the search space
        int min_edgeThreshold = std::max(min_param_size, edgeThresholds[best_index] - s_smart_search_parameter_ranges[ite+1]);
        int max_edgeThreshold = std::min(max_param_size, edgeThresholds[best_index] + s_smart_search_parameter_ranges[ite+1]);
        int min_patchSize = std::max(min_param_size, patchSizes[best_index] - s_smart_search_parameter_ranges[ite+1]);
        int max_patchSize = std::min(max_param_size, patchSizes[best_index] + s_smart_search_parameter_ranges[ite+1]);

        arange(edgeThresholds, min_edgeThreshold, max_edgeThreshold);
        arange(patchSizes, min_patchSize, max_patchSize);

        ite++;

    } while (1);

    homography = best_homography;
}


inline double scoreHomography(const cv::Mat& homography)
{
    double score = std::numeric_limits<double>::max();
    if (homography.empty())
        return score;

    score = 0.0;

    score += std::abs(homography.at<double>(0, 1) - s_smart_search_x_shear);
    score += std::abs(homography.at<double>(1, 0) - s_smart_search_y_shear);
    score += std::abs(homography.at<double>(0, 0) - s_smart_search_x_scale);
    score += std::abs(homography.at<double>(1, 1) - s_smart_search_y_scale);

    return score;
}

inline void scoreHomographies(const std::vector<cv::Mat>& homographies, std::vector<double>& scores, size_t& best_index)
{
    scores.resize(homographies.size());
    double best_score = std::numeric_limits<double>::max();
    for (size_t i = 0; i < homographies.size(); i++)
    {
        scores[i] = scoreHomography(homographies[i]);
        if (scores[i] < best_score)
        {
            best_score = scores[i];
            best_index = i;
        }
    }
}

inline void writeResults(const cv::Mat& H)
{
    const char* file = s_output_file.c_str();
    const char* image1_file = s_images[0].c_str();
    const char* image2_file = s_images[1].c_str();
    const char* identifier = s_images_identifier.c_str();

    // create hdf5 file, overwrite if exists
    hid_t file_id = H5Fcreate(file, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // create dataspace, [3x3] matrix of doubles
    hsize_t dims[2] = { 3, 3 };
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

    // create dataset
    hid_t dataset_id = H5Dcreate2(file_id, "H", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // write data
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, H.data);

    // close dataset
    H5Dclose(dataset_id);

    // close dataspace
    H5Sclose(dataspace_id);

    // Create dataspace for image1 name, variale length string
    hid_t dataspace_id2 = H5Screate(H5S_SCALAR);
    hid_t string_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type, H5T_VARIABLE);

    // Create dataset for image1 name
    hid_t dataset_id2 = H5Dcreate2(file_id, "image1", string_type, dataspace_id2, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write image1 name
    H5Dwrite(dataset_id2, string_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, &image1_file);

    // Close dataset
    H5Dclose(dataset_id2);

    // Close dataspace
    H5Sclose(dataspace_id2);

    // Create dataspace for image2 name, variale length string
    hid_t dataspace_id3 = H5Screate(H5S_SCALAR);
    hid_t string_type2 = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type2, H5T_VARIABLE);

    // Create dataset for image2 name
    hid_t dataset_id3 = H5Dcreate2(file_id, "image2", string_type2, dataspace_id3, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write image2 name
    H5Dwrite(dataset_id3, string_type2, H5S_ALL, H5S_ALL, H5P_DEFAULT, &image2_file);

    // Close dataset
    H5Dclose(dataset_id3);

    // Close dataspace
    H5Sclose(dataspace_id3);

    // Write the images identifier
    hid_t dataspace_id5 = H5Screate(H5S_SCALAR);
    hid_t string_type3 = H5Tcopy(H5T_C_S1);
    H5Tset_size(string_type3, H5T_VARIABLE);

    // Create dataset for images identifier
    hid_t dataset_id5 = H5Dcreate2(file_id, "identifier", string_type3, dataspace_id5, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Write images identifier
    H5Dwrite(dataset_id5, string_type3, H5S_ALL, H5S_ALL, H5P_DEFAULT, &identifier);

    // Close dataset
    H5Dclose(dataset_id5);

    // Close dataspace
    H5Sclose(dataspace_id5);

    // close hdf5 file
    H5Fclose(file_id);

}


