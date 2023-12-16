
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

// Check for gpu support
#ifdef HAVE_OPENCV_CUDAFEATURES2D
#define GPU (1)
#include <opencv2/cudafeatures2d.hpp>
#endif

#include <boost/program_options.hpp>

#include <hdf5.h>

#include <iostream>
#include <functional>
#include <iomanip>
#include <random>
#include <thread>
#include <limits>
#include <chrono>

namespace po = boost::program_options;

struct ORBInput
{
    int nfeatures;
    double scaleFactor;
    int nlevels;
    int edgeThreshold;
    int firstLevel;
    int scoreType;
    int patchSize;
    int fastThreshold;
    int ransacIters;
    int maxMatches;
    double ransacReprojThreshold;
    double confidence;

    cv::Mat H;
};

static cv::Mat s_image1;
static cv::Mat s_image2;
static std::string s_output_file;
static cv::Mat s_image1_mask;
static cv::Mat s_image2_mask;

static double s_max_x_scaling;
static double s_max_y_scaling;
static double s_max_x_shearing;
static double s_max_y_shearing;

static bool s_smart_search = false;
static int s_nthreads;
static int s_max_iterations;

static bool s_verbose = false;

static cv::Mat s_homography;

static std::mutex s_mutex;

static cv::Ptr<cv::ORB> s_orb;
static cv::Ptr<cv::BFMatcher> s_matcher;


// main worker
void calc_homography(ORBInput& input);

// smart search function
void smart_search(ORBInput& input);

// single search function
void single_search(ORBInput& input);

// write results to hdf5 file
void write_homography(const std::string& filename, const cv::Mat& H);

// evolve input parameters
void evolve_inputs(std::vector<ORBInput>& inputs);

// best homography
int best_homography(const std::vector<ORBInput>& homographies);


inline void prettyPrintHomography(const cv::Mat& H)
{
    const char* floatfmt = "%15.9lf";

    for (int i = 0; i < H.rows; i++)
    {
        for (int j = 0; j < H.cols; j++)
        {
            fprintf(stdout, " ");
            fprintf(stdout, floatfmt, H.at<double>(i, j));
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv)
{
    srand(42);

    // Optional arguments:
    int nfeatures = 500;
    double scaleFactor = 1.2;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    //const int WTA_K = 2; // Always 2, dont change
    int scoreType = (int)cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;
    int maxMatches = 200;
    int ransacIters = 1000;
    double ransacReprojThreshold = 3.0;
    double confidence = 0.995;

    // Arguments for scoring the homography matrix
    constexpr double infinity = std::numeric_limits<double>::infinity();

    // usage example: ./main "../images/im1.jpg" "../images/im2.jpg" "../results/homography.h5" [OPTIONS]

    // Declare the supported options.
    po::options_description helpers("Helper options");
    po::options_description required("Required options");
    po::options_description desc("ORB options");
    po::options_description homography_desc("Homography options");

    helpers.add_options()
        ("help,h", "Produce help message")
        ("verbose,v", "Verbose output")
        ("nthreads", po::value<int>(&s_nthreads)->default_value(1), "[int] Number of threads");

    required.add_options()
        ("inputs,i", po::value<std::vector<std::string>>()->multitoken(), "[strings] 2 input images on disk")
        ("output,o", po::value<std::string>(), "[string] Output file");

    desc.add_options()
        ("nfeatures", po::value<int>(&nfeatures)->default_value(500), "[int] Number of features")
        ("scale-factor", po::value<double>(&scaleFactor)->default_value(1.2), "[double] Scale factor")
        ("nlevels", po::value<int>(&nlevels)->default_value(8), "[int] Number of levels")
        ("edge-threshold", po::value<int>(&edgeThreshold)->default_value(31), "[int] Edge threshold")
        ("first-level", po::value<int>(&firstLevel)->default_value(0), "[int] First level")
        ("score-type", po::value<int>(&scoreType)->default_value(cv::ORB::HARRIS_SCORE), "[int] Score type, 0 = Harris, 1 = FAST")
        ("patch-size", po::value<int>(&patchSize)->default_value(31), "[int] Patch size")
        ("fast-threshold", po::value<int>(&fastThreshold)->default_value(20), "[int] Fast threshold")
        ("image1-mask", po::value<std::string>(), "[string] Image 1 mask")
        ("image2-mask", po::value<std::string>(), "[string] Image 2 mask");
    
    homography_desc.add_options()
        ("smart-search", "Use smart search. Smart search should be used with at least 1 of the scaling and shearing arguments")
        ("max-matches", po::value<int>(&maxMatches)->default_value(200), "[int] Maximum number of matches to keep")
        ("ransac-iters", po::value<int>(&ransacIters)->default_value(1000), "[int] Maximum number of RANSAC iterations")
        ("ransac-reproj-threshold", po::value<double>(&ransacReprojThreshold)->default_value(3.0), "[double] Maximum allowed reprojection error to treat a point pair as an inlier")
        ("confidence", po::value<double>(&confidence)->default_value(0.995), "[double] Confidence level of the estimated homography matrix")
        ("max-iterations", po::value<int>(&s_max_iterations)->default_value(50), "[int] Maximum number of smart search iterations")
        ("target-x-scaling", po::value<double>(&s_max_x_scaling)->default_value(infinity), "[double] Estimated scaling in x-direction of the computed homography matrix")
        ("target-y-scaling", po::value<double>(&s_max_y_scaling)->default_value(infinity), "[double] Estimated scaling in y-direction of the computed homography matrix")
        ("target-x-shearing", po::value<double>(&s_max_x_shearing)->default_value(infinity), "[double] Estimated shearing in x-direction of the computed homography matrix")
        ("target-y-shearing", po::value<double>(&s_max_y_shearing)->default_value(infinity), "[double] Estimated shearing in y-direction of the computed homography matrix");


    po::variables_map vm;

    po::options_description all_options;
    all_options.add(helpers).add(required).add(desc).add(homography_desc);

    // Parse command line arguments
    po::store(po::parse_command_line(argc, argv, all_options), vm);

    // Check if required arguments are given
    if (vm.count("inputs") == 0)
    {
        std::cerr << "Error: No input images given\n";
        // Print usage
        all_options.print(std::cout);

        std::cout << "\nUsage: " << argv[0] << " -i 'image1' 'image2' -o 'output' [OPTIONS]\n\n";
        return 1;
    }

    if (vm.count("output") == 0)
    {
        std::cerr << "Error: No output file given\n";
        // Print usage
        all_options.print(std::cout);

        std::cout << "\nUsage: " << argv[0] << " -i 'image1' 'image2' -o 'output' [OPTIONS]\n\n";
        return 1;
    }

    po::notify(vm);


    // Check if help is requested
    if (vm.count("help"))
    {
        // Print usage
        all_options.print(std::cout);

        std::cout << "\nUsage: " << argv[0] << " -i 'image1' 'image2' -o 'output' [OPTIONS]\n\n";
        return 1;
    }

    s_verbose = vm.count("verbose") > 0;
    s_smart_search = vm.count("smart-search") > 0;

    // Read input images
    std::vector<std::string> inputs = vm["inputs"].as<std::vector<std::string>>();
    std::string im1;
    std::string im2;
    // check if there are 2 input images
    if (inputs.size() != 2)
    {
        std::cerr << "Error: Expected 2 input images, got " << inputs.size() << "\n";
        // Print usage
        all_options.print(std::cout);

        std::cout << "\nUsage: " << argv[0] << " -i 'image1' 'image2' -o 'output' [OPTIONS]\n\n";
        return 1;
    }
    im1 = inputs[0];
    im2 = inputs[1];
    s_image1 = cv::imread(im1, cv::IMREAD_GRAYSCALE);
    s_image2 = cv::imread(im2, cv::IMREAD_GRAYSCALE);

    if (s_image1.empty())
    {
        std::cerr << "Error: Could not read image1: " << im1 << "\n";
        return 1;
    }

    if (s_image2.empty())
    {
        std::cerr << "Error: Could not read image2: " << im2 << "\n";
        return 1;
    }

    s_output_file = vm["output"].as<std::string>();

    // Read masks
    if (vm.count("image1-mask"))
    {
        std::string mask1 = vm["image1-mask"].as<std::string>();
        s_image1_mask = cv::imread(mask1, cv::IMREAD_GRAYSCALE);
        if (s_image1_mask.empty())
        {
            std::cerr << "Warning: Could not read image1 mask: " << mask1 << "\n";
            std::cerr << "Using entire image\n";
            s_image1_mask = cv::Mat(); //cv::Mat::ones(s_image1.size(), CV_8U);
        }
    }
    else
        s_image1_mask = cv::Mat(); //cv::Mat::ones(s_image1.size(), CV_8U);

    if (vm.count("image2-mask"))
    {
        std::string mask2 = vm["image2-mask"].as<std::string>();
        s_image2_mask = cv::imread(mask2, cv::IMREAD_GRAYSCALE);
        if (s_image2_mask.empty())
        {
            std::cerr << "Warning: Could not read image2 mask: " << mask2 << "\n";
            std::cerr << "Using entire image\n";
            s_image2_mask = cv::Mat(); //cv::Mat::ones(s_image2.size(), CV_8U);
        }
    }
    else
        s_image2_mask = cv::Mat(); //cv::Mat::ones(s_image2.size(), CV_8U);


    ORBInput input;
    input.nfeatures = nfeatures;
    input.scaleFactor = scaleFactor;
    input.nlevels = nlevels;
    input.edgeThreshold = edgeThreshold;
    input.firstLevel = firstLevel;
    input.scoreType = scoreType;
    input.patchSize = patchSize;
    input.fastThreshold = fastThreshold;
    input.ransacIters = ransacIters;
    input.maxMatches = maxMatches;
    input.ransacReprojThreshold = ransacReprojThreshold;
    input.confidence = confidence;

    s_orb = cv::ORB::create(
        input.nfeatures,
        input.scaleFactor,
        input.nlevels,
        input.edgeThreshold,
        input.firstLevel,
        2,
        (cv::ORB::ScoreType)input.scoreType,
        input.patchSize,
        input.fastThreshold
    );

    s_matcher = cv::BFMatcher::create(cv::NORM_HAMMING);

    if (s_smart_search)
    {
        smart_search(input);
    }
    else
    {
        single_search(input);
    }

    write_homography(s_output_file, s_homography);

    if (s_verbose && s_smart_search)
    {
        std::cout << "\n====================================================\n";
        std::cout << "Final Homography\n";
        prettyPrintHomography(s_homography);
        std::cout << "\n ORB Params\n";
        std::cout << "     nfeatures: " << input.nfeatures << "\n";
        std::cout << "   scaleFactor: " << input.scaleFactor << "\n";
        std::cout << "       nlevels: " << input.nlevels << "\n";
        std::cout << " edgeThreshold: " << input.edgeThreshold << "\n";
        std::cout << "    firstLevel: " << input.firstLevel << "\n";
        std::cout << "     scoreType: " << input.scoreType << "\n";
        std::cout << "     patchSize: " << input.patchSize << "\n";
        std::cout << " fastThreshold: " << input.fastThreshold << "\n";
        std::cout << "====================================================\n";
    }

    return 0;
}

void calc_homography(ORBInput& input)
{
    thread_local std::vector<cv::Point2f> points1, points2;

    {
        thread_local std::vector<cv::KeyPoint> keypoints1, keypoints2;
        thread_local std::vector<cv::DMatch> matches;

        s_mutex.lock();

        // set the orb parameters
        s_orb->setMaxFeatures(input.nfeatures);
        s_orb->setScaleFactor(input.scaleFactor);
        s_orb->setEdgeThreshold(input.edgeThreshold);
        s_orb->setScoreType((cv::ORB::ScoreType)input.scoreType);
        s_orb->setPatchSize(input.patchSize);
        s_orb->setFastThreshold(input.fastThreshold);

        {
            thread_local cv::Mat descriptors1, descriptors2;

            // detect keypoints and compute descriptors
            s_orb->detectAndCompute(s_image1, s_image1_mask, keypoints1, descriptors1);
            s_orb->detectAndCompute(s_image2, s_image2_mask, keypoints2, descriptors2);

            // Match descriptors
            s_matcher->match(descriptors1, descriptors2, matches);
        }
        s_mutex.unlock();

        // Sort matches by distance
        std::sort(matches.begin(), matches.end());

        // Keep only the best matches
        thread_local const size_t num_matches = input.maxMatches > matches.size() ? matches.size() : input.maxMatches;

        points1.resize(num_matches);
        points2.resize(num_matches);

        // Convert keypoints to points
        for (size_t i = 0; i < num_matches; i++)
        {
            points1[i] = keypoints1[matches[i].queryIdx].pt;
            points2[i] = keypoints2[matches[i].trainIdx].pt;
        }
    }

    if (points1.size() < 4 || points2.size() < 4)
    {
        // Set homography to an empty matrix
        input.H = cv::Mat();
        return;
    }

    // Find homography
    s_mutex.lock();
    try
    {
        input.H = cv::findHomography(points1, points2, cv::RANSAC, input.ransacReprojThreshold, cv::noArray(), input.ransacIters, input.confidence);
    }
    catch (cv::Exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "            nfeatures: " << input.nfeatures << "\n";
        std::cerr << "          scaleFactor: " << input.scaleFactor << "\n";
        std::cerr << "              nlevels: " << input.nlevels << "\n";
        std::cerr << "        edgeThreshold: " << input.edgeThreshold << "\n";
        std::cerr << "           firstLevel: " << input.firstLevel << "\n";
        std::cerr << "            scoreType: " << input.scoreType << "\n";
        std::cerr << "            patchSize: " << input.patchSize << "\n";
        std::cerr << "        fastThreshold: " << input.fastThreshold << "\n";
        std::cout << "          ransacIters: " << input.ransacIters << "\n";
        std::cout << "           maxMatches: " << input.maxMatches << "\n";
        std::cout << "ransacReprojThreshold: " << input.ransacReprojThreshold << "\n";
        std::cout << "           confidence: " << input.confidence << "\n\n";
        std::cerr << "              points1: " << points1.size() << "\n";
        std::cerr << "              points2: " << points2.size() << "\n\n";
        input.H = cv::Mat::zeros(3, 3, CV_64F);
        input.H.at<double>(2,2) = 1.0;
    }
    s_mutex.unlock();

}

void write_homography(const std::string& filename, const cv::Mat& H)
{
    const char* file = filename.c_str();

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

    // close hdf5 file
    H5Fclose(file_id);
}

void single_search(ORBInput& input)
{
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;
    if (s_verbose)
    {
        if (s_verbose)
        {
            std::cout << "\n====================================================\n";
            std::cout << "ORB single search\n";
            std::cout << " Calulating homographies: 1\n\n";
        }
    }

    t1 = std::chrono::high_resolution_clock::now();
    calc_homography(input);
    t2 = std::chrono::high_resolution_clock::now();
    s_homography = input.H;

    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    if (s_verbose)
    {
        std::cout << " ... Done " << time_span.count() << "[s]\n";
        std::cout << " Homography\n";
        prettyPrintHomography(input.H);
        std::cout << "\n ORB Params\n";
        std::cout << "     nfeatures: " << input.nfeatures << "\n";
        std::cout << "   scaleFactor: " << input.scaleFactor << "\n";
        std::cout << "       nlevels: " << input.nlevels << "\n";
        std::cout << " edgeThreshold: " << input.edgeThreshold << "\n";
        std::cout << "    firstLevel: " << input.firstLevel << "\n";
        std::cout << "     scoreType: " << input.scoreType << "\n";
        std::cout << "     patchSize: " << input.patchSize << "\n";
        std::cout << " fastThreshold: " << input.fastThreshold << "\n";
        std::cout << "====================================================\n";
    }

}

void smart_search(ORBInput& input)
{
    // How many threads to use
    const int nthreads = s_nthreads;
    int iteration = 0;
    size_t best_index = 0;
    std::vector<std::thread> threads(nthreads);
    std::vector<ORBInput> inputs(nthreads);

    // Copy first input to first inputs element
    inputs[0] = input;
    std::chrono::high_resolution_clock::time_point t1;
    std::chrono::high_resolution_clock::time_point t2;
    std::chrono::duration<double> time_span;

    do
    {
        // evolve the rest of the inputs
        evolve_inputs(inputs);

        if (s_verbose)
        {
            std::cout << "\n====================================================\n";
            std::cout << "ORB smart search iteration: " << iteration+1 << " | " << s_max_iterations << "\n";
            std::cout << " Calulating homographies: " << nthreads << "\n\n";
        }
        t1 = std::chrono::high_resolution_clock::now();
        // Start threads
        for (int i = 0; i < nthreads; i++)
            threads[i] = std::thread(calc_homography, std::ref(inputs[i]));

        // Join threads
        for (int i = 0; i < nthreads; i++)
            threads[i].join();

        t2 = std::chrono::high_resolution_clock::now();
        time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

        // Find best homography
        best_index = best_homography(inputs);

        // Copy best input to first inputs element
        inputs[0] = inputs[best_index];
        if (s_verbose)
        {
            std::cout << " ... Done " << time_span.count() << "[s]\n";
            std::cout << " Best Homography\n";
            prettyPrintHomography(inputs[0].H);
            std::cout << "\n Best ORB Params\n";
            std::cout << "     nfeatures: " << inputs[0].nfeatures << "\n";
            std::cout << "   scaleFactor: " << inputs[0].scaleFactor << "\n";
            std::cout << "       nlevels: " << inputs[0].nlevels << "\n";
            std::cout << " edgeThreshold: " << inputs[0].edgeThreshold << "\n";
            std::cout << "    firstLevel: " << inputs[0].firstLevel << "\n";
            std::cout << "     scoreType: " << inputs[0].scoreType << "\n";
            std::cout << "     patchSize: " << inputs[0].patchSize << "\n";
            std::cout << " fastThreshold: " << inputs[0].fastThreshold << "\n";
            std::cout << "====================================================\n";
            if ((iteration+1) < s_max_iterations)
            {
                std::cout << "Modyfing ORB parameters for next gen|\n";
                std::cout << "====================================+\n";
            }
        }
        iteration += 1;

    } while (iteration < s_max_iterations);

    // Copy best homography to s_homography
    s_homography = inputs[best_index].H;
    input = inputs[best_index];
}


void evolve_inputs(std::vector<ORBInput>& inputs)
{
    // Merseene Twister random number generators
    std::random_device rd;
    std::mt19937 gen(rd());

    // integer range [-5, 5]
    std::uniform_int_distribution<int> int_dist(-20, 20);

    // double range [-0.005, 0.005]
    std::uniform_real_distribution<double> double_dist3(-0.005, 0.005);

    // double range [-0.05, 0.05]
    std::uniform_real_distribution<double> double_dist(-0.05, 0.05);

    // double range [-0.5, 0.5]
    std::uniform_real_distribution<double> double_dist2(-0.5, 0.5);

    // integer range [-5000, 5000]
    std::uniform_int_distribution<int> int_dist2(-5000, 5000);

    // integer range [-200, 200]
    std::uniform_int_distribution<int> int_dist3(-200, 200);

    // The first element is the best input, do not modify it

    // Evolve the rest of the inputs
    for (size_t i = 1; i < inputs.size(); i++)
    {
        // Evolve nfeatures
        inputs[i].nfeatures = inputs[0].nfeatures + int_dist3(gen);
        inputs[i].nfeatures = inputs[i].nfeatures < 100 ? 100 : inputs[i].nfeatures;

        // Evolve scaleFactor, modify by +- 0.05, minimum 1.2, maximum 2.5
        inputs[i].scaleFactor = inputs[0].scaleFactor + double_dist(gen);
        inputs[i].scaleFactor = inputs[i].scaleFactor < 1.2 ? 1.2 : inputs[i].scaleFactor;
        inputs[i].scaleFactor = inputs[i].scaleFactor > 2.5 ? 2.5 : inputs[i].scaleFactor;

        // Evolve edgeThreshold, apply same difference to patchSize, minimum 2
        const int diff = int_dist(gen);
        inputs[i].edgeThreshold = inputs[0].edgeThreshold + diff;
        inputs[i].patchSize = inputs[0].patchSize + diff;
        inputs[i].edgeThreshold = inputs[i].edgeThreshold < 2 ? 2 : inputs[i].edgeThreshold;
        inputs[i].patchSize = inputs[i].patchSize < 2 ? 2 : inputs[i].patchSize;

        // do not evolve firstLevel
        //inputs[i].firstLevel = inputs[0].firstLevel + (rand() % 2 == 0 ? 1 : -1) * (rand() % 2 + 1);

        // do not Evolve scoreType
        //inputs[i].scoreType = inputs[0].scoreType + (rand() % 2 == 0 ? 1 : -1) * (rand() % 2 + 1);

        // Evolve fastThreshold by +- 5, minimum 5
        inputs[i].fastThreshold = inputs[0].fastThreshold + int_dist(gen);
        inputs[i].fastThreshold = inputs[i].fastThreshold < 5 ? 5 : inputs[i].fastThreshold;

        // Evolve ransacIters, add or subtract a random number -5000 to 5000, minimum 200
        inputs[i].ransacIters = inputs[0].ransacIters + int_dist2(gen);
        inputs[i].ransacIters = inputs[i].ransacIters < 200 ? 200 : inputs[i].ransacIters;

        // Evolve maxMatches, +- 200, minimum 100
        inputs[i].maxMatches = inputs[0].maxMatches + int_dist3(gen);
        inputs[i].maxMatches = inputs[i].maxMatches < 100 ? 100 : inputs[i].maxMatches;

        // Evolve ransacReprojThreshold, +- 0.5, minimum 1.5, maximum 10.0
        inputs[i].ransacReprojThreshold = inputs[0].ransacReprojThreshold + double_dist2(gen);
        inputs[i].ransacReprojThreshold = inputs[i].ransacReprojThreshold < 1.5 ? 1.5 : inputs[i].ransacReprojThreshold;
        inputs[i].ransacReprojThreshold = inputs[i].ransacReprojThreshold > 10.0 ? 10.0 : inputs[i].ransacReprojThreshold;

        // Evolve confidence, +- 0.005, minimum 0.85 maximum 0.999
        inputs[i].confidence = inputs[0].confidence + double_dist3(gen);
        inputs[i].confidence = inputs[i].confidence < 0.85 ? 0.85 : inputs[i].confidence;
        inputs[i].confidence = inputs[i].confidence > 0.999 ? 0.999 : inputs[i].confidence;

        // Update homgraphy to zeros
        inputs[i].H = cv::Mat::zeros(3, 3, CV_64F);

        // Copy the remaining fields from the first input
        inputs[i].firstLevel = inputs[0].firstLevel;
        inputs[i].scoreType = inputs[0].scoreType;
        inputs[i].nlevels = inputs[0].nlevels;
    }

    // Update the last input to massive changes
    
    // nfeatures, update by [-20000,20000], minimum 500
    std::uniform_int_distribution<int> int_dist4(-20000, 20000);
    inputs[inputs.size() - 1].nfeatures = int_dist4(gen);
    inputs[inputs.size() - 1].nfeatures = inputs[inputs.size() - 1].nfeatures < 500 ? 500 : inputs[inputs.size() - 1].nfeatures;

    // edgeThreshold && patchSize, update by [-200,200], minimum 2
    std::uniform_int_distribution<int> int_dist5(-200, 200);
    int diff = int_dist5(gen);
    inputs[inputs.size() - 1].edgeThreshold = inputs[0].edgeThreshold + diff;
    inputs[inputs.size() - 1].patchSize = inputs[0].patchSize + diff;
    inputs[inputs.size() - 1].edgeThreshold = inputs[inputs.size() - 1].edgeThreshold < 2 ? 2 : inputs[inputs.size() - 1].edgeThreshold;
    inputs[inputs.size() - 1].patchSize = inputs[inputs.size() - 1].patchSize < 2 ? 2 : inputs[inputs.size() - 1].patchSize;

    // ransac iters, update by [0-20000]
    inputs[inputs.size() - 1].ransacIters = inputs[inputs.size() - 1].ransacIters + int_dist4(gen);
}



// A homography matrix is a 3x3 matrix of doubles
// | a b c |
// | d e f |
// | g h i |
// {a} is the scaling in x-direction
// {e} is the scaling in y-direction
// {b} is the shearing in x-direction
// {d} is the shearing in y-direction


// inline score functions, all functions should accept a cv::Mat and 4 doubles, and return a double
// One of these functions will be assigned to a lambda function in best_homography function
inline double score_dx(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx);
}
inline double score_dy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    return std::fabs(H.at<double>(1, 1) - dy);
}
inline double score_sx(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    return std::fabs(H.at<double>(0, 1) - sx);
}
inline double score_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(1, 0) - sy);
}
inline double score_dx_dy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx) + std::fabs(H.at<double>(1, 1) - dy);
}
inline double score_dx_sx(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx) + std::fabs(H.at<double>(0, 1) - sx);
}
inline double score_dx_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx) + std::fabs(H.at<double>(1, 0) - sy);
}
inline double score_dy_sx(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    return std::fabs(H.at<double>(1, 1) - dy) + std::fabs(H.at<double>(0, 1) - sx);
}
inline double score_dy_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(1, 1) - dy) + std::fabs(H.at<double>(1, 0) - sy);
}
inline double score_sx_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0) 
{
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(0, 1) - sx) + std::fabs(H.at<double>(1, 0) - sy);
}
inline double score_dx_dy_sx(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx) + std::fabs(H.at<double>(1, 1) - dy) + std::fabs(H.at<double>(0, 1) - sx);
}
inline double score_dx_dy_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx) + std::fabs(H.at<double>(1, 1) - dy) + std::fabs(H.at<double>(1, 0) - sy);
}
inline double score_dx_sx_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx) + std::fabs(H.at<double>(0, 1) - sx) + std::fabs(H.at<double>(1, 0) - sy);
}
inline double score_dy_sx_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(1, 1) - dy) + std::fabs(H.at<double>(0, 1) - sx) + std::fabs(H.at<double>(1, 0) - sy);
}
inline double score_dx_dy_sx_sy(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0)
{
    // dx is the maximum allowed scaling in x-direction
    // this value is stored at position (0, 0) in the homography matrix
    // dy is the maximum allowed scaling in y-direction
    // this value is stored at position (1, 1) in the homography matrix
    // sx is the maximum allowed shearing in x-direction
    // this value is stored at position (0, 1) in the homography matrix
    // sy is the maximum allowed shearing in y-direction
    // this value is stored at position (1, 0) in the homography matrix
    return std::fabs(H.at<double>(0, 0) - dx) + std::fabs(H.at<double>(1, 1) - dy) + std::fabs(H.at<double>(0, 1) - sx) + std::fabs(H.at<double>(1, 0) - sy);
}
inline double default_score(const cv::Mat& H, double dx=0.0, double dy=0.0, double sx=0.0, double sy=0.0) { return 0.0; }


int best_homography(const std::vector<ORBInput>& homographies)
{
    const double dx = s_max_x_scaling;
    const double dy = s_max_y_scaling;
    const double sx = s_max_x_shearing;
    const double sy = s_max_y_shearing;

    // if all the limits are still infinity, return the first homography
    if (
        dx == std::numeric_limits<double>::infinity() &&
        dy == std::numeric_limits<double>::infinity() &&
        sx == std::numeric_limits<double>::infinity() &&
        sy == std::numeric_limits<double>::infinity()
    )
    {
        return 0;
    }

    // at least one of the limits is not infinity, find the best homography
    // 15 options for which limits to use
    // Create binary representation of the limits
    // XXXX, 0 = infinity, 1 = limit
    uint8_t limits = 0;
    
    limits |=
        (dx == std::numeric_limits<double>::infinity() ? 0 : 1) << 3 |
        (dy == std::numeric_limits<double>::infinity() ? 0 : 1) << 2 |
        (sx == std::numeric_limits<double>::infinity() ? 0 : 1) << 1 |
        (sy == std::numeric_limits<double>::infinity() ? 0 : 1);

    // Possible options for the limits
    // dx only = 1000
    // dy only = 0100
    // sx only = 0010
    // sy only = 0001
    // ...


    // Use a switch case to choose function to use
    std::function<double(const cv::Mat&, double, double, double, double)> score;

    switch (limits)
    {
        case 0b1000:
            // dx only
            score = score_dx;
            break;

        case 0b0100:
            // dy only
            score = score_dy;
            break;
        
        case 0b0010:
            // sx only
            score = score_sx;
            break;
        
        case 0b0001:
            // sy only
            score = score_sy;
            break;
        
        case 0b1100:
            // dx and dy
            score = score_dx_dy;
            break;
        
        case 0b1010:
            // dx and sx
            score = score_dx_sx;
            break;
        
        case 0b1001:
            // dx and sy
            score = score_dx_sy;
            break;
        
        case 0b0110:
            // dy and sx
            score = score_dy_sx;
            break;
        
        case 0b0101:
            // dy and sy
            score = score_dy_sy;
            break;
        
        case 0b0011:
            // sx and sy
            score = score_sx_sy;
            break;
        
        case 0b1110:
            // dx, dy and sx
            score = score_dx_dy_sx;
            break;
        
        case 0b1101:
            // dx, dy and sy
            score = score_dx_dy_sy;
            break;
        
        case 0b1011:
            // dx, sx and sy
            score = score_dx_sx_sy;
            break;
        
        case 0b0111:
            // dy, sx and sy
            score = score_dy_sx_sy;
            break;
        
        case 0b1111:
            // dx, dy, sx and sy
            score = score_dx_dy_sx_sy;
            break;
        
        default:
            // default score function
            score = default_score;
            break;
    }

    std::vector<double> scores(homographies.size());

    // Calculate scores
    for (size_t i = 0; i < homographies.size(); i++)
    {
        // Check if homography is empty
        if (homographies[i].H.empty())
            scores[i] = std::numeric_limits<double>::infinity();
        else
            scores[i] = score(homographies[i].H, dx, dy, sx, sy);
    }

    // Find index of minimum score
    size_t min_index = 0;
    for (size_t i = 1; i < scores.size(); i++)
        if (scores[i] < scores[min_index])
            min_index = i;

    return min_index;
}


