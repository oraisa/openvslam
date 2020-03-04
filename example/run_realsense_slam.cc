#ifdef USE_PANGOLIN_VIEWER
#include "pangolin_viewer/viewer.h"
#elif USE_SOCKET_PUBLISHER
#include "socket_publisher/publisher.h"
#endif

#include "openvslam/system.h"
#include "openvslam/config.h"
#include "openvslam/util/stereo_rectifier.h"

#include <iostream>
#include <chrono>
#include <numeric>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>
#include <popl.hpp>

#include <librealsense2/rs.hpp>

#ifdef USE_STACK_TRACE_LOGGER
#include <glog/logging.h>
#endif

#ifdef USE_GOOGLE_PERFTOOLS
#include <gperftools/profiler.h>
#endif

void mono_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path,
                   const float scale, const std::string& map_db_path) {
    cv::setUseOptimized( true );

    rs2::config config;

    rs2::pipeline pipeline;
    config.enable_stream(RS2_STREAM_FISHEYE, 0);
    config.enable_stream(RS2_STREAM_FISHEYE, 1);

    rs2::pipeline_profile pipeline_profile = pipeline.start( );//config );

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup(false);

    if (!map_db_path.empty()) {
        // load the map database
        try {
            SLAM.load_map_database(map_db_path, true);
        } catch (...) {
        }
    }

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    rs2::frameset frameset;

    cv::Mat fisheye_frame;
    cv::Mat mask = {};

    double timestamp = 0.0;
    std::vector<double> track_times;

    unsigned int num_frame = 0;

    unsigned int frame_id = 0;

    // run the SLAM in another thread
    std::thread thread([&]() {
        while (true) {
            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }

            frameset = pipeline.wait_for_frames();

            if (frame_id == frameset.get_fisheye_frame().get_frame_number()) {
                continue;
            };
            frame_id = frameset.get_fisheye_frame().get_frame_number();

            fisheye_frame = cv::Mat(800, 848, CV_8U, const_cast<void*>(frameset.get_fisheye_frame().get_data()));

            timestamp = frameset.get_fisheye_frame().get_timestamp() * 0.001;
                
            if (scale != 1.0) {
                cv::resize(fisheye_frame, fisheye_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            // input the current frame and estimate the camera pose
            SLAM.feed_monocular_frame(fisheye_frame, timestamp, mask);

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            track_times.push_back(track_time);

            //timestamp += 1.0 / cfg->camera_->fps_;
            ++num_frame;
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // run the viewer in the current thread
    #ifdef USE_PANGOLIN_VIEWER
    viewer.run();
    #elif USE_SOCKET_PUBLISHER
    publisher.run();
    #endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path, true);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

}

void stereo_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path,
                   const float scale, const std::string& map_db_path) {
    cv::setUseOptimized( true );

    rs2::config config;

    rs2::pipeline pipeline;
    config.enable_stream(RS2_STREAM_FISHEYE, 0);
    config.enable_stream(RS2_STREAM_FISHEYE, 1);

    rs2::pipeline_profile pipeline_profile = pipeline.start( );//config );

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup(false);

    if (!map_db_path.empty()) {
        // load the map database
        try {
            SLAM.load_map_database(map_db_path, true);
        } catch (...) {
        }
    }

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    rs2::frameset frameset;

    cv::Mat fisheye_frame_left;
    cv::Mat fisheye_frame_right;
    cv::Mat mask = {};
    
    double timestamp = 0.0;
    std::vector<double> track_times;

    unsigned int num_frame = 0;

    unsigned int frame_id = 0;

    // run the SLAM in another thread
    std::thread thread([&]() {
        while (true) {
            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }

            frameset = pipeline.wait_for_frames();

            if (frame_id == frameset.get_fisheye_frame().get_frame_number()) {
                continue;
            };
            frame_id = frameset.get_fisheye_frame().get_frame_number();

            fisheye_frame_left = cv::Mat(800, 848, CV_8U, const_cast<void*>(frameset.get_fisheye_frame(1).get_data()));
            fisheye_frame_right = cv::Mat(800, 848, CV_8U, const_cast<void*>(frameset.get_fisheye_frame(0).get_data()));

            timestamp = frameset.get_fisheye_frame().get_timestamp() * 0.001;
                
            if (scale != 1.0) {
                cv::resize(fisheye_frame_left, fisheye_frame_left, cv::Size(), scale, scale, cv::INTER_LINEAR);
                cv::resize(fisheye_frame_right, fisheye_frame_right, cv::Size(), scale, scale, cv::INTER_LINEAR);
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            // input the current frame and estimate the camera pose
            SLAM.feed_stereo_frame(fisheye_frame_left, fisheye_frame_right, timestamp, mask);

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            track_times.push_back(track_time);

            //timestamp += 1.0 / cfg->camera_->fps_;
            ++num_frame;
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // run the viewer in the current thread
    #ifdef USE_PANGOLIN_VIEWER
    viewer.run();
    #elif USE_SOCKET_PUBLISHER
    publisher.run();
    #endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path, true);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

}

void rgbd_tracking(const std::shared_ptr<openvslam::config>& cfg,
                   const std::string& vocab_file_path,
                   const float scale, const std::string& map_db_path) {
    cv::setUseOptimized( true );

    rs2::config config;

    rs2::pipeline pipeline;
    config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 30);
    config.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);

    rs2::align align_to_color(RS2_STREAM_COLOR);

    rs2::pipeline_profile pipeline_profile = pipeline.start( config );

    // build a SLAM system
    openvslam::system SLAM(cfg, vocab_file_path);
    // startup the SLAM process
    SLAM.startup(false);

    if (!map_db_path.empty()) {
        // load the map database
        try {
            SLAM.load_map_database(map_db_path, true);
        } catch (...) {
        }
    }

    // create a viewer object
    // and pass the frame_publisher and the map_publisher
#ifdef USE_PANGOLIN_VIEWER
    pangolin_viewer::viewer viewer(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#elif USE_SOCKET_PUBLISHER
    socket_publisher::publisher publisher(cfg, &SLAM, SLAM.get_frame_publisher(), SLAM.get_map_publisher());
#endif

    rs2::frameset frameset;

    cv::Mat color_frame;
    cv::Mat depth_frame;
    cv::Mat mask = {};

    double timestamp = 0.0;
    std::vector<double> track_times;

    unsigned int num_frame = 0;

    unsigned int frame_id = 0;

    // run the SLAM in another thread
    std::thread thread([&]() {
        while (true) {
            // check if the termination of SLAM system is requested or not
            if (SLAM.terminate_is_requested()) {
                break;
            }

            frameset = align_to_color.process(pipeline.wait_for_frames());

            if (frame_id == frameset.get_color_frame().get_frame_number()) {
                continue;
            };
            frame_id = frameset.get_color_frame().get_frame_number();

            color_frame = cv::Mat(720, 1280, CV_8UC3, const_cast<void*>(frameset.get_color_frame().get_data()));
            depth_frame = cv::Mat(720, 1280, CV_16UC1, const_cast<void*>(frameset.get_depth_frame().get_data()));

            if (frameset.get_color_frame().get_timestamp() > frameset.get_depth_frame().get_timestamp()) {
                timestamp = frameset.get_color_frame().get_timestamp() * 0.001;
            } else {
                timestamp = frameset.get_depth_frame().get_timestamp() * 0.001;
            }

            if (scale != 1.0) {
                cv::resize(color_frame, color_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
                cv::resize(depth_frame, depth_frame, cv::Size(), scale, scale, cv::INTER_LINEAR);
            }

            const auto tp_1 = std::chrono::steady_clock::now();

            // input the current frame and estimate the camera pose
            SLAM.feed_RGBD_frame(color_frame, depth_frame, timestamp, mask);

            const auto tp_2 = std::chrono::steady_clock::now();

            const auto track_time = std::chrono::duration_cast<std::chrono::duration<double>>(tp_2 - tp_1).count();
            track_times.push_back(track_time);

            //timestamp += 1.0 / cfg->camera_->fps_;
            ++num_frame;
        }

        // wait until the loop BA is finished
        while (SLAM.loop_BA_is_running()) {
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
        }
    });

    // run the viewer in the current thread
    #ifdef USE_PANGOLIN_VIEWER
    viewer.run();
    #elif USE_SOCKET_PUBLISHER
    publisher.run();
    #endif

    thread.join();

    // shutdown the SLAM process
    SLAM.shutdown();

    if (!map_db_path.empty()) {
        // output the map database
        SLAM.save_map_database(map_db_path, true);
    }

    std::sort(track_times.begin(), track_times.end());
    const auto total_track_time = std::accumulate(track_times.begin(), track_times.end(), 0.0);
    std::cout << "median tracking time: " << track_times.at(track_times.size() / 2) << "[s]" << std::endl;
    std::cout << "mean tracking time: " << total_track_time / track_times.size() << "[s]" << std::endl;

}

int main(int argc, char* argv[]) {
#ifdef USE_STACK_TRACE_LOGGER
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif

    // create options
    popl::OptionParser op("Allowed options");
    auto help = op.add<popl::Switch>("h", "help", "produce help message");
    auto vocab_file_path = op.add<popl::Value<std::string>>("v", "vocab", "vocabulary file path");
    auto config_file_path = op.add<popl::Value<std::string>>("c", "config", "config file path");
    auto scale = op.add<popl::Value<float>>("s", "scale", "scaling ratio of images", 1.0);
    auto map_db_path = op.add<popl::Value<std::string>>("p", "map-db", "store a map database at this path after SLAM", "");
    auto debug_mode = op.add<popl::Switch>("", "debug", "debug mode");
    try {
        op.parse(argc, argv);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // check validness of options
    if (help->is_set()) {
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }
    if (!vocab_file_path->is_set() || !config_file_path->is_set()) {
        std::cerr << "invalid arguments" << std::endl;
        std::cerr << std::endl;
        std::cerr << op << std::endl;
        return EXIT_FAILURE;
    }

    // setup logger
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] %^[%L] %v%$");
    if (debug_mode->is_set()) {
        spdlog::set_level(spdlog::level::debug);
    }
    else {
        spdlog::set_level(spdlog::level::info);
    }

    // load configuration
    std::shared_ptr<openvslam::config> cfg;
    try {
        cfg = std::make_shared<openvslam::config>(config_file_path->value());
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStart("slam.prof");
#endif

    //cfg->camera_->model_->fx_ *= scale->value();
    //cfg->camera_->model_->fy_ *= scale->value();
    //cfg->camera_->model_->cx_ *= scale->value();
    //cfg->camera_->model_->fy_ *= scale->value();

    // run tracking
    if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::RGBD) {
        rgbd_tracking(cfg, vocab_file_path->value(), scale->value(), map_db_path->value());
    } else if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Monocular) {
        mono_tracking(cfg, vocab_file_path->value(), scale->value(), map_db_path->value());
    } else if (cfg->camera_->setup_type_ == openvslam::camera::setup_type_t::Stereo) {
        stereo_tracking(cfg, vocab_file_path->value(), scale->value(), map_db_path->value());
    }
    else {
        throw std::runtime_error("Invalid setup type: " + cfg->camera_->get_setup_type_string());
    }

#ifdef USE_GOOGLE_PERFTOOLS
    ProfilerStop();
#endif

    return EXIT_SUCCESS;
}
