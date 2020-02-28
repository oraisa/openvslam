#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/camera_database.h"
#include "openvslam/data/bow_database.h"
#include "openvslam/data/map_database.h"
#include "openvslam/io/map_database_io.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace openvslam {
namespace io {

map_database_io::map_database_io(data::camera_database* cam_db, data::map_database* map_db,
                                 data::bow_database* bow_db, data::bow_vocabulary* bow_vocab)
    : cam_db_(cam_db), map_db_(map_db), bow_db_(bow_db), bow_vocab_(bow_vocab) {}

void map_database_io::save_message_pack(const std::string& path) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    assert(cam_db_ && map_db_);
    const auto cameras = cam_db_->to_json();
    nlohmann::json keyfrms;
    nlohmann::json landmarks;
    map_db_->to_json(keyfrms, landmarks);

    nlohmann::json json{{"cameras", cameras},
                        {"keyframes", keyfrms},
                        {"landmarks", landmarks},
                        {"frame_next_id", static_cast<unsigned int>(data::frame::next_id_)},
                        {"keyframe_next_id", static_cast<unsigned int>(data::keyframe::next_id_)},
                        {"landmark_next_id", static_cast<unsigned int>(data::landmark::next_id_)}};

    std::ofstream ofs(path, std::ios::out | std::ios::binary);

    if (ofs.is_open()) {
        spdlog::info("save the MessagePack file of database to {}", path);
        const auto msgpack = nlohmann::json::to_msgpack(json);
        ofs.write(reinterpret_cast<const char*>(msgpack.data()), msgpack.size() * sizeof(uint8_t));
        ofs.close();
    }
    else {
        spdlog::critical("cannot create a file at {}", path);
    }
}

void map_database_io::load_message_pack(const std::string& path) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    // 1. initialize database

    assert(cam_db_ && map_db_ && bow_db_ && bow_vocab_);
    map_db_->clear();
    bow_db_->clear();

    // 2. load binary bytes

    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::critical("cannot load the file at {}", path);
        throw std::runtime_error("cannot load the file at " + path);
    }

    spdlog::info("load the MessagePack file of database from {}", path);
    std::vector<uint8_t> msgpack;
    while (true) {
        uint8_t buffer;
        ifs.read(reinterpret_cast<char*>(&buffer), sizeof(uint8_t));
        if (ifs.eof()) {
            break;
        }
        msgpack.push_back(buffer);
    }
    ifs.close();

    // 3. parse into JSON

    const auto json = nlohmann::json::from_msgpack(msgpack);

    // 4. load database

    // load static variables
    data::frame::next_id_ = json.at("frame_next_id").get<unsigned int>();
    data::keyframe::next_id_ = json.at("keyframe_next_id").get<unsigned int>();
    data::landmark::next_id_ = json.at("landmark_next_id").get<unsigned int>();
    // load database
    const auto json_cameras = json.at("cameras");
    cam_db_->from_json(json_cameras);
    const auto json_keyfrms = json.at("keyframes");
    const auto json_landmarks = json.at("landmarks");
    map_db_->from_json(cam_db_, bow_vocab_, bow_db_, json_keyfrms, json_landmarks);
    const auto keyfrms = map_db_->get_all_keyframes();
    for (const auto keyfrm : keyfrms) {
        bow_db_->add_keyframe(keyfrm);
    }
}

void map_database_io::save_binary_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
    assert(cam_db_ && map_db_);
    const auto cameras = cam_db_->to_buffer();
    std::vector<data::keyframe::keyframe_data> keyframes;
    std::vector<data::landmark::landmark_data> landmarks;
    map_db_->to_buffer(keyframes, landmarks);
    std::ofstream ofs(path, std::ios::out | std::ios::binary | std::ios::trunc);

    if (ofs.is_open()) {

        std::map<std::string, int> camera_ids;

        uint32_t frame_next_id = data::frame::next_id_;
        uint32_t keyframe_next_id = data::keyframe::next_id_;
        uint32_t landmark_next_id = data::landmark::next_id_;

        ofs.write(reinterpret_cast<const char *>(&frame_next_id), sizeof(frame_next_id));
        ofs.write(reinterpret_cast<const char *>(&keyframe_next_id), sizeof(keyframe_next_id));
        ofs.write(reinterpret_cast<const char *>(&landmark_next_id), sizeof(landmark_next_id));

        uint32_t camera_count = cameras.size();

        ofs.write(reinterpret_cast<const char *>(&camera_count), sizeof(camera_count));

        for (auto& camera : cameras) {
            auto color_order = camera::color_order_to_string[(int) camera.color_order];
            auto model_type = camera::model_type_to_string[(int) camera.model_type];
            auto setup_type = camera::setup_type_to_string[(int) camera.setup_type];

            camera_ids[camera.camera_name] = camera_ids.size();

            ofs.write(camera.camera_name.c_str(), camera.camera_name.size() + 1);
            ofs.write(model_type.c_str(), model_type.size() + 1);
            ofs.write(setup_type.c_str(), setup_type.size() + 1);
            ofs.write(color_order.c_str(), color_order.size() + 1);

            ofs.write(reinterpret_cast<const char *>(&camera.rows), sizeof(camera.rows));
            ofs.write(reinterpret_cast<const char *>(&camera.cols), sizeof(camera.cols));
            ofs.write(reinterpret_cast<const char *>(&camera.fps), sizeof(camera.fps));
            switch (camera.model_type) {
                case camera::model_type_t::Perspective: {
                    ofs.write(reinterpret_cast<const char *>(&camera.perspective), sizeof(camera.perspective));
                    break;
                }
                case camera::model_type_t::Fisheye: {
                    ofs.write(reinterpret_cast<const char *>(&camera.fisheye), sizeof(camera.fisheye));
                    break;
                }
                case camera::model_type_t::Equirectangular: {
                    ofs.write(reinterpret_cast<const char *>(&camera.equirectangular), sizeof(camera.equirectangular));
                    break;
                }
            }
        }

        uint32_t keyframe_count = keyframes.size();

        ofs.write(reinterpret_cast<const char *>(&keyframe_count), sizeof(keyframe_count));

        for (auto& keyframe : keyframes) {
            unsigned int camera_id = camera_ids[keyframe.camera_name];

            ofs.write(reinterpret_cast<const char *>(&keyframe.id), sizeof(keyframe.id));
            ofs.write(reinterpret_cast<const char *>(&keyframe.depth_thr), sizeof(keyframe.depth_thr));
            ofs.write(reinterpret_cast<const char *>(&keyframe.n_keypoints), sizeof(keyframe.n_keypoints));
            ofs.write(reinterpret_cast<const char *>(&keyframe.num_scale_levels), sizeof(keyframe.num_scale_levels));
            ofs.write(reinterpret_cast<const char *>(&keyframe.scale_factor), sizeof(keyframe.scale_factor));
            ofs.write(reinterpret_cast<const char *>(&keyframe.source_frame_id), sizeof(keyframe.source_frame_id));
            ofs.write(reinterpret_cast<const char *>(&keyframe.span_parent), sizeof(keyframe.span_parent));
            ofs.write(reinterpret_cast<const char *>(&keyframe.timestamp), sizeof(keyframe.timestamp));

            ofs.write(reinterpret_cast<const char *>(&keyframe.rot_x), sizeof(keyframe.rot_x));
            ofs.write(reinterpret_cast<const char *>(&keyframe.rot_y), sizeof(keyframe.rot_y));
            ofs.write(reinterpret_cast<const char *>(&keyframe.rot_z), sizeof(keyframe.rot_z));
            ofs.write(reinterpret_cast<const char *>(&keyframe.rot_w), sizeof(keyframe.rot_w));
            ofs.write(reinterpret_cast<const char *>(&keyframe.trans_x), sizeof(keyframe.trans_x));
            ofs.write(reinterpret_cast<const char *>(&keyframe.trans_y), sizeof(keyframe.trans_y));
            ofs.write(reinterpret_cast<const char *>(&keyframe.trans_z), sizeof(keyframe.trans_z));

            ofs.write(reinterpret_cast<const char *>(&camera_id), sizeof(camera_id));

            uint32_t depth_count = keyframe.depths.size();
            ofs.write(reinterpret_cast<const char *>(&depth_count), sizeof(depth_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.depths[0]), sizeof(keyframe.depths[0])*depth_count);

            uint32_t descriptor_count = keyframe.descriptors.size();
            ofs.write(reinterpret_cast<const char *>(&descriptor_count), sizeof(descriptor_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.descriptors[0]), sizeof(keyframe.descriptors[0])*descriptor_count);

            uint32_t landmark_count = keyframe.landmark_ids.size();
            ofs.write(reinterpret_cast<const char *>(&landmark_count), sizeof(landmark_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.landmark_ids[0]), sizeof(keyframe.landmark_ids[0])*landmark_count);

            uint32_t loop_edge_count = keyframe.loop_edge_ids.size();
            ofs.write(reinterpret_cast<const char *>(&loop_edge_count), sizeof(loop_edge_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.loop_edge_ids[0]), sizeof(keyframe.loop_edge_ids[0])*loop_edge_count);

            uint32_t spanning_child_count = keyframe.spanning_child_ids.size();
            ofs.write(reinterpret_cast<const char *>(&spanning_child_count), sizeof(spanning_child_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.spanning_child_ids[0]), sizeof(keyframe.spanning_child_ids[0])*spanning_child_count);

            uint32_t x_right_count = keyframe.x_rights.size();
            ofs.write(reinterpret_cast<const char *>(&x_right_count), sizeof(x_right_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.x_rights[0]), sizeof(keyframe.x_rights[0])*x_right_count);

            uint32_t keypoint_count = keyframe.keypoints.size();
            ofs.write(reinterpret_cast<const char *>(&keypoint_count), sizeof(keypoint_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.keypoints[0]), sizeof(keyframe.keypoints[0])*keypoint_count);

            uint32_t undistorted_keypoint_count = keyframe.undistorted_keypoints.size();
            ofs.write(reinterpret_cast<const char *>(&undistorted_keypoint_count), sizeof(undistorted_keypoint_count));
            ofs.write(reinterpret_cast<const char *>(&keyframe.undistorted_keypoints[0]), sizeof(keyframe.undistorted_keypoints[0])*undistorted_keypoint_count);
        }

        uint32_t landmark_count = landmarks.size();

        ofs.write(reinterpret_cast<const char *>(&landmark_count), sizeof(landmark_count));
        ofs.write(reinterpret_cast<const char *>(&landmarks[0]), sizeof(landmarks[0])*landmark_count);

        ofs.close();
    }
    else {
        spdlog::critical("cannot create a file at {}", path);
    }
}

void map_database_io::load_binary_file(const std::string& path) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    map_db_->clear();
    bow_db_->clear();

    std::ifstream ifs(path, std::ios::in | std::ios::binary);

    if (!ifs.is_open()) {
        spdlog::critical("cannot load the file at {}", path);
        throw std::runtime_error("cannot load the file at " + path);
    }
    spdlog::info("load the binary file of database from {}", path);

    std::vector<data::keyframe::keyframe_data> keyframes;
    std::vector<data::landmark::landmark_data> landmarks;
    std::vector<data::camera_database::camera_data> cameras;

    std::map<int, std::string> camera_names;

    uint32_t frame_next_id;
    uint32_t keyframe_next_id;
    uint32_t landmark_next_id;

    ifs.read(reinterpret_cast<char *>(&frame_next_id), sizeof(frame_next_id));
    ifs.read(reinterpret_cast<char *>(&keyframe_next_id), sizeof(keyframe_next_id));
    ifs.read(reinterpret_cast<char *>(&landmark_next_id), sizeof(landmark_next_id));

    data::frame::next_id_ = frame_next_id;
    data::keyframe::next_id_ = keyframe_next_id;
    data::landmark::next_id_ = landmark_next_id;

    uint32_t camera_count;
    ifs.read(reinterpret_cast<char *>(&camera_count), sizeof(camera_count));

    for (unsigned int i = 0; i < camera_count; i++) {
        data::camera_database::camera_data d;
        std::string model_type;
        std::string setup_type;
        std::string color_order;
        std::getline(ifs, d.camera_name, '\0');
        std::getline(ifs, model_type, '\0');
        std::getline(ifs, setup_type, '\0');
        std::getline(ifs, color_order, '\0');
        d.model_type = camera::base::load_model_type(model_type);
        d.setup_type = camera::base::load_setup_type(setup_type);
        d.color_order = camera::base::load_color_order(color_order);
        ifs.read(reinterpret_cast<char *>(&d.rows), sizeof(d.rows));
        ifs.read(reinterpret_cast<char *>(&d.cols), sizeof(d.cols));
        ifs.read(reinterpret_cast<char *>(&d.fps), sizeof(d.fps));
        switch(d.model_type) {
            case camera::model_type_t::Perspective: {
                ifs.read(reinterpret_cast<char *>(&d.perspective), sizeof(d.perspective));
                break;
            }
            case camera::model_type_t::Fisheye: {
                ifs.read(reinterpret_cast<char *>(&d.fisheye), sizeof(d.fisheye));
                break;
            }
            case camera::model_type_t::Equirectangular: {
                ifs.read(reinterpret_cast<char *>(&d.equirectangular), sizeof(d.equirectangular));
                break;
            }
        }
        cameras.push_back(d);
        camera_names[camera_names.size()] = d.camera_name;
    }

    uint32_t keyframe_count;
    ifs.read(reinterpret_cast<char *>(&keyframe_count), sizeof(keyframe_count));

    for (unsigned int i = 0; i < keyframe_count; i++) {
        data::keyframe::keyframe_data d;
        ifs.read(reinterpret_cast<char *>(&d.id), sizeof(d.id));
        ifs.read(reinterpret_cast<char *>(&d.depth_thr), sizeof(d.depth_thr));
        ifs.read(reinterpret_cast<char *>(&d.n_keypoints), sizeof(d.n_keypoints));
        ifs.read(reinterpret_cast<char *>(&d.num_scale_levels), sizeof(d.num_scale_levels));
        ifs.read(reinterpret_cast<char *>(&d.scale_factor), sizeof(d.scale_factor));
        ifs.read(reinterpret_cast<char *>(&d.source_frame_id), sizeof(d.source_frame_id));
        ifs.read(reinterpret_cast<char *>(&d.span_parent), sizeof(d.span_parent));
        ifs.read(reinterpret_cast<char *>(&d.timestamp), sizeof(d.timestamp));

        ifs.read(reinterpret_cast<char *>(&d.rot_x), sizeof(d.rot_x));
        ifs.read(reinterpret_cast<char *>(&d.rot_y), sizeof(d.rot_y));
        ifs.read(reinterpret_cast<char *>(&d.rot_z), sizeof(d.rot_z));
        ifs.read(reinterpret_cast<char *>(&d.rot_w), sizeof(d.rot_w));
        ifs.read(reinterpret_cast<char *>(&d.trans_x), sizeof(d.trans_x));
        ifs.read(reinterpret_cast<char *>(&d.trans_y), sizeof(d.trans_y));
        ifs.read(reinterpret_cast<char *>(&d.trans_z), sizeof(d.trans_z));

        uint32_t camera_id;
        ifs.read(reinterpret_cast<char *>(&camera_id), sizeof(camera_id));


        d.camera_name = camera_names[camera_id];

        uint32_t depth_count;
        ifs.read(reinterpret_cast<char *>(&depth_count), sizeof(depth_count));
        d.depths.reserve(depth_count);
        for (unsigned int j = 0; j < depth_count; j++) {
            float v;
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            d.depths.push_back(v);
        }

        uint32_t descriptor_count;
        ifs.read(reinterpret_cast<char *>(&descriptor_count), sizeof(descriptor_count));
        d.descriptors.reserve(descriptor_count);
        for (unsigned int j = 0; j < descriptor_count; j++) {
            std::array<uint32_t, 8> v = {};
            ifs.read(reinterpret_cast<char *>(&v), sizeof(uint32_t)*8);
            d.descriptors.push_back(v);
        }

        uint32_t landmark_count;
        ifs.read(reinterpret_cast<char *>(&landmark_count), sizeof(landmark_count));
        d.landmark_ids.reserve(landmark_count);
        for (unsigned int j = 0; j < landmark_count; j++) {
            int v;
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            d.landmark_ids.push_back(v);
        }

        uint32_t loop_edge_count;
        ifs.read(reinterpret_cast<char *>(&loop_edge_count), sizeof(loop_edge_count));
        d.loop_edge_ids.reserve(loop_edge_count);
        for (unsigned int j = 0; j < loop_edge_count; j++) {
            int v;
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            d.loop_edge_ids.push_back(v);
        }

        uint32_t spanning_child_count;
        ifs.read(reinterpret_cast<char *>(&spanning_child_count), sizeof(spanning_child_count));
        d.spanning_child_ids.reserve(spanning_child_count);
        for (unsigned int j = 0; j < spanning_child_count; j++) {
            int v;
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            d.spanning_child_ids.push_back(v);
        }

        uint32_t x_right_count;
        ifs.read(reinterpret_cast<char *>(&x_right_count), sizeof(x_right_count));
        d.x_rights.reserve(x_right_count);
        for (unsigned int j = 0; j < x_right_count; j++) {
            float v;
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            d.x_rights.push_back(v);
        }

        uint32_t keypoint_count;
        ifs.read(reinterpret_cast<char *>(&keypoint_count), sizeof(keypoint_count));
        d.keypoints.reserve(keypoint_count);
        for (unsigned int j = 0; j < keypoint_count; j++) {
            data::keyframe::keypoint_data v = {};
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            d.keypoints.push_back(v);
        }

        uint32_t undistorted_keypoint_count;
        ifs.read(reinterpret_cast<char *>(&undistorted_keypoint_count), sizeof(undistorted_keypoint_count));
        d.undistorted_keypoints.reserve(undistorted_keypoint_count);
        for (unsigned int j = 0; j < undistorted_keypoint_count; j++) {
            data::keyframe::undistorted_keypoint_data v = {};
            ifs.read(reinterpret_cast<char *>(&v), sizeof(v));
            d.undistorted_keypoints.push_back(v);
        }
        keyframes.push_back(d);

    }

    uint32_t landmark_count;
    ifs.read(reinterpret_cast<char *>(&landmark_count), sizeof(landmark_count));

    auto landmark_buffer = (data::landmark::landmark_data*) malloc(sizeof(data::landmark::landmark_data)*landmark_count);
    ifs.read(reinterpret_cast<char *>(landmark_buffer), sizeof(data::landmark::landmark_data)*landmark_count);
    landmarks.assign(landmark_buffer, landmark_buffer+landmark_count);
    free(landmark_buffer);

    ifs.close();

    cam_db_->from_buffer(cameras);
    map_db_->from_buffer(cam_db_, bow_vocab_, bow_db_, keyframes, landmarks);

    const auto keyfrms = map_db_->get_all_keyframes();
    for (const auto keyfrm : keyfrms) {
        bow_db_->add_keyframe(keyfrm);
    }

}

} // namespace io
} // namespace openvslam
