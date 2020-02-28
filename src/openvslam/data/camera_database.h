#ifndef OPENVSLAM_DATA_CAMERA_DATABASE_H
#define OPENVSLAM_DATA_CAMERA_DATABASE_H

#include <mutex>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

namespace openvslam {

namespace camera {
class base;
} // namespace camera

namespace data {

class camera_database {
public:
    explicit camera_database(camera::base* curr_camera);

    ~camera_database();

    camera::base* get_camera(const std::string& camera_name) const;

    void from_json(const nlohmann::json& json_cameras);

    nlohmann::json to_json() const;

    struct perspective_data {
        double fx;
        double fy;
        double cx;
        double cy;
        double k1;
        double k2;
        double p1;
        double p2;
        double k3;
        double focal_x_baseline;
    };

    struct fisheye_data {
        double fx;
        double fy;
        double cx;
        double cy;
        double k1;
        double k2;
        double k3;
        double k4;
        double focal_x_baseline;
    };

    struct equirectangular_data {

    };

    struct camera_data {
        std::string camera_name;
        camera::model_type_t model_type;
        camera::setup_type_t setup_type;
        camera::color_order_t color_order;
        unsigned int cols;
        unsigned int rows;
        double fps;
        union {
            perspective_data perspective;
            fisheye_data fisheye;
            equirectangular_data equirectangular;
        };
    };

    void from_buffer(std::vector<camera_data> cameras);

    std::vector<camera_data> to_buffer();

private:
    //-----------------------------------------
    //! mutex to access the database
    mutable std::mutex mtx_database_;
    //! pointer to the camera which used in the current tracking
    //! (NOTE: the object is owned by config class,
    //!  thus this class does NOT delete the object of curr_camera_)
    camera::base* curr_camera_ = nullptr;
    //! database (key: camera name, value: pointer of camera::base)
    //! (NOTE: tracking camera must NOT be contained in the database)
    std::unordered_map<std::string, camera::base*> database_;
};

} // namespace data
} // namespace openvslam

#endif // OPENVSLAM_DATA_CAMERA_DATABASE_H
