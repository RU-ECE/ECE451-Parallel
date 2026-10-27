#pragma once
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "hittable.h"
#include "material.h"

class camera {
  public:
    double aspect_ratio;          // Ratio of image width over height
    int    image_width;           // Rendered image width in pixel count
    int    image_height;         // Rendered image height
    int    samples_per_pixel;     // Count of random samples for each pixel
    int    max_depth;             // Maximum number of ray bounces into scene

    double vfov;                  // Vertical view angle (field of view)
    point3 lookfrom;              // Point camera is looking from
    point3 lookat;                // Point camera is looking at
    vec3   vup;                  // Camera-relative "up" direction

    double defocus_angle;        // Variation angle of rays through each pixel
    double focus_dist;           // Distance from camera lookfrom point to plane of perfect focus

    camera() {
        aspect_ratio = 1.0;
        image_width = 100;
        samples_per_pixel = 10;
        max_depth = 10;
        vfov = 90;
        lookfrom = point3(0,0,0);
        lookat = point3(0,0,-1);
        vup = vec3(0,1,0);
        defocus_angle = 0;
        focus_dist = 10;
    }


    inline uint32_t conv_color(float c) {
        if (c < 0)
          return 0;
        if (c > 0.999) c = 0.999;
        return uint32_t(255.999 * sqrt(c));
    }

    inline uint32_t conv_color(const color& pixel_color) {
        auto r = linear_to_gamma(pixel_color.x());
        auto g = linear_to_gamma(pixel_color.y());
        auto b = linear_to_gamma(pixel_color.z());

        // Translate the [0,1] component values to the byte range [0,255].
        static const interval intensity(0.000, 0.999);
        int rbyte = int(256 * intensity.clamp(r));
        int gbyte = int(256 * intensity.clamp(g));
        int bbyte = int(256 * intensity.clamp(b));

        return 0xFF000000 | (bbyte << 16) | (gbyte << 8) | rbyte;
    }


    void render(const hittable& world, uint32_t* pixels, const std::string& filename, int frame) {
        initialize();
        std::cout << "Scanlines remaining:\n";
        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                color pixel_color(0,0,0);
                for (int sample = 0; sample < samples_per_pixel; sample++) {
                    ray r = get_ray(i, j);
                    pixel_color += ray_color(r, max_depth, world);
                }
#if 0
            color scaled_p = pixel_samples_scale * pixel_color;
            uint32_t r = conv_color(scaled_p.x());
            uint32_t g = conv_color(scaled_p.y());
            uint32_t b = conv_color(scaled_p.z());
            uint32_t p = (((((0xFF00 | b) << 8) | g) << 8) | r); // this is the same.
#endif
                uint32_t p2 = conv_color(pixel_samples_scale * pixel_color);
                pixels[j*image_width + i] = p2;
            }
        }
    }

  private:
    double pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;               // Camera center
    point3 pixel00_loc;          // Location of pixel 0, 0
    vec3   pixel_delta_u;        // Offset to pixel to the right
    vec3   pixel_delta_v;        // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    void initialize() {

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta/2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (double(image_width)/image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    ray get_ray(int i, int j) const {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.

        auto offset = sample_square();
        auto pixel_sample = pixel00_loc
                          + ((i + offset.x()) * pixel_delta_u)
                          + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample();
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

    vec3 sample_square() const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_double() - 0.5, random_double() - 0.5, 0);
    }

    vec3 sample_disk(double radius) const {
        // Returns a random point in the unit (radius 0.5) disk centered at the origin.
        return radius * random_in_unit_disk();
    }

    point3 defocus_disk_sample() const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk();
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }

    color ray_color(const ray& r, int depth, const hittable& world) const {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if (depth <= 0)
            return color(0,0,0);

        hit_record rec;

        if (world.hit(r, interval(0.001, infinity), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered))
                return attenuation * ray_color(scattered, depth-1, world);
            return color(0,0,0);
        }

        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5*(unit_direction.y() + 1.0);
        return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
    }
};

