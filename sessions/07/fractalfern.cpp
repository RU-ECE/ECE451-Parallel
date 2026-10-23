#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <webp/encode.h>

using namespace std;

constexpr uint32_t WIDTH = 1200;
constexpr uint32_t HEIGHT = 1600;
constexpr uint32_t POINTS = 20'000'000;

constexpr double X_MIN = -2.7;
constexpr double X_MAX = 2.7;
constexpr double Y_MIN = 0.0;
constexpr double Y_MAX = 10.0;

struct Point
{
    double x;
    double y;
};

Point transform(Point p, uint32_t choice)
{
    switch (choice)
    {
        case 0:
            return {0.0, 0.16 * p.y};

        case 1:
            return {
                0.85 * p.x + 0.04 * p.y,
               -0.04 * p.x + 0.85 * p.y + 1.6
            };

        case 2:
            return {
                0.20 * p.x - 0.26 * p.y,
                0.23 * p.x + 0.22 * p.y + 1.6
            };

        default:
            return {
               -0.15 * p.x + 0.28 * p.y,
                0.26 * p.x + 0.24 * p.y + 0.44
            };
    }
}

void compute_fern(vector<uint32_t>& image)
{
    mt19937_64 rng(12345);
    uniform_int_distribution<uint32_t> distribution(0, 99);

    Point p{0.0, 0.0};

    for (uint32_t i = 0; i < POINTS; ++i)
    {
        const uint32_t r = distribution(rng);

        // try cache?  choice = lookup[r];
        const uint32_t choice =
            r < 1  ? 0 :
            r < 86 ? 1 :
            r < 93 ? 2 : 3;

        p = transform(p, choice);

        const int px = static_cast<int>(
            (p.x - X_MIN) * (WIDTH - 1) / (X_MAX - X_MIN));

        const int py = static_cast<int>(
            (Y_MAX - p.y) * (HEIGHT - 1) / (Y_MAX - Y_MIN));

        if (px >= 0 && px < static_cast<int>(WIDTH) &&
            py >= 0 && py < static_cast<int>(HEIGHT))
        {
            ++image[static_cast<size_t>(py) * WIDTH + px];
        }
    }
}

bool write_webp(const vector<uint32_t>& image)
{
    vector<uint8_t> pixels(
        static_cast<size_t>(WIDTH) * HEIGHT * 4);

    for (size_t i = 0; i < image.size(); ++i)
    {
        const double intensity = image[i] == 0
            ? 0.0
            : min(1.0,
                log1p(image[i]) / log1p(1000.0));

        const uint8_t value =
            static_cast<uint8_t>(255.0 * intensity);

        pixels[4 * i + 0] = value;
        pixels[4 * i + 1] = value;
        pixels[4 * i + 2] = value;
        pixels[4 * i + 3] = 255;
    }

    uint8_t* encoded = nullptr;
    const size_t size = WebPEncodeLosslessRGBA(
        pixels.data(), WIDTH, HEIGHT, WIDTH * 4, &encoded);

    if (size == 0)
    {
        fprintf(stderr, "WebP encoding failed\n");
        return false;
    }

    FILE* file = fopen("fern.webp", "wb");
    if (file == nullptr)
    {
        perror("fern.webp");
        WebPFree(encoded);
        return false;
    }

    fwrite(encoded, 1, size, file);
    fclose(file);
    WebPFree(encoded);
    return true;
}

int main()
{
    vector<uint32_t> image(
        static_cast<size_t>(WIDTH) * HEIGHT, 0);

    compute_fern(image);

    return write_webp(image) ? 0 : 1;
}
