#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <QString>

namespace geometrize
{
class Bitmap;
class Shape;
}

namespace geometrize
{
namespace gpu
{

/**
 * @brief GPU-based batch shape evaluator.
 * Evaluates many candidate shapes in parallel with OpenGL compute shaders and
 * returns the best scoring candidate index.
 */
class GpuEvaluator
{
public:
    struct BatchResult {
        bool success = false;
        std::size_t bestCandidateIndex = 0U;
        double bestScore = 1.0;
        std::uint32_t packedColor = 0U;
        QString error{};
    };

    GpuEvaluator();
    ~GpuEvaluator();
    GpuEvaluator(const GpuEvaluator&) = delete;
    GpuEvaluator& operator=(const GpuEvaluator&) = delete;

    bool isAvailable();
    BatchResult evaluate(const geometrize::Bitmap& target,
                         const geometrize::Bitmap& current,
                         const std::vector<std::shared_ptr<geometrize::Shape>>& candidates,
                         std::uint8_t alpha,
                         double lastScore);
    QString lastError() const;

private:
    class GpuEvaluatorImpl;
    std::unique_ptr<GpuEvaluatorImpl> d;
};

}
}
