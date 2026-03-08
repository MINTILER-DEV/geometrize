#include "imagetaskworker.h"

#include <algorithm>
#include <thread>
#include <vector>

#include <QDebug>
#include <QtGlobal>

#include "src/gpu/GpuEvaluator.h"
#include "geometrize/commonutil.h"
#include "geometrize/core.h"
#include "geometrize/bitmap/bitmap.h"
#include "geometrize/bitmap/rgba.h"
#include "geometrize/model.h"
#include "geometrize/runner/imagerunner.h"
#include "geometrize/shape/shapefactory.h"
#include "geometrize/shaperesult.h"

namespace
{

constexpr std::uint32_t kMinGpuBatch = 512U;
constexpr std::uint32_t kMaxGpuBatch = 4096U;

std::uint32_t getThreadHint(const geometrize::ImageRunnerOptions& options)
{
    if(options.maxThreads > 0U) {
        return options.maxThreads;
    }
    const auto hwThreads = std::thread::hardware_concurrency();
    return hwThreads > 0U ? hwThreads : 1U;
}

std::uint32_t getGpuBatchSize(const geometrize::ImageRunnerOptions& options)
{
    const std::uint32_t rawBatch = std::max(1U, options.shapeCount) * getThreadHint(options);
    return std::clamp(rawBatch, kMinGpuBatch, kMaxGpuBatch);
}

bool isGpuDisabledByEnvironment()
{
    bool ok = false;
    const int disabled = qEnvironmentVariableIntValue("GEOMETRIZE_DISABLE_GPU", &ok);
    return ok && disabled != 0;
}

}

namespace geometrize
{

namespace task
{

ImageTaskWorker::ImageTaskWorker(Bitmap& bitmap) :
    QObject(),
    m_runner{bitmap},
    m_working{false},
    m_gpuEvaluator{std::make_unique<geometrize::gpu::GpuEvaluator>()},
    m_gpuSeedOffset{0U}
{
}

ImageTaskWorker::ImageTaskWorker(Bitmap& bitmap, const Bitmap& initial) :
    QObject(),
    m_runner{bitmap, initial},
    m_working{false},
    m_gpuEvaluator{std::make_unique<geometrize::gpu::GpuEvaluator>()},
    m_gpuSeedOffset{0U}
{
}

ImageTaskWorker::~ImageTaskWorker()
{
}

void ImageTaskWorker::step(const geometrize::ImageRunnerOptions options,
                           const std::function<std::shared_ptr<geometrize::Shape>()> shapeCreator,
                           const geometrize::core::EnergyFunction energyFunction,
                           const geometrize::ShapeAcceptancePreconditionFunction addShapePreconditionFunction)
{
    emit signal_willStep();
    m_working = true;

    std::vector<geometrize::ShapeResult> results;
    const auto runCpuFallback = [&]() {
        results = m_runner.step(options, shapeCreator, energyFunction, addShapePreconditionFunction);
    };

    if(isGpuDisabledByEnvironment() || energyFunction != nullptr || !m_gpuEvaluator || !m_gpuEvaluator->isAvailable()) {
        runCpuFallback();
        m_working = false;
        emit signal_didStep(results);
        return;
    }

    std::function<std::shared_ptr<geometrize::Shape>()> candidateFactory = shapeCreator;
    if(!candidateFactory) {
        const auto [xMin, yMin, xMax, yMax] = geometrize::commonutil::mapShapeBoundsToImage(options.shapeBounds, m_runner.getTarget());
        candidateFactory = geometrize::createDefaultShapeCreator(options.shapeTypes, xMin, yMin, xMax, yMax);
    }

    if(!candidateFactory) {
        runCpuFallback();
        m_working = false;
        emit signal_didStep(results);
        return;
    }

    const std::uint32_t batchSize = getGpuBatchSize(options);
    const std::uint32_t seed = options.seed + m_gpuSeedOffset;
    m_gpuSeedOffset += batchSize;
    geometrize::commonutil::seedRandomGenerator(seed);

    std::vector<std::shared_ptr<geometrize::Shape>> candidates;
    candidates.reserve(batchSize);
    for(std::uint32_t i = 0; i < batchSize; i++) {
        auto shape = candidateFactory();
        if(!shape) {
            candidates.clear();
            break;
        }
        if(shape->mutate && options.maxShapeMutations > 0U) {
            const auto mutationCount = static_cast<std::uint32_t>(geometrize::commonutil::randomRange(0, static_cast<std::int32_t>(options.maxShapeMutations)));
            for(std::uint32_t j = 0; j < mutationCount; j++) {
                shape->mutate(*shape);
            }
        }
        candidates.emplace_back(std::move(shape));
    }

    if(candidates.empty()) {
        runCpuFallback();
        m_working = false;
        emit signal_didStep(results);
        return;
    }

    const double lastScore = geometrize::core::differenceFull(m_runner.getTarget(), m_runner.getCurrent());
    const auto gpuResult = m_gpuEvaluator->evaluate(m_runner.getTarget(), m_runner.getCurrent(), candidates, options.alpha, lastScore);
    if(!gpuResult.success || gpuResult.bestCandidateIndex >= candidates.size()) {
        qWarning() << "GPU evaluation fallback:" << m_gpuEvaluator->lastError();
        runCpuFallback();
        m_working = false;
        emit signal_didStep(results);
        return;
    }

    const auto bestShape = candidates.at(gpuResult.bestCandidateIndex);
    geometrize::ImageRunnerOptions applyOptions = options;
    applyOptions.shapeCount = 0U;
    applyOptions.maxShapeMutations = 0U;
    applyOptions.maxThreads = 1U;

    const auto fixedShapeCreator = [bestShape]() { return bestShape->clone(); };
    results = m_runner.step(applyOptions, fixedShapeCreator, nullptr, addShapePreconditionFunction);

    m_working = false;
    emit signal_didStep(results);
}

void ImageTaskWorker::drawShape(const std::shared_ptr<geometrize::Shape> shape, const geometrize::rgba color)
{
    emit signal_willStep();
    m_working = true;
    const geometrize::ShapeResult result{m_runner.getModel().drawShape(shape, color)};
    m_working = false;
    emit signal_didStep({ result });
}

geometrize::Bitmap& ImageTaskWorker::getCurrent()
{
    return m_runner.getCurrent();
}

geometrize::Bitmap& ImageTaskWorker::getTarget()
{
    return m_runner.getTarget();
}

const geometrize::Bitmap& ImageTaskWorker::getCurrent() const
{
    return m_runner.getCurrent();
}

const geometrize::Bitmap& ImageTaskWorker::getTarget() const
{
    return m_runner.getTarget();
}

geometrize::ImageRunner& ImageTaskWorker::getRunner()
{
    return m_runner;
}

bool ImageTaskWorker::isStepping() const
{
    return m_working;
}

}

}
