#include "GpuEvaluator.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFunctions_4_3_Core>
#include <QSurfaceFormat>

#include "geometrize/bitmap/bitmap.h"
#include "geometrize/shape/circle.h"
#include "geometrize/shape/ellipse.h"
#include "geometrize/shape/line.h"
#include "geometrize/shape/rectangle.h"
#include "geometrize/shape/rotatedellipse.h"
#include "geometrize/shape/rotatedrectangle.h"
#include "geometrize/shape/shape.h"
#include "geometrize/shape/triangle.h"

namespace
{

constexpr std::int32_t GPU_RECTANGLE = 1;
constexpr std::int32_t GPU_ROTATED_RECTANGLE = 2;
constexpr std::int32_t GPU_TRIANGLE = 4;
constexpr std::int32_t GPU_ELLIPSE = 8;
constexpr std::int32_t GPU_ROTATED_ELLIPSE = 16;
constexpr std::int32_t GPU_CIRCLE = 32;
constexpr std::int32_t GPU_LINE = 64;

template<typename T> T clampValue(const T value, const T lo, const T hi)
{
    return std::max(lo, std::min(value, hi));
}

struct alignas(16) PackedCandidate final
{
    // std430-aligned shape payload consumed by shaders/evaluate_shapes.comp
    std::array<std::int32_t, 4> header{};
    std::array<std::int32_t, 4> bounds{};
    std::array<float, 4> params0{};
    std::array<float, 4> params1{};
};
static_assert(sizeof(PackedCandidate) == 64, "Unexpected PackedCandidate layout");

struct alignas(16) ReductionResult final
{
    // Packed output from shaders/error_reduce.comp (score/index/color/valid).
    float score = 1.0F;
    std::uint32_t index = 0U;
    std::uint32_t color = 0U;
    std::uint32_t valid = 0U;
};
static_assert(sizeof(ReductionResult) == 16, "Unexpected ReductionResult layout");

QString readTextFile(const QString& relativePath)
{
    const QStringList roots{
        QDir::currentPath(),
        QCoreApplication::applicationDirPath(),
        QDir(QCoreApplication::applicationDirPath()).filePath("..")
    };
    for(const QString& root : roots) {
        const QString path = QDir(root).filePath(relativePath);
        QFile file(path);
        if(file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return QString::fromUtf8(file.readAll());
        }
    }
    return {};
}

std::array<std::pair<float, float>, 4> getRotatedRectangleCorners(const geometrize::RotatedRectangle& r)
{
    const float x1 = std::fmin(r.m_x1, r.m_x2);
    const float x2 = std::fmax(r.m_x1, r.m_x2);
    const float y1 = std::fmin(r.m_y1, r.m_y2);
    const float y2 = std::fmax(r.m_y1, r.m_y2);
    const float cx = (x1 + x2) * 0.5F;
    const float cy = (y1 + y2) * 0.5F;
    const float ox1 = x1 - cx;
    const float ox2 = x2 - cx;
    const float oy1 = y1 - cy;
    const float oy2 = y2 - cy;
    const float rad = r.m_angle * 3.1415926535F / 180.0F;
    const float co = std::cos(rad);
    const float si = std::sin(rad);

    return {{
        {ox1 * co - oy1 * si + cx, ox1 * si + oy1 * co + cy},
        {ox2 * co - oy1 * si + cx, ox2 * si + oy1 * co + cy},
        {ox2 * co - oy2 * si + cx, ox2 * si + oy2 * co + cy},
        {ox1 * co - oy2 * si + cx, ox1 * si + oy2 * co + cy}
    }};
}

}

namespace geometrize
{
namespace gpu
{

class GpuEvaluator::GpuEvaluatorImpl
{
public:
    ~GpuEvaluatorImpl()
    {
        destroy();
    }

    bool isAvailable()
    {
        return ensureInitialized();
    }

    GpuEvaluator::BatchResult evaluate(const geometrize::Bitmap& target,
                                       const geometrize::Bitmap& current,
                                       const std::vector<std::shared_ptr<geometrize::Shape>>& candidates,
                                       const std::uint8_t alpha,
                                       const double lastScore)
    {
        GpuEvaluator::BatchResult result{};
        if(candidates.empty()) {
            result.error = "No candidates were provided for GPU evaluation.";
            m_lastError = result.error;
            return result;
        }

        if(target.getWidth() != current.getWidth() || target.getHeight() != current.getHeight()) {
            result.error = "Target and current bitmap dimensions must match.";
            m_lastError = result.error;
            return result;
        }

        if(!ensureInitialized()) {
            result.error = m_lastError;
            return result;
        }

        if(!m_context->makeCurrent(m_surface.get())) {
            m_lastError = "Failed to make OpenGL context current.";
            result.error = m_lastError;
            return result;
        }

        const auto width = static_cast<std::int32_t>(target.getWidth());
        const auto height = static_cast<std::int32_t>(target.getHeight());
        if(!ensureResources(width, height, static_cast<std::int32_t>(candidates.size()))) {
            m_context->doneCurrent();
            result.error = m_lastError;
            return result;
        }

        std::vector<PackedCandidate> packedCandidates;
        packedCandidates.reserve(candidates.size());
        for(const auto& candidate : candidates) {
            PackedCandidate packed{};
            if(!packCandidate(*candidate, width, height, packed)) {
                m_lastError = "Encountered unsupported shape type for GPU evaluation.";
                m_context->doneCurrent();
                result.error = m_lastError;
                return result;
            }
            packedCandidates.push_back(packed);
        }

        if(!uploadTexture(m_targetTexture, target) || !uploadTexture(m_currentTexture, current)) {
            m_context->doneCurrent();
            result.error = m_lastError;
            return result;
        }

        m_gl->glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_candidateBuffer);
        m_gl->glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                              0,
                              static_cast<GLsizeiptr>(packedCandidates.size() * sizeof(PackedCandidate)),
                              packedCandidates.data());

        const auto candidateCount = static_cast<std::uint32_t>(packedCandidates.size());

        m_gl->glUseProgram(m_evaluateProgram);
        m_gl->glUniform1ui(m_evaluateCandidateCountUniform, candidateCount);
        m_gl->glUniform1ui(m_evaluateAlphaUniform, alpha);
        m_gl->glUniform1f(m_evaluateBaseScoreUniform, static_cast<float>(lastScore));
        m_gl->glUniform1f(m_evaluateRgbaCountUniform, static_cast<float>(target.getWidth() * target.getHeight() * 4U));
        m_gl->glActiveTexture(GL_TEXTURE0);
        m_gl->glBindTexture(GL_TEXTURE_2D, m_targetTexture);
        m_gl->glActiveTexture(GL_TEXTURE1);
        m_gl->glBindTexture(GL_TEXTURE_2D, m_currentTexture);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_candidateBuffer);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_scoreBuffer);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_colorBuffer);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_validBuffer);
        m_gl->glDispatchCompute(candidateCount, 1, 1);
        m_gl->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // GPU-side reduction step: scan candidate scores and keep best index/color.
        m_gl->glUseProgram(m_reduceProgram);
        m_gl->glUniform1ui(m_reduceCandidateCountUniform, candidateCount);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, m_scoreBuffer);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, m_colorBuffer);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, m_validBuffer);
        m_gl->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, m_reduceBuffer);
        m_gl->glDispatchCompute(1, 1, 1);
        m_gl->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        m_gl->glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_reduceBuffer);
        void* mapped = m_gl->glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(ReductionResult), GL_MAP_READ_BIT);
        if(mapped == nullptr) {
            m_lastError = "Failed to read GPU reduction output.";
            m_context->doneCurrent();
            result.error = m_lastError;
            return result;
        }
        const auto reduction = *reinterpret_cast<const ReductionResult*>(mapped);
        m_gl->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        m_context->doneCurrent();

        if(reduction.valid == 0U || reduction.index >= packedCandidates.size()) {
            m_lastError = "GPU reduction did not return a valid best candidate.";
            result.error = m_lastError;
            return result;
        }

        result.success = true;
        result.bestCandidateIndex = reduction.index;
        result.bestScore = reduction.score;
        result.packedColor = reduction.color;
        return result;
    }

    QString lastError() const
    {
        return m_lastError;
    }

private:
    bool ensureInitialized()
    {
        if(m_initialized) {
            return true;
        }

        QSurfaceFormat format;
        format.setRenderableType(QSurfaceFormat::OpenGL);
        format.setProfile(QSurfaceFormat::CoreProfile);
        format.setVersion(4, 3);

        m_surface = std::make_unique<QOffscreenSurface>();
        m_surface->setFormat(format);
        m_surface->create();
        if(!m_surface->isValid()) {
            m_lastError = "Failed to create offscreen OpenGL surface.";
            return false;
        }

        m_context = std::make_unique<QOpenGLContext>();
        m_context->setFormat(format);
        if(!m_context->create()) {
            m_lastError = "Failed to create OpenGL 4.3 context.";
            return false;
        }

        if(!m_context->makeCurrent(m_surface.get())) {
            m_lastError = "Failed to activate OpenGL context.";
            return false;
        }

        m_gl = m_context->versionFunctions<QOpenGLFunctions_4_3_Core>();
        if(m_gl == nullptr || !m_gl->initializeOpenGLFunctions()) {
            m_context->doneCurrent();
            m_lastError = "OpenGL 4.3 compute shader support is unavailable.";
            return false;
        }

        if(!compilePrograms()) {
            m_context->doneCurrent();
            return false;
        }

        m_initialized = true;
        m_context->doneCurrent();
        return true;
    }

    void destroy()
    {
        if(!(m_context && m_surface && m_gl)) {
            return;
        }
        if(!m_context->makeCurrent(m_surface.get())) {
            return;
        }

        if(m_evaluateProgram != 0U) {
            m_gl->glDeleteProgram(m_evaluateProgram);
        }
        if(m_reduceProgram != 0U) {
            m_gl->glDeleteProgram(m_reduceProgram);
        }
        if(m_targetTexture != 0U) {
            m_gl->glDeleteTextures(1, &m_targetTexture);
        }
        if(m_currentTexture != 0U) {
            m_gl->glDeleteTextures(1, &m_currentTexture);
        }
        if(m_candidateBuffer != 0U) {
            m_gl->glDeleteBuffers(1, &m_candidateBuffer);
        }
        if(m_scoreBuffer != 0U) {
            m_gl->glDeleteBuffers(1, &m_scoreBuffer);
        }
        if(m_colorBuffer != 0U) {
            m_gl->glDeleteBuffers(1, &m_colorBuffer);
        }
        if(m_validBuffer != 0U) {
            m_gl->glDeleteBuffers(1, &m_validBuffer);
        }
        if(m_reduceBuffer != 0U) {
            m_gl->glDeleteBuffers(1, &m_reduceBuffer);
        }
        m_context->doneCurrent();
    }

    bool compilePrograms()
    {
        const QString evaluateSource = readTextFile("shaders/evaluate_shapes.comp");
        const QString reduceSource = readTextFile("shaders/error_reduce.comp");
        if(evaluateSource.isEmpty() || reduceSource.isEmpty()) {
            m_lastError = "Failed to load compute shaders from the shaders/ directory.";
            return false;
        }

        m_evaluateProgram = buildComputeProgram(evaluateSource.toUtf8(), "evaluate_shapes.comp");
        if(m_evaluateProgram == 0U) {
            return false;
        }
        m_reduceProgram = buildComputeProgram(reduceSource.toUtf8(), "error_reduce.comp");
        if(m_reduceProgram == 0U) {
            return false;
        }

        m_gl->glUseProgram(m_evaluateProgram);
        m_gl->glUniform1i(m_gl->glGetUniformLocation(m_evaluateProgram, "uTargetTex"), 0);
        m_gl->glUniform1i(m_gl->glGetUniformLocation(m_evaluateProgram, "uCurrentTex"), 1);
        m_evaluateCandidateCountUniform = m_gl->glGetUniformLocation(m_evaluateProgram, "uCandidateCount");
        m_evaluateAlphaUniform = m_gl->glGetUniformLocation(m_evaluateProgram, "uAlpha");
        m_evaluateBaseScoreUniform = m_gl->glGetUniformLocation(m_evaluateProgram, "uBaseScore");
        m_evaluateRgbaCountUniform = m_gl->glGetUniformLocation(m_evaluateProgram, "uRgbaCount");

        m_gl->glUseProgram(m_reduceProgram);
        m_reduceCandidateCountUniform = m_gl->glGetUniformLocation(m_reduceProgram, "uCandidateCount");
        return true;
    }

    GLuint buildComputeProgram(const QByteArray& source, const char* shaderLabel)
    {
        const GLuint shader = m_gl->glCreateShader(GL_COMPUTE_SHADER);
        const auto* sourcePtr = source.constData();
        const GLint sourceLength = source.size();
        m_gl->glShaderSource(shader, 1, &sourcePtr, &sourceLength);
        m_gl->glCompileShader(shader);

        GLint compileSuccess = 0;
        m_gl->glGetShaderiv(shader, GL_COMPILE_STATUS, &compileSuccess);
        if(compileSuccess == 0) {
            m_lastError = QString("Failed to compile %1:\n%2").arg(shaderLabel, shaderLog(shader));
            m_gl->glDeleteShader(shader);
            return 0U;
        }

        const GLuint program = m_gl->glCreateProgram();
        m_gl->glAttachShader(program, shader);
        m_gl->glLinkProgram(program);
        m_gl->glDeleteShader(shader);

        GLint linkSuccess = 0;
        m_gl->glGetProgramiv(program, GL_LINK_STATUS, &linkSuccess);
        if(linkSuccess == 0) {
            m_lastError = QString("Failed to link %1:\n%2").arg(shaderLabel, programLog(program));
            m_gl->glDeleteProgram(program);
            return 0U;
        }

        return program;
    }

    QString shaderLog(const GLuint shader) const
    {
        GLint length = 0;
        m_gl->glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        if(length <= 1) {
            return {};
        }
        std::string log(static_cast<std::size_t>(length), '\0');
        m_gl->glGetShaderInfoLog(shader, length, nullptr, log.data());
        return QString::fromStdString(log);
    }

    QString programLog(const GLuint program) const
    {
        GLint length = 0;
        m_gl->glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        if(length <= 1) {
            return {};
        }
        std::string log(static_cast<std::size_t>(length), '\0');
        m_gl->glGetProgramInfoLog(program, length, nullptr, log.data());
        return QString::fromStdString(log);
    }

    bool ensureResources(const std::int32_t width, const std::int32_t height, const std::int32_t candidateCapacity)
    {
        if(width <= 0 || height <= 0 || candidateCapacity <= 0) {
            m_lastError = "Invalid dimensions for GPU resource creation.";
            return false;
        }

        if(m_targetTexture == 0U) {
            m_gl->glGenTextures(1, &m_targetTexture);
            m_gl->glBindTexture(GL_TEXTURE_2D, m_targetTexture);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        if(m_currentTexture == 0U) {
            m_gl->glGenTextures(1, &m_currentTexture);
            m_gl->glBindTexture(GL_TEXTURE_2D, m_currentTexture);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        if(m_textureWidth != width || m_textureHeight != height) {
            m_textureWidth = width;
            m_textureHeight = height;
            m_gl->glBindTexture(GL_TEXTURE_2D, m_targetTexture);
            m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            m_gl->glBindTexture(GL_TEXTURE_2D, m_currentTexture);
            m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        }

        if(candidateCapacity <= m_candidateCapacity) {
            return true;
        }

        m_candidateCapacity = candidateCapacity;
        createOrResizeBuffer(m_candidateBuffer, sizeof(PackedCandidate) * static_cast<GLsizeiptr>(m_candidateCapacity));
        createOrResizeBuffer(m_scoreBuffer, sizeof(float) * static_cast<GLsizeiptr>(m_candidateCapacity));
        createOrResizeBuffer(m_colorBuffer, sizeof(std::uint32_t) * static_cast<GLsizeiptr>(m_candidateCapacity));
        createOrResizeBuffer(m_validBuffer, sizeof(std::uint32_t) * static_cast<GLsizeiptr>(m_candidateCapacity));
        createOrResizeBuffer(m_reduceBuffer, sizeof(ReductionResult));
        return true;
    }

    void createOrResizeBuffer(GLuint& buffer, const GLsizeiptr size)
    {
        if(buffer == 0U) {
            m_gl->glGenBuffers(1, &buffer);
        }
        m_gl->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        m_gl->glBufferData(GL_SHADER_STORAGE_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
    }

    bool uploadTexture(const GLuint texture, const geometrize::Bitmap& bitmap)
    {
        const auto& data = bitmap.getDataRef();
        if(data.empty()) {
            m_lastError = "Bitmap data is empty.";
            return false;
        }
        m_gl->glBindTexture(GL_TEXTURE_2D, texture);
        m_gl->glTexSubImage2D(GL_TEXTURE_2D,
                              0,
                              0,
                              0,
                              static_cast<GLsizei>(bitmap.getWidth()),
                              static_cast<GLsizei>(bitmap.getHeight()),
                              GL_RGBA,
                              GL_UNSIGNED_BYTE,
                              data.data());
        return true;
    }

    bool packCandidate(const geometrize::Shape& shape, const std::int32_t width, const std::int32_t height, PackedCandidate& packed) const
    {
        packed = PackedCandidate{};
        auto finalizeBounds = [&](float minXF, float minYF, float maxXF, float maxYF) -> bool {
            auto minX = static_cast<std::int32_t>(std::floor(minXF));
            auto minY = static_cast<std::int32_t>(std::floor(minYF));
            auto maxX = static_cast<std::int32_t>(std::ceil(maxXF));
            auto maxY = static_cast<std::int32_t>(std::ceil(maxYF));
            minX = clampValue(minX, 0, width - 1);
            minY = clampValue(minY, 0, height - 1);
            maxX = clampValue(maxX, 0, width - 1);
            maxY = clampValue(maxY, 0, height - 1);
            if(maxX < minX || maxY < minY) {
                return false;
            }
            packed.header[1] = minX;
            packed.header[2] = minY;
            packed.bounds[0] = maxX;
            packed.bounds[1] = maxY;
            return true;
        };

        if(const auto* rectangle = dynamic_cast<const geometrize::Rectangle*>(&shape)) {
            packed.header[0] = GPU_RECTANGLE;
            packed.params0 = {rectangle->m_x1, rectangle->m_y1, rectangle->m_x2, rectangle->m_y2};
            return finalizeBounds(std::min(rectangle->m_x1, rectangle->m_x2),
                                  std::min(rectangle->m_y1, rectangle->m_y2),
                                  std::max(rectangle->m_x1, rectangle->m_x2),
                                  std::max(rectangle->m_y1, rectangle->m_y2));
        }

        if(const auto* rotatedRectangle = dynamic_cast<const geometrize::RotatedRectangle*>(&shape)) {
            packed.header[0] = GPU_ROTATED_RECTANGLE;
            packed.params0 = {rotatedRectangle->m_x1, rotatedRectangle->m_y1, rotatedRectangle->m_x2, rotatedRectangle->m_y2};
            packed.params1[0] = rotatedRectangle->m_angle;

            const auto corners = getRotatedRectangleCorners(*rotatedRectangle);
            float minX = std::numeric_limits<float>::max();
            float minY = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float maxY = std::numeric_limits<float>::lowest();
            for(const auto& corner : corners) {
                minX = std::min(minX, corner.first);
                minY = std::min(minY, corner.second);
                maxX = std::max(maxX, corner.first);
                maxY = std::max(maxY, corner.second);
            }
            return finalizeBounds(minX, minY, maxX, maxY);
        }

        if(const auto* triangle = dynamic_cast<const geometrize::Triangle*>(&shape)) {
            packed.header[0] = GPU_TRIANGLE;
            packed.params0 = {triangle->m_x1, triangle->m_y1, triangle->m_x2, triangle->m_y2};
            packed.params1 = {triangle->m_x3, triangle->m_y3, 0.0F, 0.0F};
            return finalizeBounds(std::min({triangle->m_x1, triangle->m_x2, triangle->m_x3}),
                                  std::min({triangle->m_y1, triangle->m_y2, triangle->m_y3}),
                                  std::max({triangle->m_x1, triangle->m_x2, triangle->m_x3}),
                                  std::max({triangle->m_y1, triangle->m_y2, triangle->m_y3}));
        }

        if(const auto* ellipse = dynamic_cast<const geometrize::Ellipse*>(&shape)) {
            packed.header[0] = GPU_ELLIPSE;
            packed.params0 = {ellipse->m_x, ellipse->m_y, ellipse->m_rx, ellipse->m_ry};
            return finalizeBounds(ellipse->m_x - std::abs(ellipse->m_rx),
                                  ellipse->m_y - std::abs(ellipse->m_ry),
                                  ellipse->m_x + std::abs(ellipse->m_rx),
                                  ellipse->m_y + std::abs(ellipse->m_ry));
        }

        if(const auto* rotatedEllipse = dynamic_cast<const geometrize::RotatedEllipse*>(&shape)) {
            packed.header[0] = GPU_ROTATED_ELLIPSE;
            packed.params0 = {rotatedEllipse->m_x, rotatedEllipse->m_y, rotatedEllipse->m_rx, rotatedEllipse->m_ry};
            packed.params1[0] = rotatedEllipse->m_angle;

            const float rad = rotatedEllipse->m_angle * 3.1415926535F / 180.0F;
            const float co = std::cos(rad);
            const float si = std::sin(rad);
            const float rx = std::abs(rotatedEllipse->m_rx);
            const float ry = std::abs(rotatedEllipse->m_ry);
            const float boundX = std::sqrt((rx * co) * (rx * co) + (ry * si) * (ry * si));
            const float boundY = std::sqrt((rx * si) * (rx * si) + (ry * co) * (ry * co));
            return finalizeBounds(rotatedEllipse->m_x - boundX,
                                  rotatedEllipse->m_y - boundY,
                                  rotatedEllipse->m_x + boundX,
                                  rotatedEllipse->m_y + boundY);
        }

        if(const auto* circle = dynamic_cast<const geometrize::Circle*>(&shape)) {
            packed.header[0] = GPU_CIRCLE;
            packed.params0 = {circle->m_x, circle->m_y, circle->m_r, 0.0F};
            const float r = std::abs(circle->m_r);
            return finalizeBounds(circle->m_x - r, circle->m_y - r, circle->m_x + r, circle->m_y + r);
        }

        if(const auto* line = dynamic_cast<const geometrize::Line*>(&shape)) {
            packed.header[0] = GPU_LINE;
            packed.params0 = {line->m_x1, line->m_y1, line->m_x2, line->m_y2};
            constexpr float linePadding = 2.0F;
            return finalizeBounds(std::min(line->m_x1, line->m_x2) - linePadding,
                                  std::min(line->m_y1, line->m_y2) - linePadding,
                                  std::max(line->m_x1, line->m_x2) + linePadding,
                                  std::max(line->m_y1, line->m_y2) + linePadding);
        }

        return false;
    }

    std::unique_ptr<QOffscreenSurface> m_surface;
    std::unique_ptr<QOpenGLContext> m_context;
    QOpenGLFunctions_4_3_Core* m_gl = nullptr;
    bool m_initialized = false;
    QString m_lastError;

    GLuint m_evaluateProgram = 0U;
    GLuint m_reduceProgram = 0U;
    GLint m_evaluateCandidateCountUniform = -1;
    GLint m_evaluateAlphaUniform = -1;
    GLint m_evaluateBaseScoreUniform = -1;
    GLint m_evaluateRgbaCountUniform = -1;
    GLint m_reduceCandidateCountUniform = -1;

    GLuint m_targetTexture = 0U;
    GLuint m_currentTexture = 0U;
    std::int32_t m_textureWidth = 0;
    std::int32_t m_textureHeight = 0;

    GLuint m_candidateBuffer = 0U;
    GLuint m_scoreBuffer = 0U;
    GLuint m_colorBuffer = 0U;
    GLuint m_validBuffer = 0U;
    GLuint m_reduceBuffer = 0U;
    std::int32_t m_candidateCapacity = 0;
};

GpuEvaluator::GpuEvaluator() : d{std::make_unique<GpuEvaluatorImpl>()}
{
}

GpuEvaluator::~GpuEvaluator() = default;

bool GpuEvaluator::isAvailable()
{
    return d->isAvailable();
}

GpuEvaluator::BatchResult GpuEvaluator::evaluate(const geometrize::Bitmap& target,
                                                 const geometrize::Bitmap& current,
                                                 const std::vector<std::shared_ptr<geometrize::Shape>>& candidates,
                                                 const std::uint8_t alpha,
                                                 const double lastScore)
{
    return d->evaluate(target, current, candidates, alpha, lastScore);
}

QString GpuEvaluator::lastError() const
{
    return d->lastError();
}

}
}
