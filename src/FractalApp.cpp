#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/FileWatcher.h"
#include "cinder/Utilities.h"
#include "cinder/CameraUi.h"
#include "cinder/ImageIo.h"
#include "cinder/gl/gl.h"
#include "cinder/Log.h"

class FractalApp : public ci::app::App
{
public:
    FractalApp();

    void setup() override;
    void update() override;
    void draw() override;
    void resize() override;

    void mouseWheel(ci::app::MouseEvent) override;
    void mouseDrag(ci::app::MouseEvent) override;
    void mouseDown(ci::app::MouseEvent) override;
    void mouseUp(ci::app::MouseEvent) override;
    void keyDown(ci::app::KeyEvent) override;

    ci::CameraPersp mCameraPersp;
    ci::CameraUi mCameraUi;

    std::shared_ptr<ci::gl::GlslProg> mGlslProg;
};

FractalApp::FractalApp() : mCameraUi(&mCameraPersp) {}

void FractalApp::setup()
{
    std::vector<ci::fs::path> glslPaths = {"shader.vert", "shader.frag"};
    ci::FileWatcher::instance().watch(glslPaths, [&, this](const ci::WatchEvent &watchEvent) {
        try
        {
            mGlslProg = ci::gl::GlslProg::create(loadAsset(glslPaths[0]), loadAsset(glslPaths[1]));
            CI_LOG_I("Loaded shaders");
        }
        catch (const std::exception &exception)
        {
            CI_LOG_EXCEPTION("Failed to load shaders", exception);
        }
    });

    ci::gl::enableDepthRead();
    ci::gl::enableDepthWrite();

    mCameraPersp.lookAt(ci::vec3(0.0, 0.0, 3.0), ci::vec3(0.0, 0.0, 0.0));
}

void FractalApp::update()
{
    if (mGlslProg)
    {
        mGlslProg->uniform("uCameraPosition", mCameraPersp.getEyePoint());
        mGlslProg->uniform("uViewMatrix", mCameraPersp.getViewMatrix());
        mGlslProg->uniform("uProjectionMatrix", mCameraPersp.getProjectionMatrix());
        mGlslProg->uniform("uElapsedSeconds", static_cast<float>(getElapsedSeconds()));
    }
}

void FractalApp::draw()
{
    ci::gl::clear(ci::Color(1, 0, 0));
    ci::gl::setMatricesWindow(getWindowSize());
    ci::gl::ScopedGlslProg scopedGlslProg(mGlslProg);
    {
        ci::gl::drawSolidRect(ci::Rectf(ci::vec2(0, 0), ci::vec2(getWindowSize())));
    }
}

void FractalApp::resize()
{
    mCameraPersp.setAspectRatio(getWindow()->getAspectRatio());
    if (mGlslProg)
    {
        mGlslProg->uniform("uResolution", ci::vec2(ci::app::toPixels(getWindowSize())));
    }
}

void FractalApp::mouseWheel(ci::app::MouseEvent mouseEvent)
{
    mCameraUi.mouseWheel(mouseEvent);
}

void FractalApp::mouseDrag(ci::app::MouseEvent mouseEvent)
{
    mCameraUi.mouseDrag(mouseEvent);
}

void FractalApp::mouseDown(ci::app::MouseEvent mouseEvent)
{
    mCameraUi.mouseDown(mouseEvent);
}

void FractalApp::keyDown(ci::app::KeyEvent keyEvent)
{
    if (keyEvent.getChar() == 'c')
    {
        ci::writeImage(ci::getHomeDirectory() / "cinder" / "images" / (ci::toString(getElapsedFrames()) + ".png"), ci::app::copyWindowSurface());
    }
}

void FractalApp::mouseUp(ci::app::MouseEvent mouseEvent)
{
    mCameraUi.mouseUp(mouseEvent);
}

CINDER_APP(FractalApp, ci::app::RendererGl(ci::app::RendererGl::Options().msaa(16)), [&](ci::app::App::Settings *settings) {
    settings->setHighDensityDisplayEnabled(true);
    settings->setWindowSize(500, 500);
})
