#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

using namespace ci;
using namespace ci::app;
using namespace std;

class MandelbulbApp : public App {
  public:
	void setup() override;
	void mouseDown( MouseEvent event ) override;
	void update() override;
	void draw() override;
};

void MandelbulbApp::setup()
{
}

void MandelbulbApp::mouseDown( MouseEvent event )
{
}

void MandelbulbApp::update()
{
}

void MandelbulbApp::draw()
{
	gl::clear( Color( 0, 0, 0 ) ); 
}

CINDER_APP( MandelbulbApp, RendererGl )
