CXXFLAGS = -I include  -std=c++11 -O3

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp
#OPENCV = `pkg-config --cflags --libs opencv`
INCLUDES = -I/data_1/software/anaconda/envs/PSENet/include
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml
LIBDIRS = -L/data_1/software/anaconda/envs/PSENet/lib

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(INCLUDES) $(LIBDIRS) $(LIBS)
	# $(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC $(OPENCV)

clean:
	rm -rf $(LIB_SO)
