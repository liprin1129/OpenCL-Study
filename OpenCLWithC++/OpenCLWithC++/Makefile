CPP_COMMON = ../../Include
INC = -I $(CPP_COMMON)

# Change this variable to specify the device type
# to the OpenCL device type of choice. You can also
# edit the variable in the source.
ifndef DEVICE
	DEVICE = CL_DEVICE_TYPE_DEFAULT
endif

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
CPPC = clang++
CCFLAGS += -std=c++11 -stdlib=libc++
LIBS = -framework OpenCL

vadd: main.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@


clean:
	rm -f vadd
