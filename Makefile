# or use gcc with CC=g++-11
CC=g++

# install some stuff with homebrew, clone raylib and raygui repo in the same directory level as this repo
# build raylib with: make PLATFORM=PLATFORM_DESKTOP
HOMEBREW=/opt/homebrew/Cellar

BOOST_PATH=$(HOMEBREW)/boost/1.87.0_1
LIBTORCH_PATH=/Users/wrqpjzc/source/libtorch
#GRAPHVIZ_PATH=$(HOMEBREW)/graphviz/12.2.1

PROJECT_ROOT=$(shell pwd)

#UNIVERSAL_FLAGS = -arch arm64 -arch x86_64
UNIVERSAL_FLAGS=

INCLUDE=-isystem$(BOOST_PATH)/include/ \
		-isystem$(LIBTORCH_PATH)/include/ \
		-isystem$(LIBTORCH_PATH)/include/torch/csrc/api/include/
LINK=-L$(LIBTORCH_PATH)/lib -Wl,-rpath,$(LIBTORCH_PATH)/lib -ltorch -ltorch_cpu -lc10
# -L$(GRAPHVIZ_PATH)/lib -lgvc -lcgraph \
#	 -L$(BOOST_PATH)/lib/ -lboost_filesystem

#DEBUG=-g
DEBUG=-g -O3
RELEASE=-O3 -DNDEBUG

# don't forget to clean if you change OPT

# Set OPT based on the target
ifeq ($(MAKECMDGOALS), test)
    OPT=$(DEBUG)
endif

# Set OPT for release targets
ifneq ($(filter beat-it shared,$(MAKECMDGOALS)),)
    OPT=$(RELEASE)
endif

$(info OPT=$(OPT))
FLAGS=-std=c++23 -Wall -pedantic $(OPT) $(UNIVERSAL_FLAGS) $(INCLUDE) -fPIC -c

# Automatically reference all local .cpp files
SOURCES=$(wildcard *.cpp) $(wildcard games/*.cpp)
ODIR=obj
OBJS=$(patsubst %.cpp,$(ODIR)/%.o,$(SOURCES))
DEPS=$(patsubst %.cpp,$(ODIR)/%.d,$(SOURCES))
$(info DEPS=$(DEPS))

MAIN_OBJS=$(filter-out $(ODIR)/test.o, $(OBJS))
TEST_OBJS=$(filter-out $(ODIR)/main.o, $(OBJS))
SHARED_OBJS=$(filter-out $(ODIR)/main.o $(ODIR)/test.o, $(OBJS))

beat-it: $(MAIN_OBJS)
	$(CC) $(UNIVERSAL_FLAGS) -o $(ODIR)/beat-it $(MAIN_OBJS) $(LINK)

test: $(TEST_OBJS)
	$(CC) $(UNIVERSAL_FLAGS) -o $(ODIR)/test $(TEST_OBJS) $(LINK)

shared: $(SHARED_OBJS)
	$(CC) $(UNIVERSAL_FLAGS) -shared -o $(ODIR)/libalphazero.so $(SHARED_OBJS) $(LINK)

$(ODIR)/%.o: %.cpp | $(ODIR)
	$(CC) $(FLAGS) -MMD -MP -c $< -o $@

$(ODIR):
	mkdir -p $(ODIR)/games

-include $(DEPS)

clean:
	rm -f $(ODIR)/*.o $(ODIR)/games/*.o $(ODIR)/beat-it $(ODIR)/test $(ODIR)/libalphazero.so $(ODIR)/*.d $(ODIR)/games/*.d
