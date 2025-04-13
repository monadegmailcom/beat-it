# or use gcc with CC=g++-11
CC=g++

# install some stuff with homebrew, clone raylib and raygui repo in the same directory level as this repo
# build raylib with: make PLATFORM=PLATFORM_DESKTOP
HOMEBREW=/opt/homebrew/Cellar

BOOST_PATH=$(HOMEBREW)/boost/1.85.0
GRAPHVIZ_PATH=$(HOMEBREW)/graphviz/12.2.1
GPERF_PATH=$(HOMEBREW)/gperftools/2.16

PROJECT_ROOT=$(shell pwd)

#UNIVERSAL_FLAGS = -arch arm64 -arch x86_64
UNIVERSAL_FLAGS=

INCLUDE=-isystem$(BOOST_PATH)/include/ \
#		-isystem$(GRAPHVIZ_PATH)/include 
LINK=-L$(GPERF_PATH)/lib -lprofiler
# -L$(GRAPHVIZ_PATH)/lib -lgvc -lcgraph \
#	 -L$(BOOST_PATH)/lib/ -lboost_filesystem

DEBUG=-g -O3
RELEASE=-O3 -DNDEBUG

# don't forget to clean if you change OPT
#OPT=$(DEBUG)
#OPT=$(RELEASE)

# Set OPT based on the target
ifeq ($(MAKECMDGOALS), test)
    OPT=$(DEBUG)
endif

ifeq ($(MAKECMDGOALS), beat-it)
    OPT=$(RELEASE)
endif

$(info OPT=$(OPT))
FLAGS=-std=c++20 -Wall -pedantic $(OPT) $(UNIVERSAL_FLAGS) $(INCLUDE) -c

# Automatically reference all local .cpp files
SOURCES=$(wildcard *.cpp)
ODIR=obj
OBJS=$(patsubst %.cpp,$(ODIR)/%.o,$(SOURCES))
DEPS=$(patsubst %.cpp,$(ODIR)/%.d,$(SOURCES))
$(info DEPS=$(DEPS))

MAIN_OBJS=$(filter-out $(ODIR)/test.o, $(OBJS))
TEST_OBJS=$(filter-out $(ODIR)/main.o, $(OBJS))

beat-it: $(MAIN_OBJS)
	$(CC) $(UNIVERSAL_FLAGS) -o $(ODIR)/beat-it $(MAIN_OBJS) $(LINK)

test: $(TEST_OBJS)
	$(CC) $(UNIVERSAL_FLAGS) -o $(ODIR)/test $(TEST_OBJS) $(LINK)
#	$(CC) $(FLAGS) -o $(ODIR)/test.o test.cpp
#	$(CC) -o $(ODIR)/test $(LINK) $(ODIR)/test.o

$(ODIR)/%.o: %.cpp | $(ODIR)
	$(CC) $(FLAGS) -MMD -MP -c $< -o $@
#$(ODIR)/gui/%.o: gui/%.cpp | $(ODIR)/gui
#	$(CC) $(FLAGS) -MMD -MP -c $< -o $@

$(ODIR):
	mkdir -p $(ODIR)

#$(ODIR)/gui:
#	mkdir -p $(ODIR)/gui

-include $(DEPS)

clean:
	rm -f $(ODIR)/*.o $(ODIR)/beat-it $(ODIR)/test obj/*.d
