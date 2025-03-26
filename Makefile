# or use gcc with CC=g++-11
CC=g++

# install some stuff with homebrew, clone raylib and raygui repo in the same directory level as this repo
# build raylib with: make PLATFORM=PLATFORM_DESKTOP
HOMEBREW=/opt/homebrew/Cellar

BOOST_PATH=$(HOMEBREW)/boost/1.85.0
GRAPHVIZ_PATH=$(HOMEBREW)/graphviz/12.2.1

PROJECT_ROOT=$(shell pwd)

#UNIVERSAL_FLAGS = -arch arm64 -arch x86_64
UNIVERSAL_FLAGS=

INCLUDE=-isystem$(BOOST_PATH)/include/ \
		-isystem$(GRAPHVIZ_PATH)/include 
LINK=-L$(GRAPHVIZ_PATH)/lib -lgvc -lcgraph \
	 -L$(BOOST_PATH)/lib/ -lboost_filesystem

DEBUG=-g
RELEASE=-O3 -DNDEBUG

# don't forget to clean if you change OPT
#OPT=$(DEBUG)
OPT=$(RELEASE)

FLAGS=-std=c++20 -Wall $(OPT) $(UNIVERSAL_FLAGS) $(INCLUDE) -c

SOURCES=game.cpp
ODIR=obj
OBJS=$(patsubst %.cpp,$(ODIR)/%.o,$(SOURCES))
DEPS=$(patsubst %.cpp,$(ODIR)/%.d,$(SOURCES))
#$(info DEPS=$(DEPS))

MAIN_OBJS=$(OBJS) $(ODIR)/main.o
TEST_OBJS=$(OBJS) $(ODIR)/test.o

beat-it: $(MAIN_OBJS)
	$(CC) $(UNIVERSAL_FLAGS) -o $(ODIR)/beat-it $(MAIN_OBJS) $(LINK)

test: $(TEST_OBJS)
	$(eval OPT=$(DEBUG))
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
