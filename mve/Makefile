# Compilateur et options
CXX = g++
CXXFLAGS = -std=c++17 $(shell pkg-config --cflags opencv4) -Iexternal
LDFLAGS = $(shell pkg-config --libs opencv4) -lavcodec -lavformat -lavutil -lswscale

# Dossiers
SRC_DIR = src/mvextractor
BIN_DIR = $(SRC_DIR)
EXE = $(BIN_DIR)/main

# Fichiers sources et objets
SRC = $(SRC_DIR)/main_cpp.cpp
OBJ = $(SRC:.cpp=.o)

# Règle principale
all: $(EXE)

# Compilation du programme
$(EXE): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Nettoyage des fichiers générés
clean:
	rm -f $(EXE) $(OBJ)

