/// @author (Dinger: https://github.com/Dinngger) @copyright Dinger
#pragma once
#include <GL/glut.h>
#include <functional>
#include "utils/consts.h"

void display();

void reshape(int w, int h);

void setKeysCallback(std::function<bool(u_char)> callback);

void viewer(int argc, char* argv[]);

