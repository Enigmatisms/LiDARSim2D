/// @author (Dinger: https://github.com/Dinngger) @copyright Dinger
#include <cmath>
#include "gl/viewer.hpp"
#include "gl/glMap.h"

int width = 1200, height = 900;
double cameraX = 600, cameraY = 450;

void display() {
    glEnable(GL_DEPTH_TEST);

    glClearColor(L_GRAY[0], L_GRAY[1], L_GRAY[2], 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, 0);
    glRotatef(0, 0, 0, 1);
    glScalef(1, 1, 1);

    /// @todo viz_function
    mapViz();

    glutSwapBuffers();
}

void reshape(int w, int h) {
    width = w;
    height = h;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, w, 0, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glutPostRedisplay();
}

void viewer(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(1200, 900);
    glutInitWindowPosition(200, 20);
    glutCreateWindow("MapViz");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutMainLoop();
}
