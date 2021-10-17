#include "gl/glMap.h"

std::unique_ptr<Walls> wall_ptr;
GLdouble borders[8][6] = {
    {1200, 0, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]},
    {0, 0, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]}, 
    {0, 900, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]},
    {1200, 900, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]},

    {30, 30, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]},
    {1170, 30, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]},
    {1170, 870, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]},
    {30, 870, 0, D_GRAY[0], D_GRAY[1], D_GRAY[2]}
};

void vertexCallback(GLvoid *vertex){
    const GLdouble *pointer;

    pointer = (GLdouble *) vertex;
    glColor3dv(pointer+3);
    glVertex3dv((double *)vertex);
}

void combineCallback(GLdouble coords[3], 
                    GLdouble *vertex_data[4],
                    GLfloat weight[4], GLdouble **dataOut ) {
    GLdouble *vertex;
    int i;

    vertex = (GLdouble *) malloc(3 * sizeof(GLdouble));
    vertex[0] = coords[0];
    vertex[1] = coords[1];
    vertex[2] = coords[2];
    for (i = 3; i < 7; i++)
       vertex[i] = weight[0] * vertex_data[0][i] 
                   + weight[1] * vertex_data[1][i]
                   + weight[2] * vertex_data[2][i] 
                   + weight[3] * vertex_data[3][i];
    *dataOut = vertex;
}

void beginCallback(GLenum which) {
   glBegin(which);
}

void endCallback(void) {
   glEnd();
}

void errorCallback(GLenum errorCode) {
   const GLubyte *estring;
   estring = gluErrorString(errorCode);
   fprintf (stderr, "Tessellation Error: %s\n", estring);
   exit (0);
}

void mapViz() {
    for (size_t i = 0; i < wall_ptr->size(); i++) {
        const Wall& w = wall_ptr->at(i);
        // glBegin(GL_POLYGON);
        // // glVertex3f(0, 0, 0);
        // glColor3f(D_GRAY[0], D_GRAY[1], D_GRAY[2]);
        // for (const Eigen::Vector3d& p : w)
        //     glVertex3f(p(0), p(1), 0.0);
        // glEnd();

        glBegin(GL_LINE_STRIP);
        glColor3f(BLUE[0], BLUE[1], BLUE[2]);
        for (const Eigen::Vector3d& p: w) {
            glVertex3f(p(0), p(1), 0.05);
        }
        glVertex3f(w.front()(0), w.front()(1), 0.0);
        glEnd();
        glBegin(GL_POINT);
        glColor3f(BLACK[0], BLACK[1], BLACK[2]);
        for (const Eigen::Vector3d& p: w) {
            glVertex3f(p(0), p(1), 0.0);
        }
        glEnd();
    }
    tesselation();
}

void tesselation() {
    size_t number = wall_ptr->size();
    GLuint startList = glGenLists(2);
    GLUtesselator* tobj = gluNewTess();
    gluTessCallback(tobj, GLU_TESS_VERTEX,
                    (GLvoid (*) ()) &glVertex3dv);
    gluTessCallback(tobj, GLU_TESS_BEGIN,
                    (GLvoid (*) ()) &beginCallback);
    gluTessCallback(tobj, GLU_TESS_END,
                    (GLvoid (*) ()) &endCallback);
    gluTessCallback(tobj, GLU_TESS_ERROR,
                    (GLvoid (*) ()) &errorCallback);

    glNewList(startList, GL_COMPILE);
    glShadeModel(GL_FLAT);
    gluTessBeginPolygon(tobj, NULL);
    for (size_t i = 0; i < number; i++) {
        Wall& w = wall_ptr->at(i);
        gluTessBeginContour(tobj);
        for (Eigen::Vector3d& pt: w)
            gluTessVertex(tobj, pt.data(), pt.data());
        gluTessEndContour(tobj);
    }
    // gluTessEndPolygon(tobj);
    glEndList();
    printf("Stage 1 completed.\n");
    // glVertex3dv
    gluTessCallback(tobj, GLU_TESS_VERTEX,
                   (GLvoid (*) ()) &vertexCallback);
    gluTessCallback(tobj, GLU_TESS_BEGIN,
                    (GLvoid (*) ()) &beginCallback);
    gluTessCallback(tobj, GLU_TESS_END,
                    (GLvoid (*) ()) &endCallback);
    gluTessCallback(tobj, GLU_TESS_ERROR,
                    (GLvoid (*) ()) &errorCallback);
    gluTessCallback(tobj, GLU_TESS_COMBINE,
                   (GLvoid (*) ()) &combineCallback);

    glNewList(startList + 1, GL_COMPILE_AND_EXECUTE);
    glShadeModel(GL_SMOOTH);
    gluTessProperty(tobj, GLU_TESS_WINDING_RULE,
                    GLU_TESS_WINDING_POSITIVE);
    // gluTessBeginPolygon(tobj, NULL);
        gluTessBeginContour(tobj);
        for (int i = 0; i < 8; i++) {
            gluTessVertex(tobj, borders[i], borders[i]);
        }
        gluTessEndContour(tobj);
    gluTessEndPolygon(tobj);
    glEndList();
}


/// @brief 矩形模式绘制矩形 圆形模式绘制圆形
void specialViz() {
    ;
}