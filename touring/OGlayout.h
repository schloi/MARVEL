/*
 * OGLayout.h
 *
 *  Created on: Jul 14, 2016
 *      Author: pippelmn
 */

#ifndef TOURING_OGLAYOUT_H_
#define TOURING_OGLAYOUT_H_

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

typedef struct Pair
{
        char *key;
        char *value;
} Pair;

typedef struct
{
        int sourceId;
        int targetId;

        Pair* attributes;
        int numAttributes;
        int maxAttributes;
} OgEdge;

typedef struct
{
        int nodeID;

        Pair* attributes;
        int numAttributes;
        int maxAttributes;
} OgNode;

typedef struct
{
        char *name;
        char *type;
        int id;      // removed d, from ids d0, d1 ... dn
        int forNode; // if forNode is 0, then its an edge attribute
} OgAttribute;

typedef struct
{
        int numNodes;
        OgNode* nodes;

        int numEdges;
        OgEdge* edges;

        int numAttr;
        OgAttribute *attrLookup;

} OgGraph;

// graph format

typedef enum
{
    FORMAT_DOT, FORMAT_GRAPHML, FORMAT_SVG, FORMAT_UNKNOWN
} GraphFormat;

typedef struct
{
// command line args

        int verbose;
        int cleanReverseEdges;
        int colorScheme;
        int skipLayout;

        int quadTreeLevel;
        float barnesHutTheta;
        int minLevelSize;
        float coarseningRate;
        float stepRatio;
        float optimalDistance;
        float convergenceThreshold;

        char* path_graph_in;
        char* path_graph_out;

// input graph
        GraphFormat giformat;
        GraphFormat goformat;
// overlap graph

        OgGraph *graph;

} OgLayoutContext;

#define NODE_IS_MERGED          (1 << 0)
#define NODE_HAS_PARENT         (1 << 1)
#define NODE_HAS_CHILD          (1 << 2)
#define NODE_HAS_MERGED_PARENT  (1 << 3)
#define NODE_HAS_COORDINATES    (1 << 4)
#define NODE_HAS_PATH_ID        (1 << 5)

#define EDGE_VISITED        (1 << 0)
#define EDGE_IGNORE         (1 << 1)

typedef struct ForceVector
{
        float x;
        float y;
} ForceVector;

typedef struct
{
        int flag;
        int source; // represent indexes
        int target; // from nodes (! not ids)
} Edge;

typedef struct Node
{
        int flag;
        int id;
        int idx;

        // 2D coordinates
        float x;
        float y;
        // 2d force vector
        ForceVector f;
        // if path is in attributes
        int pathID;

        int childIds[2];  // pointer for Nodes of parent graph otherwise NULL
        int parentId;     // node pointer into child graph otherwise NULL
        int firstEdgeIdx;
} Node;

typedef struct Graph
{
        int maxNodes;
        int curNodes;
        int maxEdges;
        int curEdges;

        Node *nodes;
        Edge *edges;

        int level; // coarsening level
        int maxNodeDegree;
        struct Graph *parent;
        struct Graph *child;
} Graph;

int cmpOgEdges(const void* a, const void *b);
int cmpNodesById(const void* a, const void *b);
int cmpEdgeBySourceNode(const void* a, const void *b);
void sortGraphEdges(Graph *g);
Graph* coarsenGraph(Graph *g, int verboseLevel);
void updateEdges(Graph *g);
void refineGraph(Graph *g);
void deleteGraph(Graph *g);

typedef struct YifanHuLayout
{
        float optimalDistance;
        float relativeStrength;
        float step;
        float initialStep;
        int progress;
        float stepRatio;
        int quadTreeMaxLevel;
        float barnesHutTheta;
        float convergenceThreshold;
        char adaptiveCooling;
        double energy0;
        double energy;
        int converged;
} YifanHuLayout;

typedef struct MultiLevelLayout
{
        Graph **graphs;
        int maxGraphs;
        int level;

        double minCoarseningRate;
        int minSize;

        float *bestCoordinateSet;

} MultiLevelLayout;

typedef struct QuadTree
{
        float posX;
        float posY;
        float size;
        float centerMassX;  // X and Y position of the center of mass
        float centerMassY;
        int mass;  // Mass of this tree (the number of nodes it contains)
        int maxLevel;
        struct QuadTree* children;
        int (*add)(struct QuadTree*, Node *);
        char isLeaf;
        float eps;

} QuadTree;

void initQuadTree(struct QuadTree *t, float posX, float posY, float size);
QuadTree* createQuadTree(int maxLevel);
void buildQuadTree(QuadTree* tree, Graph *g);
int addNode(QuadTree *t, Node* n);

void assimilateNode(QuadTree *t, Node *n);
int addToChildren(QuadTree *t, Node* n);
int firstAdd(QuadTree *t, Node* n);
int secondAdd(QuadTree *t, Node* n);
int rootAdd(QuadTree *t, Node *n);
int leafAdd(QuadTree *t, Node *n);

void divideTree(struct QuadTree *t);

ForceVector* createForceVectorByVector(ForceVector *f);
ForceVector* createForceVectorByPair(float x, float y);
ForceVector* createForceVector();
void addForceVectorToForceVector(ForceVector* f1, ForceVector *f2);
void multiplyForceVectorByConst(ForceVector* f1, float s);
void subtractForceVectorFromForceVector(ForceVector* f1, ForceVector *f2);
float getForceVectorEnergy(ForceVector *f);
float getForceVectorNorm(ForceVector *f);
ForceVector* normalizeForceVector(ForceVector *f);

typedef struct ElectricalForce
{
        float theta;
        float relativeStrength;
        float optimalDistance;
} ElectricalForce;

typedef struct SpringForce
{
        float optimalDistance;
} SpringForce;

ForceVector* calculateSpringForce(SpringForce* sf, Node *n1, Node *n2, float distance);
ForceVector* calculateElectricalForce(ElectricalForce* ef, Node* n, QuadTree *t, float distance);
ForceVector* calculateForce(ElectricalForce* fe, Node* n, QuadTree *t);
// ForceVector utils
float getForceVectorDistanceToQuadTree(Node *n, QuadTree *t);
float getForceVectorDistanceToNode(Node *n1, Node *n2);

// input / output

int read_dot(OgLayoutContext* octx);
int read_graphml(OgLayoutContext* octx);
void write_dot(OgLayoutContext *octx, Graph *g);
void write_svg(OgLayoutContext *octx, Graph *g);

#endif /* TOURING_OGLAYOUT_H_ */

