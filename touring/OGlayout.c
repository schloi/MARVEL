/*******************************************************************************************
 *
 *  Creates a Yifan Hu Layout from a given graph file.
 *
 *  input :     dot, graphml
 *  output:     dot, svg
 *  parameter:  adopted from Gephi (https://gephi.org/)
 *              -R         ... remove reverse edges from output, i.e. create a directed graph with arbitrary direction
 *              -q         ... QuadTree level (default: 10)
 *                             Maximum value to be used in the QuadTree representation. Greater values mean more accuracy
 *              -t         ... theta, Barnes Hut opening criteria (default: 1.2)
 *                             Smaller values mean more accuracy
 *              -l         ... Minimum level size (default: 3)
 *                             Minimum amount of nodes every level must have. Bigger values mean less levels
 *              -c         ... Coarsening rate (default: 0.75)
 *                             Minimum relative size (number of nodes) between two levels. Smaller values mean less levels
 *              -s         ... Step ratio (default: 0.97)
 *                             The ratio used to update the step size across iterations
 *              -d         ... Optimal distance (default: 100)
 *                             The natural length of the springs (edge length). Bigger values mean nodes will be further apart
 *
 *  Date    : August 2016
 *
 *  Author  : MARVEL Team
 *
 *******************************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <sys/param.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <time.h>

#include "OGlayout.h"

// todo change recursive function calls of calculateForce to get rid of ForceVector mallocs

// constants

#define DEFAULT_QUADTREE_LEVEL  10
#define DEFAULT_BARNESHUT_THETA 1.2
#define DEFAULT_MIN_LEVEL 3
#define DEFAULT_COARSEN_RATE 0.75
#define DEFAULT_STEP_RATIO  0.97
#define DEFAULT_OPTIMAL_DISTANCE 100

#define DEFAULT_CONVERGENCE_THRESHOLD 0.0001
#define DEFAULT_ADAPTIVE_COOLING 0
#define DEFAULT_RELATIVE_STRENGTH 0.2

// switches

#undef DEBUG_READ_DOT
#undef DEBUG_READ_GRAPHML

// getopt

extern char* optarg;
extern int optind, opterr, optopt;

int cmpAttributes(const void* a, const void *b)
{
    OgAttribute *at1 = (OgAttribute*) a;
    OgAttribute *at2 = (OgAttribute*) b;

    if (at1->forNode != at2->forNode)
        return at2->forNode - at1->forNode;

    return (at1->id - at2->id);
}

int cmpOgEdges(const void* a, const void *b)
{
    OgEdge *e1 = (OgEdge*) a;
    OgEdge *e2 = (OgEdge*) b;

    if (e1->sourceId != e2->sourceId)
        return e1->sourceId - e2->sourceId;

    return e1->targetId - e2->targetId;
}

int cmpNodesById(const void* a, const void *b)
{
    Node *n1 = (Node*) a;
    Node *n2 = (Node*) b;

    return n1->id - n2->id;
}

int cmpEdgeBySourceNode(const void* a, const void *b)
{
    Edge *e1 = (Edge*) a;
    Edge *e2 = (Edge*) b;

    if (e1->source != e2->source)
        return e1->source - e2->source;

    return e1->target - e2->target;
}

void assimilateNode(QuadTree *t, Node *n)
{
#if DEBUG
    printf("assimilate node: %f, %f, mass %d, cMass: %f. %f", n->x, n->y, t->mass, t->centerMassX, t->centerMassY);
#endif
    t->centerMassX = (t->mass * t->centerMassX + n->x) / (t->mass + 1);
    t->centerMassY = (t->mass * t->centerMassY + n->y) / (t->mass + 1);
    t->mass++;
#if DEBUG
    printf(" --> mass %d, cMass: %f. %f", t->mass, t->centerMassX, t->centerMassY);
#endif
}

int leafAdd(QuadTree *t, Node *n)
{
#if DEBUG
    printf("leafAdd: node id: %d, idx: %d pos (%f, %f)\n", n->id, n->idx, n->x, n->y);
#endif
    assimilateNode(t, n);
    return 1;
}

int addNode(QuadTree *t, Node* n)
{
    if (t->posX <= n->x && n->x <= t->posX + t->size && t->posY <= n->y && n->y <= t->posY + t->size)
    {
#if DEBUG
        printf("general add node: %f, %f in tree [%f, %f, - %f, %f]\n", n->x, n->y, t->posX, t->posY, t->posX + t->size, t->posY + t->size);
#endif
        return t->add(t, n);
    }
    else
    {
#if DEBUG
        printf("out of bounds\n");
#endif
        return 0;
    }
}

int addToChildren(QuadTree *t, Node* n)
{
#if DEBUG
    printf("addtochildren:  id %d, idx: %d (%.3f, %.3f)\n", n->id, n->idx, n->x, n->y);
#endif
    int i;
    for (i = 0; i < 4; i++)
    {
#if DEBUG
        printf("try to add node id %d, idx: %d (%.3f, %.3f) to child %d\n", n->id, n->idx, n->x, n->y, i);
#endif
        if (addNode(&(t->children[i]), n))
        {
#if DEBUG
            printf("success\n");
#endif
            return i + 1;
        }
    }
    return 0;
}

int rootAdd(QuadTree *t, Node *n)
{
#if DEBUG
    printf("rootAdd: node id: %d, idx: %d pos (%f, %f)\n", n->id, n->idx, n->x, n->y);
#endif
    assimilateNode(t, n);
    return addToChildren(t, n);
}

int secondAdd(QuadTree *t, Node* n)
{
#if DEBUG
    printf("secondAdd: node id: %d, idx: %d pos (%f, %f)\n", n->id, n->idx, n->x, n->y);
#endif
    divideTree(t);
    t->add = rootAdd;
    static Node dummy;
    dummy.x = t->centerMassX;
    dummy.y = t->centerMassY;
    dummy.id = -1234;
    addToChildren(t, &dummy);
    return t->add(t, n);
}

int firstAdd(QuadTree *t, Node* n)
{
#if DEBUG
    printf("firstAdd: node id: %d, idx: %d pos (%f, %f)\n", n->id, n->idx, n->x, n->y);
#endif
    t->mass = 1;
    t->centerMassX = n->x;
    t->centerMassY = n->y;

    if (t->maxLevel == 0)
    {
        t->add = leafAdd;
    }
    else
    {
        t->add = secondAdd;
    }

    return 1;
}

void initQuadTree(struct QuadTree *t, float posX, float posY, float size)
{
    t->posX = posX;
    t->posY = posY;
    t->size = size;
    t->isLeaf = 1;
    t->mass = 0;
    t->add = firstAdd;
    t->eps = (float) 1e-6;

#if DEBUG
    printf("initQuadTree: %f %f %f --> [%f, %f - %f, %f]\n", posX, posY, size, posX, posY, posX + size, posY + size);
#endif
}

void divideTree(struct QuadTree *t)
{
    float childSize = t->size / 2;

    initQuadTree(t->children, t->posX + childSize, t->posY + childSize, childSize);
    initQuadTree(t->children + 1, t->posX, t->posY + childSize, childSize);
    initQuadTree(t->children + 2, t->posX, t->posY, childSize);
    initQuadTree(t->children + 3, t->posX + childSize, t->posY, childSize);

    t->isLeaf = 0;
}

void deleteQuadTree(QuadTree* t, int maxLevel)
{
    int i;

    if (t->children != NULL)
    {
        for (i = 0; i < 4; i++)
            deleteQuadTree(t->children + i, maxLevel);
    }

    if (t->children != NULL)
        free(t->children);

    if (t->maxLevel == maxLevel)
        free(t);
}

QuadTree* createQuadTree(int maxLevel)
{
    size_t num = 0;
    int i;
    for (i = 1; i <= maxLevel; i++)
        num += pow(4, i);

    QuadTree* tree = (QuadTree*) malloc(sizeof(QuadTree) * (num + 2));
    assert(tree != NULL);
    bzero(tree, sizeof(QuadTree) * (num + 2));

    QuadTree** list = (QuadTree**) malloc(sizeof(QuadTree*) * ((maxLevel + 1) * 4));
    int n = 0;

    tree->maxLevel = maxLevel;
    list[n] = tree;

    QuadTree* pos = tree + 1;

    while (n >= 0)
    {
        QuadTree* t = list[n];
        n--;

        if (t->maxLevel <= 0)
        {
            t->children = NULL;
            continue;
        }

        t->children = pos;
        pos = pos + 4;
        for (i = 0; i < 4; i++, n++)
        {
            t->children[i].maxLevel = t->maxLevel - 1;
            list[n + 1] = t->children + i;
        }
    }
    free(list);

    return tree;
}

void buildQuadTree(QuadTree* tree, Graph *g)
{
#if DEBUG
    printf("buildQuadTree, glevel: %d, mlevel: %d\n", g->level, maxLevel);
#endif
    float minX = FLT_MAX;
    float maxX = -FLT_MAX;
    float minY = FLT_MAX;
    float maxY = -FLT_MAX;

    int i;
    for (i = 0; i < g->curNodes; i++)
    {
        minX = min(minX, g->nodes[i].x);
        maxX = max(maxX, g->nodes[i].x);
        minY = min(minY, g->nodes[i].y);
        maxY = max(maxY, g->nodes[i].y);
    }

    float size = max(maxY - minY, maxX - minX);
    initQuadTree(tree, minX, minY, size);

    for (i = 0; i < g->curNodes; i++)
    {
#if DEBUG
        printf("add node: id %d, idx %d pos(%f, %f)\n", g->nodes[i].id, g->nodes[i].idx, g->nodes[i].x, g->nodes[i].y);
#endif
        addNode(tree, (g->nodes + i));
    }
}

ForceVector* createForceVectorByVector(ForceVector *f)
{
    ForceVector *nf = (ForceVector*) malloc(sizeof(ForceVector));
    nf->x = f->x;
    nf->y = f->y;

    return nf;
}

ForceVector* createForceVectorByPair(float x, float y)
{
    ForceVector *nf = (ForceVector*) malloc(sizeof(ForceVector));
    nf->x = x;
    nf->y = y;

    return nf;
}

ForceVector* createForceVector()
{
    ForceVector *nf = (ForceVector*) malloc(sizeof(ForceVector));
    nf->x = 0;
    nf->y = 0;

    return nf;
}

void addForceVectorToForceVector(ForceVector* f1, ForceVector *f2)
{
    if (f1 != NULL && f2 != NULL)
    {
        f1->x += f2->x;
        f1->y += f2->y;
    }
}

void multiplyForceVectorByConst(ForceVector* f1, float s)
{
    if (f1 != NULL)
    {
        f1->x *= s;
        f1->y *= s;
    }
}

void subtractForceVectorFromForceVector(ForceVector* f1, ForceVector *f2)
{
    if (f1 != NULL && f2 != NULL)
    {
        f1->x -= f2->x;
        f1->y -= f2->y;
    }
}

float getForceVectorEnergy(ForceVector *f)
{
    return (f->x * f->x) + (f->y * f->y);
}

float getForceVectorNorm(ForceVector *f)
{
    return sqrtf(getForceVectorEnergy(f));
}

ForceVector* normalizeForceVector(ForceVector *f)
{
    static ForceVector fv =
    { 0, 0 };

    float norm = getForceVectorNorm(f);
    fv.x = f->x / norm;
    fv.y = f->y / norm;

    return &fv;
}

ForceVector* calculateElectricalForce(ElectricalForce *f, Node* n, QuadTree* t, float dist)
{
#if DEBUG
    printf(" ELECTRICAL_FORCE ");
#endif
    ForceVector *fv = createForceVector();

    fv->x = t->centerMassX - n->x;
    fv->y = t->centerMassY - n->y;

    float scale = -f->relativeStrength * f->optimalDistance * f->optimalDistance / (dist * dist);
    if (isnan(scale) || !isfinite(scale))
        scale = -1;
#if DEBUG
    printf("calclateElectricalForce: fv: %.3f, %.3f, scale %.3f, dist: %.3f node: %d %.3f, %.3f"
            " tree mass %.3f %.3f pos %.3f %.3f", fv->x, fv->y, scale, dist, n->id, n->x, n->y, t->centerMassX, t->centerMassY, t->posX, t->posY);
#endif
    multiplyForceVectorByConst(fv, scale);
#if DEBUG
    printf("-> final fv: %.3f, %.3f\n", fv->x, fv->y);
#endif
    return fv;
}

ForceVector* calculateSpringForce(SpringForce* f, Node *n1, Node *n2, float dist)
{
#if DEBUG
    printf(" SPRING_FORCE ");
#endif
    ForceVector *fv = createForceVector();

    fv->x = n2->x - n1->x;
    fv->y = n2->y - n1->y;

    if (((n1->flag & NODE_HAS_PATH_ID) && (n2->flag & NODE_HAS_PATH_ID)) && n1->pathID != n2->pathID)
    {
        multiplyForceVectorByConst(fv, (5 * dist / (f->optimalDistance)));
    }
//    else if (((n1->flag & NODE_HAS_PATH_ID) || (n2->flag & NODE_HAS_PATH_ID)) && n1->pathID != n2->pathID)
//    {
//        multiplyForceVectorByConst(fv, (0.8 * dist / (f->optimalDistance)));
//    }
    else
    {
        multiplyForceVectorByConst(fv, (dist / f->optimalDistance));
    }
    return fv;
}

float getForceVectorDistanceToQuadTree(Node *n, QuadTree *t)
{
    return (float) hypot(n->x - t->centerMassX, n->y - t->centerMassY);
}

float getForceVectorDistanceToNode(Node *n1, Node *n2)
{
    return (float) hypot(n1->x - n2->x, n1->y - n2->y);
}

ForceVector *calculateForce(ElectricalForce* ef, Node* n, QuadTree *t)
{
#if DEBUG
    printf("calculate force node %d %f %f and tree cmass %f, %f, mass %d, coord: %f, %f\n", n->id, n->x, n->y, t->centerMassX, t->centerMassY, t->mass, t->posX, t->posY);
#endif
    if (t->mass <= 0)
    {
#if DEBUG
        printf(" --> mass is 0\n");
#endif
        return NULL;
    }

    float distance = getForceVectorDistanceToQuadTree(n, t);
#if DEBUG
    printf("distance node id %d idx %d pos: %f %f, tree pos: %f, %f mass: %d cmass: %f %f", n->id, n->idx, n->x, n->y, t->posX, t->posY, t->mass, t->centerMassX, t->centerMassY);
#endif

    if (t->isLeaf || t->mass == 1)
    {
        if (distance < 1e-8)
        {
#if DEBUG
            printf(" --> return NULL\n");
#endif
            return NULL;
        }
#if DEBUG
        printf(" --> calculateElectricalForce(ef, n, t, distance)\n");
#endif
        return calculateElectricalForce(ef, n, t, distance);
    }

    if (distance * ef->theta > t->size)
    {
#if DEBUG
        printf("--> distance * theta > tree.size()\n");
#endif
        ForceVector *tmp = calculateElectricalForce(ef, n, t, distance);
        multiplyForceVectorByConst(tmp, t->mass);
        return tmp;
    }

    ForceVector *fr = createForceVector();

    int i;
    for (i = 0; i < 4; i++)
    {
#if DEBUG
        printf("calculate force of child : %d\n", i);
#endif
        ForceVector * tmp = calculateForce(ef, n, (t->children + i));
        if (tmp)
        {
            addForceVectorToForceVector(fr, tmp);
            free(tmp);
        }
    }
#if DEBUG
    printf(" final force: %f, %f\n", fr->x, fr->y);
#endif
    return fr;
}

char *trimwhitespace(char *str)
{
    char *end;

    // Trim leading space
    while (isspace(*str))
        str++;

    if (*str == 0)  // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while (end > str && isspace(*end))
        end--;

    // Write new null terminator
    *(end + 1) = '\0';

    return str;
}

int read_graphml(OgLayoutContext* octx)
{
    FILE* fileIn = fopen(octx->path_graph_in, "r");

    if (fileIn == NULL)
        return 0;

    char* line = NULL;
    size_t maxline = 0;

    OgGraph *graph;
    graph = (OgGraph*) malloc(sizeof(OgGraph));
    bzero(graph, sizeof(OgGraph));

    int nedges, nnodes, nattr, nEdgeAttr, nNodeAttr;
    nedges = nnodes = nattr = nEdgeAttr = nNodeAttr = 0;
    int maxNodes, maxEdges, maxAttr;
    maxNodes = maxEdges = 1000;
    maxAttr = 10;

    OgNode* nodes = (OgNode*) malloc(sizeof(OgNode) * maxNodes);
    if (nodes == NULL)
    {
        fprintf(stderr, "Cannot allocate node buffer!\n");
        exit(1);
    }
    bzero(nodes, sizeof(OgNode) * maxNodes);

    OgEdge* edges = (OgEdge*) malloc(sizeof(OgEdge) * maxEdges);
    if (edges == NULL)
    {
        fprintf(stderr, "Cannot allocate edge buffer!\n");
        exit(1);
    }
    bzero(edges, sizeof(OgEdge) * maxEdges);

    OgAttribute* attrLookup = (OgAttribute*) malloc(sizeof(OgAttribute) * maxAttr);
    if (attrLookup == NULL)
    {
        fprintf(stderr, "Cannot allocate attribute lookup buffer!\n");
        exit(1);
    }
    bzero(attrLookup, sizeof(OgAttribute) * maxAttr);

    int nline = 0;
    int len;
    char *pchrf, *pchrl;

    int nodeId;
    int maxNodeAttributes = 0;
    int maxEdgeAttributes = 0;

    while ((len = getline(&line, &maxline, fileIn)) > 0)
    {
//        printf("line: %s\n", line);
        nline++;

        char *tline = trimwhitespace(line);

        // ignore empty lines
        if (strlen(tline) == 0)
            continue;

//        printf("%s", tline);

        // if (strncmp(tline, "<key", 4) == 0)
        if (strstr(tline, "<key") != 0)
        {
#ifdef DEBUG_READ_GRAPHML
            printf("found attr: %d -> %s\n", nnodeattr + nedgeAttr, tline);
#endif

            if (nattr == maxAttr)
            {
                int prev = maxAttr;

                maxAttr = maxAttr * 1.2 + 10;
                attrLookup = (OgAttribute*) realloc(attrLookup, sizeof(OgAttribute) * maxAttr);
                if (attrLookup == NULL)
                {
                    fprintf(stderr, "Cannot increase attribute lookup buffer!\n");
                    exit(1);
                }

                bzero(attrLookup + prev, sizeof(OgAttribute) * (maxAttr - prev));
            }

            OgAttribute * cattr = attrLookup + nattr;
            size_t len;

            // parse attribute name
            pchrf = strstr(tline, "attr.name=\"");
            if (pchrf == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read attribute name at line %d\n", nline);
                exit(1);
            }

            pchrl = strchr(pchrf + 11, '\"');
            if (pchrl == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read closing quote \" for attribute name at line %d\n", nline);
                exit(1);
            }

            *pchrl = '\0';
            len = strlen(pchrf + 11);

            if(cattr->name == NULL)
                cattr->name = (char*) malloc(sizeof(char) * (len + 1));
            else
                cattr->name = (char*) realloc(cattr->name, sizeof(char) * (len + 1));

            assert(cattr->name != NULL);
            strncpy(cattr->name, pchrf + 11, len);
            cattr->name[len] = '\0';
            *pchrl = '\"';
#ifdef DEBUG_READ_GRAPHML
            printf("cattr->name: %s\n", cattr->name);
#endif
            // parse attribute type
            pchrf = strstr(pchrl, "attr.type=\"");
            if (pchrf == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read attr.type at line %d\n", nline);
                exit(1);
            }

            pchrl = strchr(pchrf + 11, '\"');
            if (pchrl == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read closing quote \" for attr.type at line %d\n", nline);
                exit(1);
            }

            *pchrl = '\0';
            len = strlen(pchrf + 11);

            if(cattr->type == NULL)
                cattr->type = (char*) malloc(sizeof(char) * (len + 1));
            else
                cattr->type = (char*) realloc(cattr->type, sizeof(char) * (len + 1));

            assert(cattr->type != NULL);
            strncpy(cattr->type, pchrf + 11, len);
            cattr->type[len] = '\0';
            *pchrl = '\"';
#ifdef DEBUG_READ_GRAPHML
            printf("cattr->type: %s\n", cattr->type);
#endif
            // parse attribute for
            pchrf = strstr(pchrl, "for=\"");
            if (pchrf == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read \"for\" at line %d\n", nline);
                exit(1);
            }

            pchrl = strchr(pchrf + 5, '\"');
            if (pchrl == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read closing quote \" for \"for\" at line %d\n", nline);
                exit(1);
            }

            *pchrl = '\0';
            if (strcmp(pchrf + 5, "node") == 0)
            {
                cattr->forNode = 1;
                nNodeAttr++;
            }
            else if (strcmp(pchrf + 5, "edge") == 0)
            {
                cattr->forNode = 0;
                nEdgeAttr++;
            }
            else
            {
                fprintf(stderr, "readGraphml: Unknown value \"%s\" in \"for\"  at line %d. Expected node or edge!\n", pchrf + 4, nline);
                exit(1);
            }
            *pchrl = '\"';
#ifdef DEBUG_READ_GRAPHML
            printf("cattr->forNode: %d\n", cattr->forNode);
#endif
            // parse attribute id
            pchrf = strstr(pchrl, "id=\"");
            if (pchrf == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read id at line %d\n", nline);
                exit(1);
            }

            pchrl = strchr(pchrf + 4, '\"');
            if (pchrl == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read closing quote \" for id at line %d\n", nline);
                exit(1);
            }

            *pchrl = '\0';
            // if id is equal to name than skip attribute
            if (strcmp(cattr->name, pchrf + 4) == 0)
            {
                ;
            }
            else
            {
                if (*(pchrf + 4) != 'd')
                {
                    fprintf(stderr, "readGraphml: invalid id format at line %d. (expected: dINT)\n", nline);
                    exit(1);
                }

                cattr->id = atoi(pchrf + 5);
                *pchrl = '\"';
#ifdef DEBUG_READ_GRAPHML
                printf("cattr->id: %d\n", cattr->id);
#endif

                nattr++;
            }
        }

        // parse edges
        else if (strncmp(tline, "<edge", 5) == 0)
        {
#ifdef DEBUG_READ_GRAPHML
            printf("found edge: %d\n", nedges);
#endif
            if (nedges == maxEdges)
            {
                int prev = maxEdges;

                maxEdges = maxEdges * 1.2 + 10;
                edges = (OgEdge*) realloc(edges, sizeof(OgEdge) * maxEdges);
                if (edges == NULL)
                {
                    fprintf(stderr, "Cannot increase edge buffer!\n");
                    exit(1);
                }

                bzero(edges + prev, sizeof(OgEdge) * (maxEdges - prev) );
            }

            // parse source id
            pchrf = strstr(tline, "source=\"");
            if (pchrf == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read edge source id at line %d\n", nline);
                exit(1);
            }

            pchrl = strchr(pchrf + 8, '\"');
            if (pchrl == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read closing quote \" for source attribute at line %d\n", nline);
                exit(1);
            }

            *pchrl = '\0';

            nodeId = atoi(pchrf + 8);

#ifdef DEBUG_READ_GRAPHML
            printf("parsed source nodeID: %d\n", nodeId);
#endif

            OgEdge* edge = edges + nedges;
            bzero(edge, sizeof(OgEdge));

            edge->maxAttributes = max(1, maxEdgeAttributes);
            edge->numAttributes = 0;
            edge->attributes = (Pair*) realloc(edge->attributes, sizeof(Pair) * (edge->maxAttributes));
            assert(edge->attributes != NULL);
            edge->sourceId = nodeId;

            *pchrl = '\"';

            // parse target id
            pchrf = strstr(tline, "target=\"");
            if (pchrf == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read edge target id at line %d\n", nline);
                exit(1);
            }

            pchrl = strchr(pchrf + 8, '\"');
            if (pchrl == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read closing quote \" for target attribute at line %d\n", nline);
                exit(1);
            }

            *pchrl = '\0';

            nodeId = atoi(pchrf + 8);
#ifdef DEBUG_READ_GRAPHML
            printf("parsed target nodeID: %d\n", nodeId);
#endif

            edge->targetId = nodeId;

            *pchrl = '\"';

            // parse edge attributes until closing edge tag is found
            while ((len = getline(&line, &maxline, fileIn)) > 0)
            {
                nline++;
                tline = trimwhitespace(line);

                pchrf = strstr(tline, "</edge>");
                if (pchrf != NULL)
                    break;

                pchrf = strstr(tline, "<data");
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected <data tag!\n", nline);
                    continue;
                }
                pchrf = strstr(tline + 5, "key=\"");
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected key attribute!\n", nline);
                    continue;
                }

                pchrl = strchr(pchrf + 5, '\"');
                if (pchrl == NULL)
                {
                    printf("ignore line %d, missing closing quote for key attribute!\n", nline);
                    continue;
                }
                *pchrl = '\0';

                // allocate attribute name buffer
                size_t len = strlen(pchrf + 5);
                Pair *attribute = edge->attributes + edge->numAttributes;
                attribute->key = (char*) malloc(sizeof(char) * (len + 1));
                assert(attribute->key != NULL);
                strncpy(attribute->key, pchrf + 5, len);
                attribute->key[len] = '\0';
                *pchrl = '\"';

                pchrf = strchr(pchrl, '>');
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected closing bracket \">\" of data tag!\n", nline);
                    continue;
                }

                pchrl = strstr(pchrf, "</data>");
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected closing tag </data>!\n", nline);
                    continue;
                }

                *pchrl = '\0';
                // allocate attribute value buffer
                len = strlen(pchrf + 1);

                attribute->value = (char*) malloc(sizeof(char) * (len + 1));
                assert(attribute->value != NULL);
                strncpy(attribute->value, pchrf + 1, len);
                attribute->value[len] = '\0';
                *pchrl = '<';

#ifdef DEBUG_READ_GRAPHML
                printf("parsed attribute %d: \"%s\" with value: \"%s\"\n", edge->numAttributes, attribute->key, attribute->value);
#endif

                edge->numAttributes++;
                if (maxEdgeAttributes < edge->numAttributes)
                    maxEdgeAttributes = edge->numAttributes;

                if (edge->numAttributes == edge->maxAttributes)
                {
                    edge->maxAttributes = (edge->numAttributes * 1.2 + 2);
                    edge->attributes = (Pair*) realloc(edge->attributes, sizeof(Pair) * (edge->maxAttributes + 1));
                    assert(edge->attributes != NULL);
                }
            }
            nedges++;
        }

        // parse nodes
        // else if (strncmp(tline, "<node", 5) == 0)
        else if (strstr(tline, "<node") != 0)
        {
            if (nnodes == maxNodes)
            {
                int prev = maxNodes;

                maxNodes = maxNodes * 1.2 + 10;
                nodes = (OgNode*) realloc(nodes, sizeof(OgNode) * maxNodes);
                if (nodes == NULL)
                {
                    fprintf(stderr, "Cannot increase node buffer!\n");
                    exit(1);
                }

                bzero(nodes + prev, sizeof(OgNode) * (maxNodes - prev));
            }

            // parse id
            pchrf = strstr(tline, "id=\"");
            if (pchrf == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read node id at line %d\n", nline);
                exit(1);
            }
            pchrl = strchr(pchrf + 4, '\"');
            if (pchrl == NULL)
            {
                fprintf(stderr, "readGraphml: cannot read node id at line %d\n", nline);
                exit(1);
            }

            *pchrl = '\0';

            nodeId = atoi(pchrf + 4);

#ifdef DEBUG_READ_GRAPHML
            printf("parsed node ID: %d\n", nodeId);
#endif
            OgNode * node = nodes + nnodes;
            bzero(node, sizeof(OgNode));

            node->maxAttributes = max(1, maxNodeAttributes);
            node->numAttributes = 0;
            node->attributes = (Pair*) realloc(node->attributes, sizeof(Pair) * (node->maxAttributes));
            node->nodeID = nodeId;
            assert(node->attributes != NULL);

            *pchrl = '\"';

            // parse node attributes until closing node tag is found
            while ((len = getline(&line, &maxline, fileIn)) > 0)
            {
                nline++;
                tline = trimwhitespace(line);

                pchrf = strstr(tline, "</node>");
                if (pchrf != NULL)
                    break;

                pchrf = strstr(tline, "<data");
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected <data tag!\n", nline);
                    continue;
                }
                pchrf = strstr(tline + 5, "key=\"");
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected key attribute!\n", nline);
                    continue;
                }

                pchrl = strchr(pchrf + 5, '\"');
                if (pchrl == NULL)
                {
                    printf("ignore line %d, missing closing quote for key attribute!\n", nline);
                    continue;
                }
                *pchrl = '\0';

                // allocate attribute name buffer
                size_t len = strlen(pchrf + 5);
                Pair *attribute = node->attributes + node->numAttributes;
                attribute->key = (char*) malloc(sizeof(char) * (len + 1));
                assert(attribute->key != NULL);
                strncpy(attribute->key, pchrf + 5, len);
                attribute->key[len] = '\0';
                *pchrl = '\"';

                pchrf = strchr(pchrl, '>');
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected closing bracket \">\" of data tag!\n", nline);
                    continue;
                }

                pchrl = strstr(pchrf, "</data>");
                if (pchrf == NULL)
                {
                    printf("ignore line %d, expected closing tag </data>!\n", nline);
                    continue;
                }

                *pchrl = '\0';
                // allocate attribute value buffer
                len = strlen(pchrf + 1);

                attribute->value = (char*) malloc(sizeof(char) * (len + 1));
                assert(attribute->value != NULL);
                strncpy(attribute->value, pchrf + 1, len);
                attribute->value[len] = '\0';
                *pchrl = '<';

#ifdef DEBUG_READ_GRAPHML
                printf("parsed attribute %d: \"%s\" with value: \"%s\"\n", node->numAttributes, attribute->key, attribute->value);
#endif

                node->numAttributes++;
                if (maxNodeAttributes < node->numAttributes)
                    maxNodeAttributes = node->numAttributes;

                if (node->numAttributes == node->maxAttributes)
                {
                    node->maxAttributes = (node->numAttributes * 1.2 + 2);
                    node->attributes = (Pair*) realloc(node->attributes, sizeof(Pair) * (node->maxAttributes + 1));
                    assert(node->attributes != NULL);
                }
            }
            nnodes++;
        }
        // ignore rest
    }

    graph->numEdges = nedges;
    graph->edges = edges;

    graph->numNodes = nnodes;
    graph->nodes = nodes;

    qsort(attrLookup, nattr, sizeof(OgAttribute), cmpAttributes);

    // sanity check
    int i;
    for (i = 0; i < nattr; i++)
    {
        if (i < nNodeAttr)
            assert(attrLookup[i].forNode == 1);

        assert(i == attrLookup[i].id);
    }

    graph->numAttr = nattr;
    graph->attrLookup = attrLookup;

    octx->graph = graph;

    return 1;
}

int read_dot(OgLayoutContext* octx)
{
    FILE* fileIn = fopen(octx->path_graph_in, "r");

    if (fileIn == NULL)
        return 0;

    char* line = NULL;
    size_t maxline = 0;

    int nnodes = 0;
    int nedges = 0;
    OgEdge* edges;
    OgNode* nodes;

    int maxNodes = 1000;
    nodes = (OgNode*) malloc(sizeof(OgNode) * maxNodes);
    if (nodes == NULL)
    {
        fprintf(stderr, "Cannot allocate node buffer!\n");
        exit(1);
    }
    bzero(nodes, sizeof(OgNode) * maxNodes);

    int maxEdges = 1000;
    edges = (OgEdge*) malloc(sizeof(OgEdge) * maxEdges);
    if (edges == NULL)
    {
        fprintf(stderr, "Cannot allocate edge buffer!\n");
        exit(1);
    }
    bzero(edges, sizeof(OgEdge) * maxEdges);

    int nline = 0;
    int len;
    char *pchrf, *pchrl;

    int nodeId;
    int maxNodeAttributes = 0;
    int maxEdgeAttributes = 0;

    while ((len = getline(&line, &maxline, fileIn)) > 0)
    {
#if DEBUG
        printf("line: %s\n", line);
#endif
        nline++;

        char *tline = trimwhitespace(line);

        if (!isdigit(tline[0]))
            continue;

        // get first space
        pchrf = strchr(tline, ' ');

        *pchrf = '\0';
        pchrl = strchr(tline, '-');

        if (pchrl == NULL) // its a node
        {
            if (nnodes == maxNodes)
            {
                maxNodes = maxNodes * 1.2 + 10;
                nodes = (OgNode*) realloc(nodes, sizeof(OgNode) * maxNodes);
                if (nodes == NULL)
                {
                    fprintf(stderr, "Cannot increase node buffer!\n");
                    exit(1);
                }
            }

            // parse nodeID
            nodeId = atoi(tline);
            OgNode * node = nodes + nnodes;
            bzero(node, sizeof(OgNode));

            node->maxAttributes = max(1, maxNodeAttributes);
            node->numAttributes = 0;
            node->attributes = (Pair*) realloc(node->attributes, sizeof(Pair) * (node->maxAttributes));
            node->nodeID = nodeId;
            assert(node->attributes != NULL);

#ifdef DEBUG_READ_DOT
            printf("parsed nodeID: %d\n", nodeId);
#endif
            // parse node attributes
            *pchrf = ' ';
            pchrl = strchr(tline, '[');
            if (pchrl == NULL)
            {
#ifdef DEBUG_READ_DOT
                printf("node has no attributes\n");
#endif
            }
            else
            {
                pchrf = pchrl;
                while ((pchrl = strchr(pchrf, '=')) != NULL)
                {
                    if (*(pchrl + 1) == '\"')
                    {
                        // get attribute name
                        *pchrl = '\0';
                        if (*pchrf == '[')
                            pchrf++;

                        // allocate attribute name buffer
                        size_t len = strlen(pchrf);
                        Pair *attribute = node->attributes + node->numAttributes;
                        attribute->key = (char*) malloc(sizeof(char) * (len + 1));
                        assert(attribute->key != NULL);
                        strncpy(attribute->key, pchrf, len);
                        attribute->key[len] = '\0';

                        *pchrl = '=';

                        pchrf = pchrl;

                        pchrl = strchr(pchrf, ' ');
                        while (pchrl != NULL && *(pchrl - 1) != ',') // ... the attribute value contains itself spaces
                            pchrl = strchr(pchrl + 1, ' ');

                        if (pchrl == NULL)
                            pchrl = strchr(pchrf, ']');
                        if (pchrl == NULL)
                        {
#ifdef DEBUG_READ_DOT
                            printf("no values for last attribute");
#endif
                            continue;
                        }

                        *pchrl = '\0';
                        // allocate attribute value buffer
                        len = strlen(pchrf);

                        attribute->value = (char*) malloc(sizeof(char) * (len + 1));
                        assert(attribute->value != NULL);

                        char *from, *to;

                        if (pchrf[len - 1] == ',')
                            to = pchrf + len - 3;
                        else
                            to = pchrf + len - 2; // last attribute

                        from = pchrf + 2;

                        if (from > to) // empty attribute
                            attribute->value[0] = '\0';
                        else
                        {
                            strncpy(attribute->value, from, to - from + 1);
                            attribute->value[to - from + 1] = '\0';
                        }

#ifdef DEBUG_READ_DOT
                        printf("parsed attribute %d: \"%s\" with value: \"%s\"\n", node->numAttributes, attribute->key, attribute->value);
#endif
                        *pchrl = ' ';

                        node->numAttributes++;
                        if (maxNodeAttributes < node->numAttributes)
                            maxNodeAttributes = node->numAttributes;

                        if (node->numAttributes == node->maxAttributes)
                        {
                            node->maxAttributes = (node->numAttributes * 1.2 + 2);
                            node->attributes = (Pair*) realloc(node->attributes, sizeof(Pair) * (node->maxAttributes + 1));
                            assert(node->attributes != NULL);
                        }
                    }
                    pchrf = pchrl + 1;
                }
            }
            nnodes++;
        }
        else // its an edge
        {
            if (nedges == maxEdges)
            {
                maxEdges = maxEdges * 1.2 + 10;
                edges = (OgEdge*) realloc(edges, sizeof(OgEdge) * maxEdges);
                if (edges == NULL)
                {
                    fprintf(stderr, "Cannot increase edge buffer!\n");
                    exit(1);
                }
            }

            // parse source and target node ids
            *pchrl = '\0';

            nodeId = atoi(tline);
#ifdef DEBUG_READ_DOT
            printf("parsed source nodeID: %d\n", nodeId);
#endif

            OgEdge* edge = edges + nedges;
            bzero(edge, sizeof(OgEdge));

            edge->maxAttributes = max(1, maxEdgeAttributes);
            edge->numAttributes = 0;
            edge->attributes = (Pair*) realloc(edge->attributes, sizeof(Pair) * (edge->maxAttributes));
            edge->sourceId = nodeId;
            assert(edge->attributes != NULL);
            *pchrl = '-';

            while (!isdigit(*pchrl))
                pchrl++;

            assert(pchrf != pchrl);
            nodeId = atoi(pchrl);

#ifdef DEBUG_READ_DOT
            printf("parsed target nodeID: %d\n", nodeId);
#endif
            edge->targetId = nodeId;

            // parse edge attributes
            *pchrf = ' ';
            pchrl = strchr(tline, '[');
            if (pchrl == NULL)
            {
#ifdef DEBUG_READ_DOT
                printf("edge has not attributes\n");
#endif
            }
            else
            {
                pchrf = pchrl;
                while ((pchrl = strchr(pchrf, '=')) != NULL)
                {
                    if (*(pchrl + 1) == '\"')
                    {
                        // get attribute name
                        *pchrl = '\0';
                        if (*pchrf == '[')
                            pchrf++;

                        // allocate attribute name buffer
                        size_t len = strlen(pchrf);
                        Pair *attribute = edge->attributes + edge->numAttributes;
                        attribute->key = (char*) malloc(sizeof(char) * (len + 1));
                        assert(attribute->key != NULL);
                        strncpy(attribute->key, pchrf, len);
                        attribute->key[len] = '\0';

                        *pchrl = '=';

                        pchrf = pchrl;

                        pchrl = strchr(pchrf, ' ');
                        while (pchrl != NULL && *(pchrl - 1) != ',') // ... the attribute value contains itself spaces
                            pchrl = strchr(pchrl + 1, ' ');

                        if (pchrl == NULL)
                            pchrl = strchr(pchrf, ']');
                        if (pchrl == NULL)
                        {
#ifdef DEBUG_READ_DOT
                            printf("no values for last attribute");
#endif
                            continue;
                        }

                        *pchrl = '\0';
                        // allocate attribute value buffer
                        len = strlen(pchrf);

                        attribute->value = (char*) malloc(sizeof(char) * (len + 1));
                        assert(attribute->value != NULL);

                        char *from, *to;

                        if (pchrf[len - 1] == ',')
                            to = pchrf + len - 3;
                        else
                            to = pchrf + len - 2; // last attribute

                        from = pchrf + 2;

                        if (from > to) // empty attribute
                            attribute->value[0] = '\0';
                        else
                        {
                            strncpy(attribute->value, from, to - from + 1);
                            attribute->value[to - from + 1] = '\0';
                        }

#ifdef DEBUG_READ_DOT
                        printf("parsed attribute %d: \"%s\" with value: \"%s\"\n", edge->numAttributes, edge->attributes->key, edge->attributes->value);
#endif
                        *pchrl = ' ';

                        edge->numAttributes++;
                        if (maxEdgeAttributes < edge->numAttributes)
                            maxEdgeAttributes = edge->numAttributes;

                        if (edge->numAttributes == edge->maxAttributes)
                        {
                            edge->maxAttributes = (edge->numAttributes * 1.2 + 2);
                            edge->attributes = (Pair*) realloc(edge->attributes, sizeof(Pair) * (edge->maxAttributes + 1));
                            assert(edge->attributes != NULL);
                        }
                    }
                    pchrf = pchrl + 1;
                }
            }
            nedges++;
        }
    }

    OgGraph *graph = (OgGraph*) malloc(sizeof(OgGraph));
    assert(graph != NULL);
    bzero(graph, sizeof(OgGraph));

    graph->numEdges = nedges;
    graph->edges = edges;

    graph->numNodes = nnodes;
    graph->nodes = nodes;

    octx->graph = graph;
    return 1;
}

void write_svg(OgLayoutContext *octx, Graph *g)
{
    FILE *f = fopen(octx->path_graph_out, "w");

    if (f == NULL)
    {
        fprintf(stderr, "Cannot write graph into file: %s!\n", octx->path_graph_out);
    }

    int nnodes, nedges;
    nnodes = nedges = 0;

    float minX, minY;
    float maxX, maxY;

    minX = minY = FLT_MAX;
    maxX = maxY = -FLT_MAX;

    int i, j;
    for (i = 0; i < g->curNodes; i++)
    {
        Node *n = g->nodes + i;

        if (n->x < minX)
            minX = n->x;

        if (n->y < minY)
            minY = n->y;

        if (n->x > maxX)
            maxX = n->x;

        if (n->y > maxY)
            maxY = n->y;
    }

    // move coordinates into positive range
    if (minX < 0 || minY < 0)
    {
        if (minX < 0)
            minX -= octx->optimalDistance;

        if (minY < 0)
            minY -= octx->optimalDistance;

        for (i = 0; i < g->curNodes; i++)
        {
            Node *n = g->nodes + i;

            if (minX < 0)
                n->x += -minX;

            if (minY < 0)
                n->y += -minY;
        }
    }

    // write header
    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\"\n");
    fprintf(f, "     xmlns:xlink=\"http://www.w3.org/1999/xlink\"\n");
    fprintf(f, "     xmlns:ev=\"http://www.w3.org/2001/xml-events\"\n");
    fprintf(f, "     version=\"1.1\" baseProfile=\"full\"\n");
    fprintf(f, "     width=\"%dpt\" height=\"%dpt\">\n", (int) ceil(minX < 0 ? maxX - minX : maxX), (int) ceil(
            minY < 0 ? maxY - minY : maxY));
    fprintf(f, "  <title>OGLayout %s</title>\n", octx->path_graph_in);

    // graph
    Node dummy;
    // edges
    fprintf(f, "  <g id=\"edges\">\n");
    if (octx->cleanReverseEdges)
    {
        for (i = 0; i < octx->graph->numEdges; i++)
        {
            OgEdge *e = octx->graph->edges + i;
            if (e->sourceId > e->targetId)
            {
                int tmp = e->sourceId;
                e->sourceId = e->targetId;
                e->targetId = tmp;
            }
        }
        qsort(octx->graph->edges, octx->graph->numEdges, sizeof(OgEdge), cmpOgEdges);
    }

    for (i = 0; i < octx->graph->numEdges; i++)
    {
        OgEdge *edge = octx->graph->edges + i;

        int use = -1;
        if (octx->cleanReverseEdges)
        {
            if (i + 1 < octx->graph->numEdges && edge->sourceId == octx->graph->edges[i + 1].sourceId && edge->targetId == octx->graph->edges[i + 1].targetId)
            {
                for (j = 0; j < edge->numAttributes; j++)
                {
                    if (octx->graph->numAttr) // i.e. attributes must be translated via idNUM into attribute name
                    {
                        int keyIdx = atoi(octx->graph->edges[i].attributes[j].key + 1);
                        assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);

                        if ((strcmp(octx->graph->attrLookup[keyIdx].name, "path") == 0) && (atoi(edge->attributes[j].value) >= 0))
                        {
                            use = i;
                            break;
                        }
                    }
                    else
                    {
                        if (strcmp(edge->attributes[j].key, "path") == 0)
                        {
                            if (atoi(edge->attributes[j].value) >= 0)
                            {
                                use = i;
                                break;
                            }
                        }
                    }
                }
                if (use < 0)
                {
                    if (octx->graph->numAttr) // i.e. attributes must be translated via idNUM into attribute name
                    {
                        for (j = 0; j < octx->graph->edges[i + 1].numAttributes; j++)
                        {
                            int keyIdx = atoi(octx->graph->edges[i + 1].attributes[j].key + 1);
                            assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);

                            if ((strcmp(octx->graph->attrLookup[keyIdx].name, "path") == 0) && (atoi(octx->graph->edges[i + 1].attributes[j].value) >= 0))
                            {
                                use = i + 1;
                                break;
                            }
                        }
                    }
                    else
                    {
                        for (j = 0; j < octx->graph->edges[i + 1].numAttributes; j++)
                        {
                            if (strcmp(octx->graph->edges[i + 1].attributes[j].key, "path") == 0)
                            {
                                if (atoi(octx->graph->edges[i + 1].attributes[j].value) >= 0)
                                {
                                    use = i + 1;
                                    break;
                                }
                            }
                        }
                    }

                }

                if (use < 0 || use > i) // take reverse edge
                    edge = octx->graph->edges + i + 1;

                i += 1;
            }
        }

        dummy.id = edge->sourceId;
        Node *c = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
        assert(c != NULL);

        fprintf(f, "    <path d=\"M %f %f", c->x, c->y);

        dummy.id = edge->targetId;
        Node *d = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
        assert(d != NULL);

        fprintf(f, " L %f %f\"", d->x, d->y);

        int foundColor = -1;
        int foundPath = -1;

        for (j = 0; j < edge->numAttributes; j++)
        {
            if (octx->graph->numAttr) // i.e. attributes must be translated via idNUM into attribute name
            {
                int keyIdx = atoi(edge->attributes[j].key + 1);
                assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);

                if (strcmp(octx->graph->attrLookup[keyIdx].name, "color") == 0)
                    foundColor = j;

                if (strcmp(octx->graph->attrLookup[keyIdx].name, "path") == 0)
                    foundPath = j;
            }
            else
            {
                if (strcmp(edge->attributes[j].key, "color") == 0)
                {
                    foundColor = j;
                }
                if (strcmp(edge->attributes[j].key, "path") == 0)
                {
                    foundPath = j;
                }
            }
        }

        switch (octx->colorScheme)
        {
            case 1:
            {
                if (foundPath >= 0 && atoi(edge->attributes[foundPath].value) >= 0)
                    fprintf(f, " style=\"fill:none;stroke:red;");
                else
                    fprintf(f, " style=\"fill:none;stroke:black;");
            }
                break;
            case 2:
                fprintf(f, " style=\"fill:none;stroke:black;");
                break;

            case 0:
            default:
            {
                if (foundColor < 0)
                    fprintf(f, " style=\"fill:none;stroke:black;");
                else
                    fprintf(f, " style=\"fill:none;stroke:%s;", edge->attributes[foundColor].value);
            }
                break;
        }

        int weight = 1;
        if (foundPath >= 0)
        {
            int path = atoi(edge->attributes[foundPath].value);
            if (path >= 0)
                weight *= 10;
        }
        fprintf(f, "stroke-width:%d;\"/>\n", weight);
        nedges++;
    }
    // closing edge group
    fprintf(f, "  </g>\n");
    // nodes
    fprintf(f, "  <g id=\"nodes\">\n");
    for (i = 0; i < octx->graph->numNodes; i++)
    {
        OgNode *node = octx->graph->nodes + i;

        dummy.id = node->nodeID;
        Node *c = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
        assert(c != NULL);

        if (c->id == 231953 || c->id == 338031)
        {
            fprintf(f, "    <circle style=\"fill:red;stroke:red;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", c->x, c->y);
        }
        else if (c->id == 401251 || c->id == 430634)
        {
            fprintf(f, "    <circle style=\"fill:orange;stroke:orange;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", c->x, c->y);
        }

        else if (octx->colorScheme > 0)
        {
            if (c->pathID >= 0)
            {
                if (octx->colorScheme == 1)
                    fprintf(f, "    <circle style=\"fill:red;stroke:red;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", c->x, c->y);
                else
                    // colorScheme is 2
                    fprintf(f, "    <circle style=\"fill:black;stroke:black;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", c->x, c->y);
            }
            else
                fprintf(f, "    <circle style=\"fill:white;stroke:black;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", c->x, c->y);
        }
        else
        {
            int foundColor = 0;
            for (j = 0; j < node->numAttributes; j++)
            {
                if (octx->graph->numAttr) // i.e. attributes must be translated via idNUM into attribute name
                {
                    int keyIdx = atoi(node->attributes[j].key + 1);
                    assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);

                    if (strcmp(octx->graph->attrLookup[keyIdx].name, "color") == 0)
                    {
                        foundColor = 1;
                        fprintf(f, "    <circle style=\"fill:%s;stroke:black;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", node->attributes[j].value, c->x, c->y);
                        break;
                    }

                }
                else
                {
                    if (strcmp(node->attributes[j].key, "color") == 0)
                    {
                        fprintf(f, "    <circle style=\"fill:%s;stroke:black;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", node->attributes[j].value, c->x, c->y);
                        foundColor = 1;
                        break;
                    }
                }

#ifdef DEBUG
                if (strcmp(node->attributes[j].key, "read") == 0)
                fprintf(f, "    <text text-anchor=\"middle\" x=\"%f\" y=\"%f\" style=\"fill:red;stroke:red;font-family:Bitstream Vera Sans;font-size:12.00pt;\">%s</text>\n", c->x, c->y, node->attributes[j].value);
#endif
            }

            if (!foundColor)
                fprintf(f, "    <circle style=\"fill:black;stroke:black;stroke-width:2;\" cx=\"%f\" cy=\"%f\" r=\"10\"/>\n", c->x, c->y);
        }
        nnodes++;
    }
    // closing node group
    fprintf(f, "  </g>\n");
    // closing svg tag
    fprintf(f, "</svg>\n");
    fclose(f);

    if (octx->verbose)
    {
        printf("write: %d nodes and %d edges\n", nnodes, nedges);
    }
}

void write_dot(OgLayoutContext *octx, Graph *g)
{
    FILE *f = fopen(octx->path_graph_out, "w");

    int nedges = 0;
    int nnodes = 0;

    if (f == NULL)
    {
        fprintf(stderr, "Cannot write graph into file: %s!\n", octx->path_graph_out);
    }

    fprintf(f, "digraph G {\n");
    // dump out nodes
    int i, j;
    Node dummy;
    for (i = 0; i < octx->graph->numNodes; i++)
    {
        OgNode *node = octx->graph->nodes + i;

        dummy.id = node->nodeID;
        Node *c = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
        assert(c != NULL);

        nnodes++;
        fprintf(f, "%d", node->nodeID);
        fprintf(f, " [");
        fflush(f);
        for (j = 0; j < node->numAttributes; j++)
        {
            // ignore previous position
            if (octx->graph->numAttr)
            {
                int keyIdx = atoi(node->attributes[j].key + 1);
                assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);

                if ((strcmp(octx->graph->attrLookup[keyIdx].name, "pos") == 0))
                    continue;

                fprintf(f, "%s=\"%s\", ", octx->graph->attrLookup[keyIdx].name, node->attributes[j].value);
            }
            else
            {
                if (strcmp(node->attributes[j].key, "pos") == 0)
                    continue;
                fprintf(f, "%s=\"%s\", ", node->attributes[j].key, node->attributes[j].value);
            }
            fflush(f);
        }
        fprintf(f, "pos=\"%f,%f\"", c->x, c->y);
        fflush(f);
        fprintf(f, "]\n");
        fflush(f);
    }

    if (octx->cleanReverseEdges)
    {
        for (i = 0; i < octx->graph->numEdges; i++)
        {
            OgEdge *e = octx->graph->edges + i;
            if (e->sourceId > e->targetId)
            {
                int tmp = e->sourceId;
                e->sourceId = e->targetId;
                e->targetId = tmp;
            }
        }
        qsort(octx->graph->edges, octx->graph->numEdges, sizeof(OgEdge), cmpOgEdges);
    }

    for (i = 0; i < octx->graph->numEdges; i++)
    {
        OgEdge *edge = octx->graph->edges + i;

        int use = -1;
        if (octx->cleanReverseEdges)
        {
            if (i + 1 < octx->graph->numEdges && edge->sourceId == octx->graph->edges[i + 1].sourceId && edge->targetId == octx->graph->edges[i + 1].targetId)
            {
                for (j = 0; j < edge->numAttributes; j++)
                {
                    if (octx->graph->numAttr) // i.e. attributes must be translated via idNUM into attribute name
                    {
                        int keyIdx = atoi(octx->graph->edges[i].attributes[j].key + 1);
                        assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);

                        if ((strcmp(octx->graph->attrLookup[keyIdx].name, "path") == 0) && (atoi(edge->attributes[j].value) >= 0))
                        {
                            use = i;
                            break;
                        }
                    }
                    else
                    {
                        if (strcmp(edge->attributes[j].key, "path") == 0)
                        {
                            if (atoi(edge->attributes[j].value) >= 0)
                            {
                                use = i;
                                break;
                            }
                        }
                    }
                }
                if (use < 0)
                {
                    if (octx->graph->numAttr) // i.e. attributes must be translated via idNUM into attribute name
                    {
                        for (j = 0; j < octx->graph->edges[i + 1].numAttributes; j++)
                        {
                            int keyIdx = atoi(octx->graph->edges[i + 1].attributes[j].key + 1);
                            assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);

                            if ((strcmp(octx->graph->attrLookup[keyIdx].name, "path") == 0) && (atoi(octx->graph->edges[i + 1].attributes[j].value) >= 0))
                            {
                                use = i + 1;
                                break;
                            }
                        }
                    }
                    else
                    {
                        for (j = 0; j < octx->graph->edges[i + 1].numAttributes; j++)
                        {
                            if (strcmp(octx->graph->edges[i + 1].attributes[j].key, "path") == 0)
                            {
                                if (atoi(octx->graph->edges[i + 1].attributes[j].value) >= 0)
                                {
                                    use = i + 1;
                                    break;
                                }
                            }
                        }
                    }

                }

                if (use < 0 || use > i) // take reverse edge
                {
                    edge = octx->graph->edges + i + 1;
                }
                i += 1;
            }
        }

        nedges++;
        fprintf(f, "%d->%d", edge->sourceId, edge->targetId);
        fprintf(f, " [");
        for (j = 0; j < edge->numAttributes; j++)
        {
            if (octx->graph->numAttr)
            {
                int keyIdx = atoi(edge->attributes[j].key + 1);
                assert(keyIdx >= 0 && keyIdx < octx->graph->numAttr);
                fprintf(f, "%s=\"%s\"", octx->graph->attrLookup[keyIdx].name, edge->attributes[j].value);
            }
            else
                fprintf(f, "%s=\"%s\"", edge->attributes[j].key, edge->attributes[j].value);

            if (j + 1 < edge->numAttributes)
                fprintf(f, ", ");
        }
        fprintf(f, "]\n");
    }

    fprintf(f, "}\n");
    fclose(f);

    if (octx->verbose)
    {
        printf("write: %d nodes and %d edges\n", nnodes, nedges);
    }
}

void deleteGraph(Graph *g)
{
    if (g != NULL)
    {
        free(g->edges);
        free(g->nodes);

        if (g->parent != NULL)
            g->parent->child = NULL;

        if (g->child != NULL)
            g->child->parent = NULL;

        free(g);
    }
}

void refineGraph(Graph *g)
{
    if (g->child == NULL)
        return;

    double r = 10;
    int count = 0;
    int refined = 0;

    Graph * child = g->child;
    Node * cNodes = child->nodes;

#if DEBUG
    printf("refine graph\n");
#endif
    int i, j;
    for (i = 0; i < g->curNodes; i++)
    {
        count++;
        Node *n = g->nodes + i;
#if DEBUG
        printf("%d (%.3f, %.3f)", i, n->x, n->y);
#endif
        if (n->flag & NODE_IS_MERGED)
        {
            refined++;
            for (j = 0; j < 2; j++)
            {
                double t = ((double) rand() / (double) (RAND_MAX));

                cNodes[n->childIds[j]].x = (float) (n->x + r * cos(t));
                cNodes[n->childIds[j]].y = (float) (n->y + r * sin(t));
#if DEBUG
                printf(" p: %d (%.3f, %.3f))", n->childIds[j], cNodes[n->childIds[j]].x, cNodes[n->childIds[j]].y);
#endif
            }
        }
        else
        {
            cNodes[n->childIds[0]].x = n->x;
            cNodes[n->childIds[0]].y = n->y;
#if DEBUG
            printf(" p: %d (%.3fq, %.3f))", n->childIds[0], cNodes[n->childIds[0]].x, cNodes[n->childIds[0]].y);
#endif
        }
#if DEBUG
        printf("\n");
#endif
    }
#if DEBUG
    printf("refined Graph:\n");
    printf("nodes: \n");
    for (i = 0; i < child->curNodes; i++)
    {
        Node *n = child->nodes + i;
        printf("node %d: id: %d, idx: %d, firstEdge: %d %d  pos: %f, %f\n", i, n->id, n->idx, child->edges[n->firstEdgeIdx].source, child->edges[n->firstEdgeIdx].target, n->x, n->y);
    }
    printf("edges: \n");
    for (i = 0; i < child->curEdges; i++)
    {
        Edge *e = child->edges + i;
        printf("edge %d: %d - %d\n", i, e->source, e->target);

    }
#endif
}

void updateEdges(Graph *g)
{
#if DEBUG
    printf("update edges:\n");
#endif
    // sort nodes and set idx appropriately
    int i;
    for (i = 0; i < g->curNodes; i++)
    {
#if DEBUG
        printf("node %d: id: %d idx: %d\n", i, g->nodes[i].id, g->nodes[i].idx);
#endif
    }

    qsort(g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
    for (i = 0; i < g->maxNodes; i++)
        g->nodes[i].idx = i;

#if DEBUG
    for (i = 0; i < g->curNodes; i++)
    {
        printf("node %d: id: %d idx: %d\n", i, g->nodes[i].id, g->nodes[i].idx);
    }
#endif

    Graph *child = g->child;
    assert(child != NULL);

    Node dummy;
    for (i = 0; i < child->curEdges; i++)
    {
        Edge *e = child->edges + i;

        if ((e->flag & EDGE_VISITED) || (e->flag & EDGE_IGNORE))
            continue;

#if DEBUG
        printf("check edge: %d - %d %d - %d, pID: %d - %d\n", e->source, e->target, child->nodes[e->source].id, child->nodes[e->target].id, child->nodes[e->source].parentId, child->nodes[e->target].parentId);
#endif

        Node *a = child->nodes + e->source;
        Node *b = child->nodes + e->target;

        if (a->parentId == b->parentId && (a->flag & NODE_HAS_MERGED_PARENT))
        {
#if DEBUG
            printf("update edges: mark %d - %d as visited\n", a->id, b->id);
#endif
            e->flag |= EDGE_VISITED;
        }

        else if ((a->flag & NODE_HAS_PARENT) && (b->flag & NODE_HAS_PARENT) && a->parentId != b->parentId)
        {
#if DEBUG
            printf("update edges: add adge %d - %d ---> %d - %d\n", a->id, b->id, a->parentId, b->parentId);
#endif
            e->flag |= EDGE_VISITED;

            dummy.id = a->parentId;
            Node *c = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
            assert(c != NULL);

            dummy.id = b->parentId;
            Node *d = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
            assert(d != NULL);

            g->edges[g->curEdges].source = c->idx;
            g->edges[g->curEdges].target = d->idx;

#if DEBUG
            printf("update edges: add adge idx: %d - %d\n", c->idx, d->idx);
#endif
            g->curEdges++;
        }
    }

    sortGraphEdges(g);
}

Graph* coarsenGraph(Graph *g, int verboseLevel)
{
    Graph *cgraph = (Graph*) malloc(sizeof(Graph));
    bzero(cgraph, sizeof(Graph));

    cgraph->level = g->level + 1;
    cgraph->child = g;
    g->parent = cgraph;

    cgraph->maxNodes = g->curNodes;
    cgraph->nodes = (Node*) malloc(sizeof(Node) * cgraph->maxNodes);
    assert(cgraph->nodes != NULL);
    bzero(cgraph->nodes, sizeof(Node) * cgraph->maxNodes);

    cgraph->maxEdges = g->curEdges;
    cgraph->edges = (Edge*) malloc(sizeof(Edge) * cgraph->maxEdges);
    assert(cgraph->edges != NULL);
    bzero(cgraph->edges, sizeof(Edge) * cgraph->maxEdges);

    int i;
    int lastId = g->nodes[g->curNodes - 1].id;

    // 1st: add all nodes the could be merge

    for (i = 0; i < g->curEdges; i++)
    {
        Edge *e = g->edges + i;

        if (e->flag & EDGE_VISITED)
            continue;

        Node *a = g->nodes + e->source;
        Node *b = g->nodes + e->target;

        if (a->parentId == b->parentId && !(a->flag & NODE_HAS_MERGED_PARENT))
        {
            Node *parent = cgraph->nodes + cgraph->curNodes;
            parent->id = ++lastId;
            parent->x = (a->x + b->x) / 2;
            parent->y = (a->y + b->y) / 2;
            parent->pathID = -1;

#if DEBUG
            printf("COARSEN EDGE: %d - %d --> new node %d\n", a->id, b->id, parent->id);
#endif

            parent->childIds[0] = a->idx;
            parent->childIds[1] = b->idx;
            parent->flag |= (NODE_HAS_CHILD | NODE_IS_MERGED);

            // add nodes and edges
            a->parentId = parent->id;
            b->parentId = parent->id;
            //        printf("a->child: %d, b->child: %d\n", a->childIdx, b->childIdx);
            a->flag |= (NODE_HAS_MERGED_PARENT | NODE_HAS_PARENT);
            b->flag |= (NODE_HAS_MERGED_PARENT | NODE_HAS_PARENT);
            e->flag |= EDGE_VISITED;

            cgraph->curNodes++;
        }
#if DEBUG
        else
        {
            printf("NO COARSENING between node %d - %d level %d, %d parents %d, %d\n", a->id, b->id,
                    a->flag & a->flag & NODE_IS_MERGED ? 1 : 0, b->flag & NODE_IS_MERGED ? 1 : 0, a->parentId, b->parentId);
        }
#endif
    }

    // 2nd add remaining nodes
    for (i = 0; i < g->curNodes; i++)
    {
        Node *c = g->nodes + i;

        if (!(c->flag & NODE_HAS_PARENT))
        {
#if DEBUG
            printf("add single node %d to parent graph\n", c->id);
#endif
            Node *parent = cgraph->nodes + cgraph->curNodes;
            parent->id = c->id;
            parent->x = c->x;
            parent->y = c->y;

            parent->childIds[0] = c->idx;
            parent->flag |= (NODE_HAS_CHILD);

            // add nodes and edges
            c->parentId = parent->id;
            //        printf("a->child: %d, b->child: %d\n", a->childIdx, b->childIdx);
            c->flag |= (NODE_HAS_PARENT);

            cgraph->curNodes++;
        }
    }

    updateEdges(cgraph);

    //
    if (verboseLevel > 1)
    {
        printf("reduced number of nodes: from %d to %d\n", g->curNodes, cgraph->curNodes);
        printf("reduced number of edges: from %d to %d\n", g->curEdges, cgraph->curEdges);
    }

    return cgraph;
}

void sortGraphEdges(Graph *g)
{
    qsort(g->edges, g->curEdges, sizeof(Edge), cmpEdgeBySourceNode);

    // 1st remove duplicate edges
    int i, j;
    i = 0;
    j = 1;
    while (j < g->curEdges)
    {
        assert(i != j);
        if (cmpEdgeBySourceNode(g->edges + i, g->edges + j) == 0)
        {
            if (j + 1 == g->curEdges)
                break;

            g->edges[i + 1] = g->edges[j + 1];
            j += 1;
        }
        else if (i + 1 < j)
        {
            g->edges[i + 1] = g->edges[j];
        }
        i++;
        j++;
    }

    g->curEdges = i + 1;

    // 2nd reset first edge indexes -1
    for (i = 0; i < g->curNodes; i++)
        g->nodes[i].firstEdgeIdx = -1;

    // 2nd updates first edge for each node
    g->nodes[g->edges[0].source].firstEdgeIdx = 0;
    int prevEdgeSourceIdx = g->edges->source;

    for (i = 1; i < g->curEdges; i++)
    {
        Edge *e = g->edges + i;
        if (prevEdgeSourceIdx != e->source)
        {
            g->nodes[e->source].firstEdgeIdx = i;
            prevEdgeSourceIdx = g->edges[i].source;
        }
    }
}

static void usage(FILE* fout, const char* app)
{
    fprintf(fout, "usage: %s [-vRS] [-f [dot|graphml]] [-F [dot|svg]] [-qC n] [-tlcsdg f] input.graph output.graph\n\n", app);

    fprintf( fout, "Computes a layout for the (usually toured) input graph.\n\n" );

    fprintf(fout, "options: -v  verbose\n");
    fprintf(fout, "         -S  skip layout step (e.g. graph format conversion, or apply other color scheme)\n");
    fprintf(fout, "         -f format  graph input format\n");
    fprintf(fout, "         -F format  graph output format\n");
    fprintf(fout, "         -C n  apply color scheme\n");
    fprintf(fout, "            0  input colors (default)\n");
    fprintf(fout, "            1  edges and nodes: black, path edges red, contig start/end: green\n");
    fprintf(fout, "            2  edges and nodes: black\n");
    fprintf(fout, "         -R  remove reverse edges from output, i.e. create a directed graph with arbitrary direction\n");

    fprintf(fout, "\nlayout:\n");
    fprintf(fout, "         -q n  QuadTree level (default: 10)\n");
    fprintf(fout, "               Maximum value to be used in the QuadTree representation. Greater values mean more accuracy\n");
    fprintf(fout, "         -t f  theta, Barnes Hut opening criteria (default: 1.2)\n");
    fprintf(fout, "               Smaller values mean more accuracy\n");
    fprintf(fout, "         -l f  Minimum level size (default: 3)\n");
    fprintf(fout, "               Minimum amount of nodes every level must have. Bigger values mean less levels\n");
    fprintf(fout, "         -c f  Coarsening rate (default: 0.75)\n");
    fprintf(fout, "               Minimum relative size (number of nodes) between two levels. Smaller values mean less levels\n");
    fprintf(fout, "         -s f  Step ratio (default: 0.97)\n");
    fprintf(fout, "               The ratio used to update the step size across iterations\n");
    fprintf(fout, "         -d f  Optimal distance (default: 100)\n");
    fprintf(fout, "               The natural length of the springs (edge length). Bigger values mean nodes will be further apart\n");
    fprintf(fout, "         -g f  convergence threshold (default: 0.0001)\n");
}


static int parseOptions(OgLayoutContext *octx, int argc, char *argv[])
{
    char* app = argv[ 0 ];
    bzero( octx, sizeof( OgLayoutContext ) );

    // set default values
    octx->quadTreeLevel = DEFAULT_QUADTREE_LEVEL;
    octx->coarseningRate = DEFAULT_COARSEN_RATE;
    octx->minLevelSize = DEFAULT_MIN_LEVEL;
    octx->optimalDistance = DEFAULT_OPTIMAL_DISTANCE;
    octx->stepRatio = DEFAULT_STEP_RATIO;
    octx->barnesHutTheta = DEFAULT_BARNESHUT_THETA;
    octx->convergenceThreshold = DEFAULT_CONVERGENCE_THRESHOLD;

    octx->giformat = FORMAT_UNKNOWN;
    octx->goformat = FORMAT_UNKNOWN;

    octx->skipLayout = 0;
    octx->colorScheme = 0;

    char* giformat = "UNK";
    char* goformat = "UNK";
    // process arguments

    opterr = 0;

    int c;
    while ((c = getopt(argc, argv, "vSRq:t:l:c:s:d:f:F:C:g:")) != -1)
    {
        switch (c)
        {
            case 'v':
                octx->verbose++;
                break;

            case 'S':
                octx->skipLayout = 1;
                break;

            case 'R':
                octx->cleanReverseEdges = 1;
                break;

            case 'q':
                octx->quadTreeLevel = atoi(optarg);
                break;

            case 't':
                octx->barnesHutTheta = atof(optarg);
                break;

            case 'l':
                octx->minLevelSize = atoi(optarg);
                break;

            case 'c':
                octx->coarseningRate = atof(optarg);
                break;

            case 'g':
                octx->convergenceThreshold = atof(optarg);
                break;

            case 's':
                octx->stepRatio = atof(optarg);
                break;

            case 'd':
                octx->optimalDistance = atoi(optarg);
                break;

            case 'C':
                octx->colorScheme = atoi(optarg);
                break;

            case 'f':
                giformat = optarg;
                break;

            case 'F':
                goformat = optarg;
                break;

            default:
                fprintf(stderr, "Unknown option: %s\n", argv[optind - 1]);
                usage(stdout, app);
                exit(1);
        }
    }

    if (argc - optind < 2)
    {
        usage(stdout, app);
        exit(1);
    }

    octx->path_graph_in = argv[optind++];
    octx->path_graph_out = argv[optind++];

    if (octx->colorScheme < 0 || octx->colorScheme > 2)
    {
        fprintf(stderr, "Unsupported color scheme %d. Available schemes: 0-2!\n", octx->colorScheme);
        usage(stdout, app);
        exit(1);
    }

    if (octx->quadTreeLevel < 1 && octx->quadTreeLevel > 100)
    {
        fprintf(stderr, "Unsupported quadtree level: %d! Allowed range: [1, 100]\n", octx->quadTreeLevel);
        return 1;
    }

    if (octx->stepRatio <= 0 && octx->quadTreeLevel >= 1.0)
    {
        fprintf(stderr, "Unsupported step ratio: %f! Allowed range: ] 0, 1 [\n", octx->stepRatio);
        return 1;
    }

    if (octx->optimalDistance < 0)
    {
        fprintf(stderr, "Unsupported optimal distance: %f! Should be a positive value\n", octx->optimalDistance);
        return 1;
    }

    if (octx->barnesHutTheta <= 0)
    {
        fprintf(stderr, "Unsupported barnes hut theta : %f! Should be a positive value greater 0\n", octx->barnesHutTheta);
        return 1;
    }

    if (octx->coarseningRate <= 0 && octx->coarseningRate >= 1.0)
    {
        fprintf(stderr, "Unsupported coarsening rate: %f! Allowed range: ] 0, 1 [\n", octx->coarseningRate);
        return 1;
    }

    if (strcmp(giformat, "dot") == 0)
    {
        octx->giformat = FORMAT_DOT;
    }
    else if (strcmp(giformat, "graphml") == 0)
    {
        octx->giformat = FORMAT_GRAPHML;
    }
    else if (strcmp(giformat, "UNK") == 0)
    {
        // try to derive format from file extension
        size_t len = strlen(octx->path_graph_in);

        if (len > 3)
        {
            if (strcasecmp(octx->path_graph_in + len - 3, "dot") == 0)
                octx->giformat = FORMAT_DOT;
        }

        if (len > 7)
        {
            if (strcasecmp(octx->path_graph_in + len - 7, "graphml") == 0)
                octx->giformat = FORMAT_GRAPHML;
        }

        if (octx->giformat == FORMAT_UNKNOWN)
        {
            fprintf(stderr, "error: cannot determine input graph format. Use option -f graphml|dot\n");
            usage(stdout, app);
            exit(1);
        }
    }
    else
    {
        fprintf(stderr, "error: unknown input graph format %s\n", giformat);
        usage(stdout, app);
        exit(1);
    }

    if (strcmp(goformat, "dot") == 0)
    {
        octx->goformat = FORMAT_DOT;
    }
    else if (strcmp(goformat, "svg") == 0)
    {
        octx->goformat = FORMAT_SVG;
    }
    else if (strcmp(giformat, "UNK") == 0)
    {
        // try to derive format from file extension
        size_t len = strlen(octx->path_graph_out);

        if (len > 3)
        {
            if (strcasecmp(octx->path_graph_out + len - 3, "dot") == 0)
                octx->goformat = FORMAT_DOT;
            else if (strcasecmp(octx->path_graph_out + len - 3, "svg") == 0)
                octx->goformat = FORMAT_SVG;

        }

        if (octx->goformat == FORMAT_UNKNOWN)
        {
            fprintf(stderr, "error: cannot determine output graph format. Use option -f dot|svg\n");
            usage(stdout, app);
            exit(1);
        }
    }
    else
    {
        fprintf(stderr, "error: unknown graph format %s\n", goformat);
        usage(stdout, app);
        exit(1);
    }

    return 0;
}

static Graph * createInitialGraph(OgLayoutContext *octx)
{
    Graph *g = (Graph*) malloc(sizeof(Graph));

    if (g == NULL)
        return NULL;

    bzero(g, sizeof(Graph));

    g->maxNodes = octx->graph->numNodes;
    g->maxEdges = octx->graph->numEdges;
    g->nodes = (Node*) calloc(g->maxNodes, sizeof(Node));
    g->edges = (Edge*) calloc(g->maxEdges, sizeof(Edge));

    int i, j;
    char *pchr;

    for (i = 0; i < g->maxNodes; i++)
    {
        OgNode *node = octx->graph->nodes + i;
        g->nodes[i].id = node->nodeID;
        g->nodes[i].pathID = -1;

        // check if coordinates are available
        for (j = 0; j < node->numAttributes; j++)
        {
            if (strcmp(node->attributes[j].key, "pos") == 0)
            {
                // get x and y coordinates

                pchr = strchr(node->attributes[j].value, ',');
                if (pchr == NULL)
                    break;

                if (strlen(pchr + 1) < 1)
                    break;

                *pchr = '\0';

                g->nodes[i].x = atof(node->attributes[j].value);
                *pchr = ',';
                g->nodes[i].y = atof(pchr + 1);

                g->nodes[i].flag |= NODE_HAS_COORDINATES;
            }
            else if (strcmp(node->attributes[j].key, "path") == 0)
            {
                int path = atoi(node->attributes[j].value);
                if (path >= 0)
                {
                    g->nodes[i].pathID = path;
                    g->nodes[i].flag |= NODE_HAS_PATH_ID;
                }
            }
        }

        g->curNodes++;
    }

    // sort nodes and set idx appropriately
    qsort(g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
    for (i = 0; i < g->maxNodes; i++)
        g->nodes[i].idx = i;

    Node dummy;
    for (i = 0; i < g->maxEdges; i++)
    {
        dummy.id = octx->graph->edges[i].sourceId;
        Node *a = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
        assert(a != NULL);

        dummy.id = octx->graph->edges[i].targetId;
        Node *b = bsearch(&dummy, g->nodes, g->curNodes, sizeof(Node), cmpNodesById);
        assert(b != NULL);

        g->edges[i].source = a->idx;
        g->edges[i].target = b->idx;

        g->curEdges++;
    }

    // sort edges according node ids and set first edge for each node
    sortGraphEdges(g);

    // if there are two edges between nodes A and B i.e. A->B and B->A
    // then ignore one of them
    int ignEdge = 0;
    for (i = 0; i < g->curEdges; i++)
    {
        Edge *e = g->edges + i;

        if (e->flag & EDGE_IGNORE)
            continue;

        Node *target = g->nodes + e->target;
        if (target->firstEdgeIdx >= 0)
        {
            Edge *re = g->edges + target->firstEdgeIdx;

            assert(re != NULL);
            while (re < g->edges + g->curEdges && re->source == e->target)
            {
                if (re->target == e->source)
                {
                    re->flag |= EDGE_IGNORE;
                    ignEdge++;
                }
                re++;
            }
        }
    }
    if (octx->verbose > 1)
    {
        printf("ignore %d reverse edges\n", ignEdge);
    }

    return g;
}

static MultiLevelLayout* initMultiLevelLayout(OgLayoutContext *octx, Graph *g)
{
    MultiLevelLayout *mlayout = (MultiLevelLayout*) malloc(sizeof(MultiLevelLayout));

    if (mlayout == NULL)
        return NULL;

    mlayout->minSize = octx->minLevelSize;
    mlayout->minCoarseningRate = octx->coarseningRate;
    mlayout->level = 0;
    mlayout->maxGraphs = 10;
    mlayout->graphs = (Graph**) malloc(sizeof(Graph*) * mlayout->maxGraphs);

    mlayout->graphs[0] = g;

    mlayout->bestCoordinateSet = (float*) malloc(sizeof(float) * g->curNodes * 2);
    if (mlayout->bestCoordinateSet == NULL)
        return NULL;

    // start coarsen graph
    while (1)
    {
        if (mlayout->level + 1 >= mlayout->maxGraphs)
        {
            mlayout->maxGraphs = mlayout->maxGraphs * 1.2 + 10;
            mlayout->graphs = (Graph**) realloc(mlayout->graphs, sizeof(Graph*) * mlayout->maxGraphs);

            assert(mlayout->graphs != NULL);
        }

        int nnodes = mlayout->graphs[mlayout->level]->curNodes;

        if (octx->verbose > 1)
            printf("coarsen graph level(%d)\n", mlayout->level);

        Graph *tmpG = coarsenGraph(mlayout->graphs[mlayout->level], octx->verbose);

        mlayout->level++;
        mlayout->graphs[mlayout->level] = tmpG;

        int nchildnodes = tmpG->curNodes;
#if DEBUG
        printf("nnodes: %d, nchildnodes: %d c-rate: %f\n", nnodes, nchildnodes, mlayout->minCoarseningRate);
#endif
        if (nchildnodes < mlayout->minSize || nchildnodes > nnodes * mlayout->minCoarseningRate)
            break;
    }

#if DEBUG
    printf("stopped coarsening at level: %d, nnodes: %d\n", mlayout->level, mlayout->graphs[mlayout->level]->curNodes - 1);
#endif

    // initially create random coordinates
    {
        Graph *startG = mlayout->graphs[mlayout->level];
        double size = 1000.0;
        srand((unsigned int) time(NULL));

        int i;
        for (i = 0; i < startG->curNodes; i++)
        {
            Node *a = startG->nodes + i;
            if (!(a->flag & NODE_HAS_COORDINATES))
            {
                a->x = (float) (-size / 2 + size * ((double) rand() / (double) (RAND_MAX)));
                a->y = (float) (-size / 2 + size * ((double) rand() / (double) (RAND_MAX)));
#if DEBUG
                printf("random coordinates: n%d: %.3f, %.3f\n", a->id, a->x, a->y);
#endif
            }
        }
    }
    return mlayout;
}

static void resetYifanHuLayout(OgLayoutContext *octx, YifanHuLayout *yfhLayout)
{
    yfhLayout->stepRatio = octx->stepRatio;
    yfhLayout->relativeStrength = DEFAULT_RELATIVE_STRENGTH;
    yfhLayout->optimalDistance = octx->optimalDistance;
    yfhLayout->initialStep = octx->optimalDistance / 5;
    yfhLayout->quadTreeMaxLevel = octx->quadTreeLevel;
    yfhLayout->barnesHutTheta = octx->barnesHutTheta;
    yfhLayout->adaptiveCooling = DEFAULT_ADAPTIVE_COOLING;
    yfhLayout->convergenceThreshold = octx->convergenceThreshold;
    //
    // init algorithm
    // todo check if forcevector of nodes is really initiallized with (0,0)
    yfhLayout->progress = 0;
    yfhLayout->converged = 0;
    yfhLayout->step = yfhLayout->initialStep;
}

static void doYiFanHuLayout(OgLayoutContext *octx, MultiLevelLayout *mlayout)
{
    YifanHuLayout yfhLayout;

    QuadTree *tree = createQuadTree(octx->quadTreeLevel);

    int i;

    int curLevel = mlayout->level;
    while (curLevel >= 0)
    {
        Graph *tmpGraph = mlayout->graphs[curLevel];

        resetYifanHuLayout(octx, &yfhLayout);

#ifdef  DEBUG
        int counter = 0;
#endif

        float bestEnergy = FLT_MAX;
        while (!yfhLayout.converged)
        {
#if DEBUG
            printf("goAlgo: %d\n", counter++);
#endif
            buildQuadTree(tree, tmpGraph);
#if DEBUG
            printf("QUADDTREE: cMass (%.3f, %.3f) pos: (%.3f, %.3f)"
                    " isLeaf %d, eps: %.3f size: %.3f, maxLevel: %d\n", tree->centerMassX, tree->centerMassY, tree->posX, tree->posY, tree->isLeaf, tree->eps, tree->size, tree->maxLevel);
#endif
            double electricEnergy = 0;
            double springEnergy = 0;

            // update node forces
            ElectricalForce ef;
            ef.theta = yfhLayout.barnesHutTheta;
            ef.optimalDistance = yfhLayout.optimalDistance;
            ef.relativeStrength = yfhLayout.relativeStrength;
            float distance;

            for (i = 0; i < tmpGraph->curNodes; i++)
            {
                Node *n = tmpGraph->nodes + i;
                ForceVector *fv = calculateForce(&ef, n, tree);
                if (fv)
                {
#if DEBUG
                    printf("node %d forcevector %.3f, %.3f, barnesForce: %.3f, %.3f\n", n->id, n->f.x, n->f.y, fv->x, fv->y);
#endif
                    addForceVectorToForceVector(&(n->f), fv);
#if DEBUG
                    printf("apply nodeforce fv: %.3f, %.3f electEnergy: %.3f\n", n->f.x, n->f.y, getForceVectorEnergy(fv));
#endif
                    electricEnergy += getForceVectorEnergy(fv);
                    free(fv);
                }
            }

            // update edge forces
            SpringForce sf;
            sf.optimalDistance = yfhLayout.optimalDistance;
            for (i = 0; i < tmpGraph->curEdges; i++)
            {
                Edge *e = tmpGraph->edges + i;

                if (e->source == e->target)
                    continue;

                Node * n1 = tmpGraph->nodes + e->source;
                Node * n2 = tmpGraph->nodes + e->target;

                distance = getForceVectorDistanceToNode(n1, n2);
                ForceVector* f = calculateSpringForce(&sf, n1, n2, distance);
#if DEBUG
                printf("edge: %d - %d dist: %.3f, fv (%.3f, %.3f)(%.3f, %.3f) force: %.3f, %.3f\n", n1->id, n2->id, distance, n1->f.x, n1->f.y, n2->f.x, n2->f.y, f->x, f->y);
#endif
                addForceVectorToForceVector(&(n1->f), f);
                subtractForceVectorFromForceVector(&(n2->f), f);
#if DEBUG
                printf("apply edge force: n%d: (%.3f, %.3f), n%d: (%.3f, %.3f)\n", n1->id, n1->f.x, n1->f.y, n2->id, n2->f.x, n2->f.y);
#endif
                free(f);
            }

            // calculate energy and max force
            yfhLayout.energy0 = yfhLayout.energy;
            yfhLayout.energy = 0;
            double maxForce = 1;
            for (i = 0; i < tmpGraph->curNodes; i++)
            {
                Node *n = tmpGraph->nodes + i;
                float norm = getForceVectorNorm(&(n->f));
#if DEBUG
                printf("node: %d energy: %.3f\n", n->id, norm);
#endif
                yfhLayout.energy += norm;
                if (maxForce < norm)
                    maxForce = norm;
            }

#if DEBUG
            printf("maxForce: %.3f energy0: %.3f energy: %.3f\n", maxForce, yfhLayout.energy0, yfhLayout.energy);
            printf("node displacement:\n");
#endif
            // displacement on nodes
            for (i = 0; i < tmpGraph->curNodes; i++)
            {
                Node *n = tmpGraph->nodes + i;

#if DEBUG
                printf("n%d c:(%.3f %.3f) f(%.5f, %.5f) step(%.10f)->", n->id, n->x, n->y, n->f.x, n->f.y, yfhLayout.step);
#endif
                n->f.x *= (1.0 / maxForce);
                n->f.y *= (1.0 / maxForce);
                n->x = (n->x) + (n->f.x * yfhLayout.step);
                n->y = (n->y) + (n->f.y * yfhLayout.step);
#if DEBUG
                printf(" nc(%.3f %.3f)\n", n->x, n->y);
#endif
            }

            // postAlgo

            // 1. update step
            if (yfhLayout.adaptiveCooling)
            {
                if (yfhLayout.energy < yfhLayout.energy0)
                {
                    yfhLayout.progress++;
                    if (yfhLayout.progress >= 5)
                    {
                        yfhLayout.progress = 0;
                        yfhLayout.step /= yfhLayout.stepRatio;
                    }
                    else
                    {
                        yfhLayout.progress = 0;
                        yfhLayout.step *= yfhLayout.stepRatio;
                    }
                }
            }
            else
            {
                yfhLayout.step *= yfhLayout.stepRatio;
            }
            // 2. is converged ??
            if (fabs(yfhLayout.energy - yfhLayout.energy0) / yfhLayout.energy < yfhLayout.convergenceThreshold)
                yfhLayout.converged = 1;

            if (bestEnergy > yfhLayout.energy)
            {
                bestEnergy = yfhLayout.energy;
                int j;
                for (i = 0, j = 0; i < tmpGraph->curNodes; i++, j += 2)
                {
                    Node *n = tmpGraph->nodes + i;
                    mlayout->bestCoordinateSet[j] = n->x;
                    mlayout->bestCoordinateSet[j + 1] = n->y;
                }
            }

            springEnergy = yfhLayout.energy - electricEnergy;

            if (octx->verbose > 1 && yfhLayout.converged)
            {
                printf("curLevel: %d of %d\n", mlayout->level-curLevel, mlayout->level);
                printf("convergence: %.8f < %.8f\n", fabs(yfhLayout.energy - yfhLayout.energy0) / yfhLayout.energy, yfhLayout.convergenceThreshold);
                printf("electric: %f spring: %f\n", electricEnergy, springEnergy);
                printf("energy0: %.8f energy: %.8f\n", yfhLayout.energy0, yfhLayout.energy);
                printf("bestEnergy found: %f diff: %f\n", bestEnergy, yfhLayout.energy - bestEnergy);
            }

            if (yfhLayout.converged && bestEnergy < yfhLayout.energy)
            {
                int j;
                for (i = 0, j = 0; i < tmpGraph->curNodes; i++, j += 2)
                {
                    Node *n = tmpGraph->nodes + i;
                    n->x = mlayout->bestCoordinateSet[j];
                    n->y = mlayout->bestCoordinateSet[j + 1];
                }
            }

//            deleteQuadTree(tree, yfhLayout.quadTreeMaxLevel);
        }

        refineGraph(tmpGraph);
#if DEBUG
        char blub[100];
        sprintf(blub, "test_%d.dot", curLevel);
        write_dot(tmpGraph, blub);
#endif
        curLevel--;
    }
    free(tree);
}

int main(int argc, char* argv[])
{
    OgLayoutContext octx;

    if (parseOptions(&octx, argc, argv))
        return 1;

    // read graph
    if (octx.giformat == FORMAT_GRAPHML)
    {
        if (!read_graphml(&octx))
        {
            fprintf(stderr, "error: failed to read %s\n", octx.path_graph_in);
            exit(1);
        }
    }
    else
    {
        if (!read_dot(&octx))
        {
            fprintf(stderr, "error: failed to read %s\n", octx.path_graph_in);
            exit(1);
        }
    }
    if (octx.verbose)
        printf("parsed: %d nodes and %d edges\n", octx.graph->numNodes, octx.graph->numEdges);

    if (octx.graph->numNodes < 3 || octx.graph->numEdges < 2)
    {
        fprintf(stderr, "WARNING: Graph too small for layouting. %d nodes, %d edges.\n", octx.graph->numNodes, octx.graph->numEdges);
        exit(0);
    }

    // init first graph based on parsed OGnodes + OGedges
    Graph *ig = createInitialGraph(&octx);
    if (ig == NULL)
    {
        fprintf(stderr, "Unable to create initial Graph. Stopp!\n");
        exit(1);
    }

    MultiLevelLayout *mlayout = NULL;
    if (octx.skipLayout == 0)
    {
        // create YifanHu Multilayout and coarsen graphs
        mlayout = initMultiLevelLayout(&octx, ig);
        if (mlayout == NULL)
        {
            fprintf(stderr, "Unable to create MultiLevelLayout. Stopp!\n");
            exit(1);
        }

        // do layout
        doYiFanHuLayout(&octx, mlayout);
    }
    // write graph
    if (octx.goformat == FORMAT_DOT)
        write_dot(&octx, ig);
    else
        write_svg(&octx, ig);

    // cleanup

    free(octx.graph->edges);
    free(octx.graph->nodes);
    free(octx.graph->attrLookup);
    free(octx.graph);

    if (octx.skipLayout == 0)
    {
        int l = mlayout->level;
        while (l >= 0)
        {
            deleteGraph(mlayout->graphs[l]);
            l--;
        }

        free(mlayout->graphs);
        free(mlayout);
    }
    else
        deleteGraph(ig);

    return 0;
}

