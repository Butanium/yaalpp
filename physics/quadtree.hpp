/**
 * header file for quadtree class
*/

#pragma once

#include "rect.hpp"
#include <vector>
#include <optional>
#include <Eigen/Core>
#include <omp.h>

using Vec2 = Eigen::Vector2f;

class QuadTree
{
    /**
     * A quadtree is a tree data structure that divides space into 4 quadrants
     * Each node has a rectangle that defines its boundaries
     * Each node has a list of points that are contained in its rectangle
     * If a node has more than a certain number of points, it subdivides into 4 quadrants
     * Each node has a list of its children
    */
private:
    int max_capacity;
    float object_radius;

    Rect rect;
    bool subdivided;
    bool empty;

    int n_points;
    std::vector<Vec2> points;

    std::vector<QuadTree *> children;

    omp_lock_t lock;

    void subdivide();

    // appends all points within collision range of v to the buffer
    void queryCircle(const Vec2& v, std::vector<Vec2> &buffer);

    // actually does the closest point search. The other is just here to initialize the optional and put a -1 in the call to this one.
    float closest(const Vec2& v, std::optional<Vec2> &bestPoint, float bestDist);

public:
    // default capacity is 1 and default radius is 0
    explicit QuadTree(Rect rect);

    QuadTree(Rect rect, float object_radius);

    QuadTree(Rect rect, int max_capacity);

    QuadTree(Rect rect, int max_capacity, float object_radius);

    ~QuadTree();

    // returns true if the point was inserted, false otherwise
    bool insert(const Vec2& v);

    // returns the closest point to v. Uses naive search (query all points within collision radius and return the naive closest)
    // If no points are found, returns
    std::optional<Vec2> naiveClosest(const Vec2& v);

    // returns the closest point to v. Uses a breadth-first search
    std::optional<Vec2> closest(const Vec2& v);

    // Given a set of Yaals, insert them into the quadtree.
    void initialize(const std::vector<Yaal>& yaals);

    void get_all_closest(const std::vector<Yaal>& yaals);
};