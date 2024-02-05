/**
 * implementation of quadtree class
*/

#include "quadtree.hpp"
#include "rect.hpp"
#include "circle.hpp"
#include <utility>
#include <vector>
#include <optional>
#include <algorithm>
#include <Eigen/Core>

using Vec2 = Eigen::Vector2f;

using namespace Eigen;

QuadTree::QuadTree(Rect rect, int max_capacity, float object_radius)
    : rect(std::move(rect)), max_capacity(max_capacity), object_radius(object_radius)
{
    subdivided = false;
    empty = true;
    n_points = 0;
}

QuadTree::QuadTree(Rect rect, int max_capacity)
    : QuadTree(std::move(rect), max_capacity, 0)
{
}

QuadTree::QuadTree(Rect rect, float object_radius)
    : QuadTree(std::move(rect), 1, object_radius)
{
}

QuadTree::QuadTree(Rect rect)
    : QuadTree(std::move(rect), 1, 0)
{
}

QuadTree::~QuadTree()
{
    // TODO : check that std::vector frees memory correctly
    for (QuadTree *child : children)
    {
        delete child;
    }
}

void QuadTree::subdivide()
{
    subdivided = true;
    children.reserve(4);

    Vec2 v1 = rect.v1;
    Vec2 v2 = rect.v2;
    Vec2 v2_half = v2 / 2;
    
    children.push_back(new QuadTree(Rect(v1, v2_half), max_capacity, object_radius));
    children.push_back(new QuadTree(Rect(v1 + Vec2(v2_half.x(), 0), v2_half), max_capacity, object_radius));
    children.push_back(new QuadTree(Rect(v1 + Vec2(0, v2_half.y()), v2_half), max_capacity, object_radius));
    children.push_back(new QuadTree(Rect(v1 + v2_half, v2_half), max_capacity, object_radius));

    for (const Vec2& v : points)
    {
        insert(v);
    }

    points.clear();
    n_points = 0;
}

bool QuadTree::insert(const Vec2& v)
{
    if (!rect.contains(v))
    {
        return false;
    }
    if (subdivided)
    {
        for (int i = 0; i < 4; i++)
        {
            if (children[i]->insert(v))
            {
                return true;
            }
        }
        throw std::runtime_error("Failed to insert in all children");
    }
    if (n_points < max_capacity)
    {
        if (n_points == 0)
        {
            points.reserve(max_capacity);
        }
        points.push_back(v);
        n_points++;
        empty = false;
        return true;
    }
    subdivide();
    assert(insert(v));
    return true;
}

void QuadTree::queryCircle(const Vec2& v, std::vector<Vec2> &buffer)
{
    if (!rect.intersects(Circle(v, object_radius)))
    {
        return;
    }
    if (subdivided)
    {
        for (QuadTree *child : children)
        {
            child->queryCircle(v, buffer);
        }
        return;
    }

    float sqr_radius = object_radius * object_radius;
    for (Vec2 p : points)
    {
        if ((p - v).squaredNorm() <= sqr_radius)
        {
            buffer.push_back(p);
        }
    }
}

std::optional<Vec2> QuadTree::naiveClosest(const Vec2& v)
{
    std::vector<Vec2> buffer;
    queryCircle(v, buffer);
    if (buffer.empty())
    {
        return std::nullopt;
    }
    Vec2 closest = buffer[0];
    float min_dist = (closest - v).squaredNorm();
    for (const Vec2& p : buffer)
    {
        float dist = (p - v).squaredNorm();
        if ((dist < min_dist && dist > 0) || min_dist == 0)
        {
            closest = p;
            min_dist = dist;
        }
    }
    return closest;
}

float QuadTree::closest(const Vec2& v, std::optional<Vec2> &bestPoint, float bestDist)
{
    if (subdivided)
    {
        std::vector<QuadTree *> ordered_children;
        ordered_children.reserve(4);
        for (QuadTree *child : children)
        {
            ordered_children.push_back(child);
        }
        std::sort(ordered_children.begin(), ordered_children.end(),
            [v](QuadTree *a, QuadTree *b) {
                Vec2  proj_a = a->rect.project(v);
                Vec2  proj_b = b->rect.project(v);
            return (proj_a - v).squaredNorm() < (proj_b - v).squaredNorm();
        });
        for (QuadTree *child : ordered_children)
        {
            Vec2 proj = child->rect.project(v);
            float dist = (proj - v).squaredNorm();
            if (dist < bestDist || bestDist < 0)
            {
                bestDist = child->closest(v, bestPoint, bestDist);
            }
        }
    }
    else
    {
        for (const Vec2& p : points)
        {
            float dist = (p - v).squaredNorm();
            if (dist == 0)
                continue;

            if (dist < bestDist || bestDist < 0)
            {
                bestDist = dist;
                bestPoint = p;
            }
        }
    }

    return bestDist;
}

std::optional<Vec2> QuadTree::closest(const Vec2& v)
{
    std::optional<Vec2> best;
    closest(v, best, -1);
    return best;
}