#pragma once

class Circle;
#include <Eigen/Core>

using Vec2 = Eigen::Vector2f;

class Rect
{
    /**
     * A rectangle can be n-d and is defined by its "top left" vector and its "width" vector, which must be all positive
    */
public:
    Vec2 v1;
    Vec2 v2;

    Rect(Vec2 v1, Vec2 v2);

    bool contains(const Vec2& v) const;

    bool intersects(const Rect& r) const;

    bool intersects(const Circle& c) const;

    Vec2 project(const Vec2& v) const;

    float sqr_dist(const Vec2& v) const;
};