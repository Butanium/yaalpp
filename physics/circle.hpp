#pragma once

class Rect;
#include <Eigen/Core>

using Vec2 = Eigen::Vector2f;

class Circle
{
    /**
     * A circle is defined by its center and its radius
    */
public:
    Vec2 center;
    float radius;

    Circle(Vec2 center, float radius);

    bool contains(const Vec2& v) const;

    bool intersects(const Circle& c) const;

    bool intersects(const Rect& r) const;
};