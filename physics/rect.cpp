#include "rect.hpp"
#include "circle.hpp"

#include <algorithm>
#include <Eigen/Core>
#include <utility>

using Vec2 = Eigen::Vector2f;

Rect::Rect(Vec2 v1, Vec2 v2)
    : v1(std::move(v1)), v2(std::move(v2))
{
}

bool Rect::contains(const Vec2& v) const
{
    return v.x() >= v1.x() && v.x() <= v1.x() + v2.x() && v.y() >= v1.y() && v.y() <= v1.y() + v2.y();
}

bool Rect::intersects(const Rect& r) const
{
    return v1.x() <= r.v1.x() + r.v2.x() && v1.x() + v2.x() >= r.v1.x() && v1.y() <= r.v1.y() + r.v2.y() && v1.y() + v2.y() >= r.v1.y();
}

bool Rect::intersects(const Circle& c) const
{
    return c.center.x() + c.radius >= v1.x() && c.center.x() - c.radius <= v1.x() + v2.x() && c.center.y() + c.radius >= v1.y() && c.center.y() - c.radius <= v1.y() + v2.y();
}

Vec2 Rect::project(const Vec2& v) const
{
    return Vec2(std::clamp(v.x(), v1.x(), v1.x() + v2.x()), std::clamp(v.y(), v1.y(), v1.y() + v2.y()));
}

float Rect::sqr_dist(const Vec2& v) const
{
    Vec2 p = project(v);
    return (v - p).squaredNorm();
}