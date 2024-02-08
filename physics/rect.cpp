#include "rect.hpp"
#include "circle.hpp"

#include <algorithm>
#include <Eigen/Core>
#include <utility>

using Vec2 = Eigen::Vector2f;

Rect::Rect(Vec2&& top_left, Vec2&& size)
    : top_left(std::move(top_left)), size(std::move(size))
{
}

Rect::Rect(const Vec2& top_left, const Vec2& size)
        : top_left(top_left), size(size)
{
}

bool Rect::contains(const Vec2& v) const
{
    return v.x() >= top_left.x() && v.x() <= top_left.x() + size.x() && v.y() >= top_left.y() && v.y() <= top_left.y() + size.y();
}

bool Rect::intersects(const Rect& r) const
{
    return top_left.x() <= r.top_left.x() + r.size.x() && top_left.x() + size.x() >= r.top_left.x() && top_left.y() <= r.top_left.y() + r.size.y() && top_left.y() + size.y() >= r.top_left.y();
}

bool Rect::intersects(const Circle& c) const
{
    return c.center.x() + c.radius >= top_left.x() && c.center.x() - c.radius <= top_left.x() + size.x() && c.center.y() + c.radius >= top_left.y() && c.center.y() - c.radius <= top_left.y() + size.y();
}

Vec2 Rect::project(const Vec2& v) const
{
    return Vec2(std::clamp(v.x(), top_left.x(), top_left.x() + size.x()), std::clamp(v.y(), top_left.y(), top_left.y() + size.y()));
}

float Rect::sqr_dist(const Vec2& v) const
{
    Vec2 p = project(v);
    return (v - p).squaredNorm();
}