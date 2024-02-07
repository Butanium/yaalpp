#include "circle.hpp"
#include "rect.hpp"
#include <Eigen/Dense>
#include <utility>

using Vec2 = Eigen::Vector2f;

Circle::Circle(Vec2 center, float radius)
        : center(std::move(center)), radius(radius) {
}

bool Circle::contains(const Vec2 &v) const {
    return (center - v).squaredNorm() <= radius * radius;
}

bool Circle::intersects(const Circle &c) const {
    return (center - c.center).squaredNorm() <= (radius + c.radius) * (radius + c.radius);
}

//bool Circle::intersects(Rect r) {
//    return r.intersects(*this);
//}

bool Circle::intersects(const Rect &r) const {
    return r.intersects(*this);
    // Rest of the Circle class implementation goes here
}