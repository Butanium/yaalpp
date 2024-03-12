/**
 * implementation of quadtree class
*/

#include "quadtree.hpp"
#include "../utils/circle.hpp"
#include <utility>
#include <algorithm>


using Vec2 = Eigen::Vector2f;

QuadTree::QuadTree(Rect &&rect, int max_capacity, float object_radius)
        : rect(std::move(rect)), max_capacity(max_capacity), object_radius(object_radius) {
    subdivided = false;
    empty = true;
    n_points = 0;
    omp_init_lock(&lock);
}

QuadTree::QuadTree(Rect &&rect, int max_capacity)
        : QuadTree(std::move(rect), max_capacity, 0) {
}

QuadTree::QuadTree(Rect &&rect, float object_radius)
        : QuadTree(std::move(rect), 1, object_radius) {
}

QuadTree::QuadTree(Rect &&rect)
        : QuadTree(std::move(rect), 1, 0) {
}

QuadTree::~QuadTree() {
    for (QuadTree *child: children) {
        delete child;
    }

    omp_destroy_lock(&lock);
}

void QuadTree::subdivide() {
    children.reserve(4);

    Vec2 v1 = rect.top_left;
    Vec2 v2 = rect.size;
    Vec2 v2_half = v2 / 2;

    children.push_back(new QuadTree(Rect(v1, v2_half), max_capacity, object_radius));
    children.push_back(new QuadTree(Rect(v1 + Vec2(v2_half.x(), 0), v2_half), max_capacity, object_radius));
    children.push_back(new QuadTree(Rect(v1 + Vec2(0, v2_half.y()), v2_half), max_capacity, object_radius));
    children.push_back(new QuadTree(Rect(v1 + v2_half, v2_half), max_capacity, object_radius));

    for (const Vec2 &v: points) {
        for (QuadTree *child: children) {
            if (child->insert_aux(v)) {
                break;
            }
        }
    }
    points.clear();
    n_points = 0;
    subdivided = true;
}

bool QuadTree::insert_aux(const Vec2 &v) {
    if (!rect.contains(v)) {
        return false;
    }
    if (!subdivided) {
        omp_set_lock(&lock);
        if (subdivided) {
            // Magic !
            omp_unset_lock(&lock);
            goto insert_in_children;
        }
        if (n_points < max_capacity) {
            if (n_points == 0) {
                points.reserve(max_capacity);
            }
            points.push_back(v);
            n_points++;
            empty = false;

            omp_unset_lock(&lock);
            return true;
        } else {
            subdivide();
            omp_unset_lock(&lock);
            assert(insert_aux(v));
            return true;
        }
    }
    insert_in_children:
    if (subdivided) {
        for (int i = 0; i < 4; i++) {
            if (children[i]->insert_aux(v)) {
                return true;
            }
        }
        throw std::runtime_error("Failed to insert in all children");
    } else {
        throw std::runtime_error("Failed to insert in leaf");
    }
}

void QuadTree::insert(const Vec2 &v) {
    if (!rect.contains(v)) {
        throw std::runtime_error("Point not in quadtree");
    }
    if (!insert_aux(v)) {
        throw std::runtime_error("Failed to insert");
    }
}

void QuadTree::queryCircle(const Vec2 &v, std::vector<Vec2> &buffer) {
    if (!rect.intersects(Circle(v, object_radius))) {
        return;
    }
    if (subdivided) {
        for (QuadTree *child: children) {
            child->queryCircle(v, buffer);
        }
        return;
    }

    float sqr_radius = object_radius * object_radius;
    for (const Vec2 &p: points) {
        if ((p - v).squaredNorm() <= sqr_radius) {
            buffer.push_back(p);
        }
    }
}

std::optional<Vec2> QuadTree::naiveClosest(const Vec2 &v) {
    std::vector<Vec2> buffer;
    queryCircle(v, buffer);
    if (buffer.empty()) {
        return std::nullopt;
    }
    Vec2 closest = buffer[0];
    float min_dist = (closest - v).squaredNorm();
    for (const Vec2 &p: buffer) {
        float dist = (p - v).squaredNorm();
        if ((dist < min_dist && dist > 0) || min_dist == 0) {
            closest = p;
            min_dist = dist;
        }
    }
    return closest;
}

float QuadTree::closest(const Vec2 &v, std::optional<Vec2> &bestPoint, float bestDist) {
    if (subdivided) {
        std::vector<QuadTree *> ordered_children;
        ordered_children.reserve(4);
        for (QuadTree *child: children) {
            ordered_children.push_back(child);
        }
        std::sort(ordered_children.begin(), ordered_children.end(),
                  [v](QuadTree *a, QuadTree *b) {
                      Vec2 proj_a = a->rect.project(v);
                      Vec2 proj_b = b->rect.project(v);
                      return (proj_a - v).squaredNorm() < (proj_b - v).squaredNorm();
                  });
        for (QuadTree *child: ordered_children) {
            Vec2 proj = child->rect.project(v);
            float dist = (proj - v).squaredNorm();
            if (dist < bestDist || bestDist < 0) {
                bestDist = child->closest(v, bestPoint, bestDist);
            }
        }
    } else {
        for (const Vec2 &p: points) {
            float dist = (p - v).squaredNorm();
            if (dist == 0)
                continue;

            if (dist < bestDist || bestDist < 0) {
                bestDist = dist;
                bestPoint = p;
            }
        }
    }

    return bestDist;
}

std::optional<Vec2> QuadTree::closest(const Vec2 &v) {
    std::optional<Vec2> best;
    closest(v, best, -1);
    return best;
}

void QuadTree::initialize(const std::vector<Yaal> &yaals) {
    /**
     * Lock mutex of the quadtree when accessing it.
     * If it is not a leaf, delock the mutex and go to the children.
     * If it is a leaf (whether full or not), insert the Yaal and delock the mutex.
     */
    int nb_Yaals = yaals.size();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nb_Yaals; i++) {
        insert(yaals[i].position);
    }
}

void QuadTree::add_plants(const std::vector<Plant> &plants) {
    int nb_plants = plants.size();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nb_plants; i++) {
        insert(plants[i].position);
    }
}

void QuadTree::get_all_closest(const std::vector<Yaal> &yaals, std::vector<Vec2> &closestPoints) {
    int nb_Yaals = (int) yaals.size();
#pragma omp parallel for schedule(static)
    for (int i = 0; i < nb_Yaals; i++) {
        std::optional<Vec2> closestPoint = closest(yaals[i].position);
        if (closestPoint.has_value()) {
            closestPoints[i] = closestPoint.value();
        } else {
            closestPoints[i] = yaals[i].position;
        }
    }
}
