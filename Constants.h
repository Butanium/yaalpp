//
// Created by clementd on 01/02/24.
//

#ifndef YAALPP_CONSTANTS_H
#define YAALPP_CONSTANTS_H

namespace Constants {
    static constexpr float DELTA_T = 1;
    static constexpr float EPSILON = 1e-6;
    static constexpr float PHYSICS_EPSILON = 1e-2;
    namespace Yaal {
        static constexpr float MIN_SPEED = 0.1f;
        static constexpr float MAX_SPEED = 1.0f;
        static constexpr int MIN_FIELD_OF_VIEW = 1;
        static constexpr int MAX_FIELD_OF_VIEW = 10;
        static constexpr int MIN_SIZE = 9;
        static constexpr int MAX_SIZE = 9;
    }
    namespace Environment {
        static constexpr int FILTER_SIZE = 7;
    }
}

#endif //YAALPP_CONSTANTS_H
