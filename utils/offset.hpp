struct Offset {
    // stores the offset of the top, bottom, left, and right of the map, for padding and shared bands
    int top;
    int bottom;
    int left;
    int right;

    Offset operator+(const Offset &other) const {
        return {top + other.top, bottom + other.bottom, left + other.left, right + other.right};
    }
};