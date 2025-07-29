#include <cmath>
#include <iostream>

class Statistics
{
public:
    void update( float value ) noexcept;
    // require: at least one update
    float mean() const;
    // require: at least one update
    float stddev() const;
    float min() const noexcept { return min_; }
    float max() const noexcept { return max_; }
    size_t count() const noexcept { return count_; }
private:
    float sum = 0.0f;
    float sum_square = 0.0f;
    float min_ = INFINITY;
    float max_ = -INFINITY;
    size_t count_ = 0;
};

std::ostream& operator<<( std::ostream& os, Statistics const& );
