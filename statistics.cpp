#include "statistics.h"

#include <stdexcept>

using namespace std;

void Statistics::update( float value ) noexcept
{
    ++count_;
    sum += value;
    sum_square += value * value;
    min_ = min_ < value ? min_ : value;
    max_ = max_ > value ? max_ : value;
}

float Statistics::mean() const
{
    if (!count_)
        throw runtime_error( "no updates" );
    return sum / count_;
}

float Statistics::stddev() const
{
    const float mean_ = mean(); // will throw if count_ == 0
    return std::sqrtf( sum_square / count_ - mean_ * mean_);
}

ostream& operator<<( ostream& os, Statistics const& stats )
{
    os
        << "mean = " << stats.mean() << ", stddev = " << stats.stddev() << '\n'
        << "min = " << stats.min() << ", max = " << stats.max() << '\n'
        << "count = " << stats.count() << '\n';
    return os;
}
