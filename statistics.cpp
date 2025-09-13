#include "statistics.h"

#include <stdexcept>
#include <source_location>

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
        throw source_location::current();
    return sum / static_cast< float >( count_ );
}

float Statistics::stddev() const
{
    const float mean_ = mean(); // will throw if count_ == 0
    return std::sqrtf( sum_square / static_cast< float >( count_ ) 
        - mean_ * mean_);
}

void Statistics::reset() noexcept
{
    sum = 0.0f;
    sum_square = 0.0f;
    min_ = INFINITY;
    max_ = -INFINITY;
    count_ = 0;
}

ostream& operator<<( ostream& os, Statistics const& stats ) // NOSONAR
{
    os
        << "mean = " << stats.mean() << ", stddev = " << stats.stddev() 
        << ", min = " << stats.min() << ", max = " << stats.max()
        << ", count = " << stats.count() << '\n';
    return os;
}
