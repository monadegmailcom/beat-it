#include "statistics.h"

using namespace std;

void Statistics::update( float value ) noexcept
{
    ++count_;
    sum += value;
    sum_square += value * value;
    min_ = min_ < value ? min_ : value;
    max_ = max_ > value ? max_ : value;
}

float Statistics::mean() const { return sum / static_cast< float >( count_ ); }

float Statistics::stddev() const
{
    const float mean_ = mean();
    return std::sqrtf( sum_square / static_cast< float >( count_ ) -
                       mean_ * mean_ );
}

void Statistics::reset() noexcept
{
    sum = 0.0f;
    sum_square = 0.0f;
    min_ = INFINITY;
    max_ = -INFINITY;
    count_ = 0;
}

void Statistics::join( Statistics const &other ) noexcept
{
    sum += other.sum;
    sum_square += other.sum_square;
    min_ = min_ < other.min_ ? min_ : other.min_;
    max_ = max_ > other.max_ ? max_ : other.max_;
    count_ += other.count_;
}

ostream &operator<<( ostream &os, Statistics const &stats ) // NOSONAR
{
    os << "mean = " << stats.mean() << ", stddev = " << stats.stddev()
       << ", min = " << stats.min() << ", max = " << stats.max()
       << ", count = " << stats.count() << '\n';
    return os;
}
