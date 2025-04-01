#include <cmath>

#include "minimax.h"

using namespace std;

namespace minimax
{

double max_value( PlayerIndex index )
{
    if (index == Player1)
        return -INFINITY;
    else
        return INFINITY;
}

function< bool (double, double) > cmp( PlayerIndex index )
{
    if (index == Player1)
        return less< double >();
    else
        return greater< double >();
}

}; // namespace minimax