#include "nim.h"

#include <iostream>

using namespace std;

namespace nim {

bool operator==( Move const& lhs, Move const& rhs )
{
    return lhs.heap_index == rhs.heap_index && lhs.count == rhs.count;
}

namespace console {

Move choose( PlayerIndex player_index, vector< size_t > const& heaps, vector< nim::Move > const& valid_moves )
{
    cout << "player " << player_index << endl
         << "heaps: " << endl;
    unsigned index = 0;
    for (auto const& heap : heaps)
        cout << ++index << ". " << heap << endl;

    while (true)
    {
        cout << "heap index? (1-" << index << "): ";
        size_t heap_index;
        std::cin >> heap_index;
        if (heap_index < 1 || heap_index > index)
        {
            cout << "invalid heap index" << endl;
            continue;
        }        
        cout << "count? (1-" << heaps[heap_index - 1] << "): ";
        size_t count;
        std::cin >> count;
        if (count < 1 || count > heaps[heap_index - 1])
        {
            cout << "invalid count" << endl;
            continue;
        }        

        auto move_itr = find_if( valid_moves.begin(), valid_moves.end(),
                                  [heap_index, count]( Move const& move )
                                  { return move == Move{ heap_index - 1, count }; } );
        if (move_itr == valid_moves.end())
        {
            cout << "invalid move" << endl;
            continue;
        }
        return *move_itr;
    }
}

} // namespace console {
} // namespace nim {
