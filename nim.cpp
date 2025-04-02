#include "nim.h"

#include <algorithm>

using namespace std;

namespace nim {

bool operator==( Move const& lhs, Move const& rhs )
{
    return lhs.heap_itr == rhs.heap_itr && lhs.count == rhs.count;
}

Game::Game( Player< Move > const& player, Player< Move > const& opponent, vector< size_t > _heaps )   
: UndecidedGame( player ), opponent( opponent ), heaps( _heaps ) 
{
    if (heaps.empty())
        throw std::invalid_argument( "at least one heap" );
    if (any_of( heaps.begin(), heaps.end(), []( size_t count ) { return count == 0; }))
        throw std::invalid_argument( "heaps must not be empty" );

    for (auto itr = heaps.begin(); itr != heaps.end(); ++itr)
        for (size_t count = 1; count <= *itr; ++count)
            moves.push_back( Move{ itr, count } );
    }

unique_ptr< ::Game > Game::apply( vector< Move >::const_iterator move_itr) const
{
    vector< size_t > new_heaps( heaps );
    auto heap_itr = new_heaps.begin();
    advance( heap_itr, distance( heaps.begin(), move_itr->heap_itr ));
    if (*heap_itr < move_itr->count)
        throw std::invalid_argument( 
            "invalid move, heap size (" + to_string(*heap_itr) 
            + ") < move count (" + to_string(move_itr->count) + ")" );
    *heap_itr -= move_itr->count;
    if (*heap_itr == 0)
        new_heaps.erase( heap_itr );
    ::Game* game = nullptr;
    if (new_heaps.empty())
        game = new WonGame( toggle( player.get_index() ));
    else
        game = new Game( opponent, player, new_heaps );

    return unique_ptr< ::Game >( game );
}

} // namespace nim {
