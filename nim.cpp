#include "nim.h"

#include <algorithm>
#include <iostream>

using namespace std;

namespace nim {

bool operator==( Move const& lhs, Move const& rhs )
{
    return lhs.heap_index == rhs.heap_index && lhs.count == rhs.count;
}

Game::Game( PlayerIndex player_index, vector< size_t >&& _heaps, vector< Move >& move_stack )
    : UndecidedGame( player_index ), heaps( _heaps ), move_stack( move_stack ), 
      moves_begin_index( move_stack.size())
    
{
    if (heaps.empty())
        throw std::invalid_argument( "at least one heap" );
    if (any_of( heaps.begin(), heaps.end(), []( size_t count ) { return count == 0; }))
        throw std::invalid_argument( "heaps must not be empty" );

    for (size_t i = 0; i < heaps.size(); ++i)
        for (size_t count = 1; count <= heaps[i]; ++count)
            move_stack.push_back( Move{ i, count } );
    moves_count = move_stack.size() - moves_begin_index;
}

Game::~Game()
{
    move_stack.erase( move_stack.begin() + moves_begin_index, move_stack.end());
}

unique_ptr< ::Game > Game::apply( size_t index ) const
{
    vector< size_t > new_heaps( heaps );
    Move const& move = move_stack[moves_begin_index + index];
    auto heap_itr = new_heaps.begin() + move.heap_index;
    if (*heap_itr < move.count)
        throw std::invalid_argument( 
            "invalid move, heap size (" + to_string(*heap_itr) 
            + ") < move count (" + to_string(move.count) + ")" );
    *heap_itr -= move.count;
    if (*heap_itr == 0)
        new_heaps.erase( heap_itr );
    ::Game* game = nullptr;
    if (new_heaps.empty())
        game = new WonGame( toggle( player_index ));
    else
        game = new Game( toggle( player_index ), std::move( new_heaps ), move_stack);

    return unique_ptr< ::Game >( game );
}

namespace console {

size_t HumanPlayer::choose( UndecidedGame< Move > const& game )
{
    auto nim_game = dynamic_cast< nim::Game const* >( &game );
    if (!nim_game)
        throw std::invalid_argument( "invalid game type" );

    cout << "player " << game.current_player_index() << endl
         << "heaps: " << endl;
    unsigned index = 0;
    for (auto const& heap : nim_game->get_heaps())
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
        cout << "count? (1-" << nim_game->get_heaps()[heap_index - 1] << "): ";
        size_t count;
        std::cin >> count;
        if (count < 1 || count > nim_game->get_heaps()[heap_index - 1])
        {
            cout << "invalid count" << endl;
            continue;
        }        
        auto move_itr = find_if( game.valid_moves().begin(), game.valid_moves().end(),
                                  [heap_index, count]( Move const& move )
                                  { return move == Move{ heap_index - 1, count }; } );
        if (move_itr == game.valid_moves().end())
        {
            cout << "invalid move" << endl;
            continue;
        }
        return move_itr - game.valid_moves().begin();
    }
}

} // namespace console {
} // namespace nim {
