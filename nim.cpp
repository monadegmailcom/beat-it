#include "nim.h"

#include <algorithm>
#include <iostream>

using namespace std;

namespace nim {

bool operator==( Move const& lhs, Move const& rhs )
{
    return lhs.heap_index == rhs.heap_index && lhs.count == rhs.count;
}

Game::Game( 
    PlayerIndex player_index, size_t heap_count, vector< size_t >& heap_stack, 
    vector< Move >& move_stack, std::mt19937& g )
    : UndecidedGame( player_index ), heap_stack( heap_stack ), 
      heap_begin_index( heap_stack.size() - heap_count ), heap_count( heap_count ), 
      move_stack( move_stack ), moves_begin_index( move_stack.size()), g( g )
{
    if (!heap_count)
        throw std::invalid_argument( "at least one heap" );
    if (heap_count > heap_stack.size())
        throw std::invalid_argument( "heaps count exceeds heap stack size" );

    for (size_t i = 0; i < heap_count; ++i)
        for (size_t count = 1; count <= heap_stack[heap_begin_index + i]; ++count)
            move_stack.push_back( Move{ i, count } );
    moves_count = move_stack.size() - moves_begin_index;

    // shuffle order of moves to avoid the same order every time
    std::shuffle( move_stack.begin() + moves_begin_index, move_stack.end(), g );
}

Game::~Game()
{
    move_stack.erase( move_stack.begin() + moves_begin_index, move_stack.end());
    heap_stack.erase( heap_stack.begin() + heap_begin_index, heap_stack.end());
}

unique_ptr< ::Game< Move > > Game::apply( size_t index ) const
{
    size_t new_heap_count = 0;
    Move const& move = move_stack[moves_begin_index + index];
    for (auto i = 0; i != heap_count; ++i)
    {
        const size_t heap_size = heap_stack[heap_begin_index + i];
        
        if (i != move.heap_index)
        {
            heap_stack.push_back( heap_size );
            ++new_heap_count;
        }
        else if (move.count < heap_size)
        {
            heap_stack.push_back( heap_size - move.count );
            ++new_heap_count;
        } 
        else if (heap_size < move.count)
            throw std::invalid_argument( 
                "invalid move, heap size (" + to_string(heap_size) 
                + ") < move count (" + to_string(move.count) + ")" );            
    }
    unique_ptr< ::Game< Move > > game;
    if (!new_heap_count)
        game.reset( new WonGame< Move >( toggle( player_index )));
    else
        game.reset( new Game( toggle( player_index ), new_heap_count, heap_stack, move_stack, g));

    return game;
}

MoveRange< Move > Game::valid_moves() const
{
    const auto begin = move_stack.cbegin() + moves_begin_index;
    return std::ranges::subrange( begin, begin + moves_count );
}

HeapRange Game::get_heaps() const
{
    const auto begin = heap_stack.cbegin() + heap_begin_index;
    return std::ranges::subrange( begin, begin + heap_count );
}

namespace console {

size_t HumanPlayer::choose( ::Game< Move > const& game )
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
