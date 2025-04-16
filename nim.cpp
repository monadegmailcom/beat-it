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
    PlayerIndex player_index, size_t heap_begin_index, vector< size_t >& heap_stack )
    : ::Game< Move >( player_index ), heap_stack( heap_stack ), 
      heap_begin_index( heap_begin_index )      
{
    if (heap_begin_index > heap_stack.size())
        throw std::invalid_argument( "heaps count exceeds heap stack size" );
}

Game::~Game()
{
    heap_stack.erase( heap_stack.begin() + heap_begin_index, heap_stack.end());
}

GameResult Game::result() const
{
    if (all_of( heap_stack.cbegin() + heap_begin_index, heap_stack.cend(),
                []( size_t heap_size ) { return heap_size == 0; } ))
    {
        if (player_index == Player1)
            return GameResult::Player1Win;
        else
            return GameResult::Player2Win;
    }
    else
        return GameResult::Undecided;
}

unique_ptr< ::Game< Move > > Game::apply( Move const& move ) const
{    
    const size_t prev_size = heap_stack.size();
    size_t heap_index = heap_begin_index + move.heap_index;

    if (heap_index > prev_size)
        throw std::invalid_argument( 
            "invalid move heap index (" + to_string(move.heap_index) + ")" );
    if (heap_stack[heap_index] < move.count)
        throw std::invalid_argument( 
            "invalid move count (" + to_string(heap_stack[heap_index]) 
            + ") < move count (" + to_string(move.count) + ")" );

    heap_stack.resize( 2 * prev_size - heap_begin_index);
    copy( heap_stack.cbegin() + heap_begin_index, heap_stack.cbegin() + prev_size, 
          heap_stack.begin() + prev_size );
    heap_index = prev_size + move.heap_index;
    heap_stack[heap_index] -= move.count;

    return make_unique< Game >( toggle( player_index ), prev_size, heap_stack );
}

void Game::append_valid_moves( std::vector< Move >& move_stack ) const
{
    for (size_t i = 0, end = heap_stack.size() - heap_begin_index; i != end; ++i)
        for (size_t count = 1; count <= heap_stack[heap_begin_index + i]; ++count)
            move_stack.push_back( Move{ i, count } );
}

HeapRange Game::get_heaps() const
{
    return std::ranges::subrange( heap_stack.cbegin() + heap_begin_index, heap_stack.cend());
}

namespace console {

Move HumanPlayer::choose( ::Game< Move > const& game )
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
        vector< Move > move_stack;
        game.append_valid_moves( move_stack );
        auto move_itr = find_if( move_stack.begin(), move_stack.end(),
                                  [heap_index, count]( Move const& move )
                                  { return move == Move{ heap_index - 1, count }; } );
        if (move_itr == move_stack.end())
        {
            cout << "invalid move" << endl;
            continue;
        }
        return *move_itr;
    }
}

} // namespace console {
} // namespace nim {
