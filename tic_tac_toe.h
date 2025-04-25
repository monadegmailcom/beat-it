#include "minimax.h"

#include <array>
#include <iostream>

namespace tic_tac_toe
{

enum Symbol : char
{
    Empty = ' ',
    Player1 = 'X',
    Player2 = 'O',
    Undecided = 'U'
};

using Move = uint8_t;
using State = std::array< Symbol, 9 >;
using Game = ::Game< Move, State >;
using Player = ::Player< Move, State >;

GameResult symbol_to_winner( Symbol );

Symbol player_index_to_symbol( PlayerIndex );

const Move no_move = 9;
const State empty_state = { 
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty };
    
namespace minimax {

class Player : public ::minimax::Player< Move, State >
{
public:
    Player( unsigned depth, std::mt19937& g ) : ::minimax::Player< Move, State >( depth, g ) {}
    double score( Game const& game ) const override;
};

} // namespace minimax {

namespace console
{

class HumanPlayer : public Player
{
public:
    Move choose( Game const& game ) override;
};


} // namespace console
} // namespace tic_tac_toe

std::ostream& operator<<( std::ostream&, tic_tac_toe::Game const& );

template<>
struct GameState< tic_tac_toe::Move, tic_tac_toe::State >
{
    static void append_valid_moves( 
        std::vector< tic_tac_toe::Move >& move_stack, PlayerIndex, tic_tac_toe::State const& state )
    {
        for (char index = 0; index != 9; ++index)
            if (state[index] == tic_tac_toe::Symbol::Empty)
                move_stack.push_back( index );
    }

    static tic_tac_toe::State apply( 
        tic_tac_toe::Move const& move, PlayerIndex player_index, tic_tac_toe::State const& state )
    {
        if (move >= 9)
            throw std::invalid_argument( "invalid move" );
        if (state[move] != tic_tac_toe::Symbol::Empty)
            throw std::invalid_argument( "invalid move" );

        tic_tac_toe::State new_state = state;
        new_state[move] = tic_tac_toe::player_index_to_symbol( player_index );
        return new_state;
    }

    static GameResult result( PlayerIndex player_index, tic_tac_toe::State const& state )
    {
        tic_tac_toe::Symbol symbol;

        // check rows
        for (size_t row = 0; row != 3; ++row)
        {
            symbol = state[row * 3];
            if (symbol != tic_tac_toe::Symbol::Empty && state[row * 3 + 1] == symbol 
                && state[row * 3 + 2] == symbol)
                return tic_tac_toe::symbol_to_winner( symbol );
        }
        // check columns
        for (size_t column = 0; column != 3; ++column)
        {
            symbol = state[column];
            if (symbol != tic_tac_toe::Symbol::Empty && state[column + 3] == symbol
                && state[column + 6] == symbol)
                return tic_tac_toe::symbol_to_winner( symbol );
        }
        // check diagonals
        symbol = state[0];
        if (symbol != tic_tac_toe::Symbol::Empty && state[4] == symbol && state[8] == symbol)
            return tic_tac_toe::symbol_to_winner( symbol );
        symbol = state[2];
        if (symbol != tic_tac_toe::Symbol::Empty && state[4] == symbol && state[6] == symbol)
            return tic_tac_toe::symbol_to_winner( symbol );

        // check draw
        for (auto symbol : state)
            if (symbol == tic_tac_toe::Symbol::Empty)
                return GameResult::Undecided;

        return GameResult::Draw;
    }    
};