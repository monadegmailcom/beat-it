#include "../minimax.h"

#include <array>
#include <iostream>

namespace ttt
{

enum Symbol : char
{
    Empty = ' ',
    Player1 = 'X',
    Player2 = 'O'
};

using Move = uint8_t;
using State = std::array< Symbol, 9 >;
using Game = ::Game< Move, State >;
using Player = ::Player< Move, State >;

extern const std::array< Move, 3 > wins[8];

Symbol player_index_to_symbol( PlayerIndex );

const Move no_move = 9;
const State empty_state = { 
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty };

namespace minimax {

double score( State const& state );

class Player : public ::minimax::Player< Move, State >
{
public:
    Player( unsigned depth, std::mt19937& g ) : ::minimax::Player< Move, State >( depth, g ) {}
    double score( Game const& game ) const override 
    { return minimax::score( game.get_state() ); };
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
} // namespace ttt

std::ostream& operator<<( std::ostream&, ttt::Game const& );

template<>
struct GameState< ttt::Move, ttt::State >
{
    static void append_valid_moves( 
        std::vector< ttt::Move >& move_stack, PlayerIndex, ttt::State const& );

    static ttt::State apply( 
        ttt::Move const&, PlayerIndex, ttt::State const& );

    static GameResult result( PlayerIndex, ttt::State const& );
};