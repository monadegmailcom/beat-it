#include "tic_tac_toe.h"
#include <iostream>

namespace uttt
{

struct Move {
    ttt::Move big_move;
    ttt::Move small_move;
};

bool operator==( uttt::Move const& lhs, uttt::Move const& rhs );

// require: small_states, big_state and last_small_move are consistent
struct State 
{
    std::array< ttt::State, 9 > small_states;
    std::array< GameResult, 9 > big_state;
    ttt::Move next_big_move = ttt::no_move;
    mutable std::optional< GameResult > game_result_cache;
};

using Game = ::Game< Move, State >;
using Player = ::Player< Move >;

extern const State empty_state; 
const Move no_move = { ttt::no_move, ttt::no_move };

namespace console
{

class HumanPlayer : public Player
{
public:
    HumanPlayer( Game const& game ) : game( game ) {}
    Move choose_move() override;
    void apply_opponent_move( Move const& ) override;
private:
    Game game;
};

} // namespace console

namespace minimax {

class Player : public ::minimax::Player< Move, State >
{
public:
    Player( Game const& game, double weight, unsigned depth, ::minimax::Data< Move >& );
    double score( Game const& game ) const override;
private:
    const double weight;
};

using Data = ::minimax::Data< Move >;
using Buffer = char[sizeof( Player )];
using PlayerFactory = ::PlayerFactory< uttt::Move >;

PlayerFactory player_factory(
    Game const& game, double weight, unsigned depth, Data& data, Buffer );

} // namespace minimax {

} // namespace uttt

std::ostream& operator<<( std::ostream&, uttt::Game const& );

template<>
struct GameState< uttt::Move, uttt::State >
{
    static void append_valid_moves( 
        std::vector< uttt::Move >& move_stack, PlayerIndex, 
        uttt::State const& );

    static uttt::State apply( 
        uttt::Move const&, PlayerIndex, uttt::State const& );

    static GameResult result( PlayerIndex, uttt::State const& );
};
