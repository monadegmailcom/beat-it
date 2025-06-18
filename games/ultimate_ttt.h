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

using Data = ::minimax::Data< Move >;

class Player : public ::minimax::Player< Move, State >
{
public:
    Player( Game const& game, double weight, unsigned depth, Data& );
    double score( Game const& game ) const override;
private:
    const double weight;
};

using PlayerFactory = ::PlayerFactory< uttt::Move >;

namespace tree {

using Data = ::minimax::tree::Data< Move, State >;
using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::minimax::tree::NodeAllocator< Move, State >;

class Player : public ::minimax::tree::Player< Move, State >
{
public:
    Player( Game const& game, double weight, unsigned depth, Data& );
    double score( Game const& game ) const override;
private:
    const double weight;
};

} // namespace tree {
} // namespace minimax {

namespace montecarlo 
{

using Data = ::montecarlo::Data< Move, State >;
using Player = ::montecarlo::Player< Move, State >;
using Buffer = char[sizeof( Player )];
using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::montecarlo::NodeAllocator< Move, State >;

} // namespace montecarlo 

namespace alphazero {

using NodeAllocator = ::alphazero::NodeAllocator< Move, State >;

const size_t G = 81;
const size_t P = 81;

struct Data : public ::alphazero::Data< Move, State, G, P >
{
    Data( std::mt19937& g, NodeAllocator& allocator )
    : ::alphazero::Data< Move, State, G, P >( g, allocator ) {}

    float predict( 
        Game const&,
        std::array< float, P >& policies ) override;
    size_t move_to_policy_index( Move const& ) const override;
    void serialize_state( 
        Game const&,
        std::array< float, G >& game_state_player1,
        std::array< float, G >& game_state_player2 ) const override;
};

using Player = ::alphazero::Player< Move, State, G, P >;

namespace training {

using SelfPlay = ::alphazero::training::SelfPlay< Move, State, G, P >;

} // namespace training {

} // namespace alphazero {

} // namespace uttt

std::ostream& operator<<( std::ostream&, uttt::Game const& );

template<>
struct GameState< uttt::Move, uttt::State > 
{
    static void next_valid_move( 
        std::optional< uttt::Move >&, PlayerIndex, uttt::State const& );

    static void get_valid_moves(
        std::vector< uttt::Move >& moves, PlayerIndex, uttt::State const& state );

    static uttt::State apply( 
        uttt::Move const&, PlayerIndex, uttt::State const& );

    static GameResult result( PlayerIndex, uttt::State const& );
};
