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
    Player( Game const& game, double weight, unsigned depth, unsigned seed );
    double score( Game const& game ) const override;
private:
    const double weight;
};

using PlayerFactory = ::PlayerFactory< uttt::Move >;

namespace tree {

using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::minimax::tree::NodeAllocator< Move, State >;

class Player : public ::minimax::tree::Player< Move, State >
{
public:
    Player( Game const& game, double weight, unsigned depth, unsigned seed,
        NodeAllocator& allocator );
    double score( Game const& game ) const override;
private:
    const double weight;
};

} // namespace tree {
} // namespace minimax {

namespace montecarlo
{

using Player = ::montecarlo::Player< Move, State >;
using Buffer = char[sizeof( Player )];
using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::montecarlo::NodeAllocator< Move, State >;

} // namespace montecarlo

namespace alphazero {

using NodeAllocator = ::alphazero::NodeAllocator< Move, State >;

const size_t G = 4 * 81;
const size_t P = 81;

namespace training {
using Position = ::alphazero::training::Position< G, P >;
using Selfplay = ::alphazero::training::SelfPlay< Move, State, G, P >;
} // namespace training {

class BasePlayer : public ::alphazero::Player< Move, State, G, P >
{
public:
    BasePlayer(
        Game const& game,
        float c_base, float c_init, size_t simulations,
        NodeAllocator& allocator );
protected:
    std::array< float, G > serialize_state( Game const& ) const override;
    size_t move_to_policy_index( Move const& ) const override;
};

namespace libtorch {
namespace sync {
using Player = ::libtorch::sync::Player< BasePlayer >;
} // namespace sync {

namespace async {
using Player = ::libtorch::async::Player< BasePlayer >;
} // namespace async {
} // namespace libtorch {
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
