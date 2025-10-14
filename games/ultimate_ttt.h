#include "tic_tac_toe.h"

#include <iostream>

namespace uttt
{

struct Move {
    ttt::Move big_move;
    ttt::Move small_move;

    friend bool operator==( uttt::Move const&, uttt::Move const& ) = default;
    friend std::ostream& operator<<( std::ostream&, uttt::Move const& );
};

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
using PlayerIndexDispatch = ::TaggedDispatch< State, PlayerIndex >;
using MoveDispatch = ::TaggedDispatch< State, Move >;
using PlayerFactory = ::PlayerFactory< uttt::Move >;

extern const State empty_state;
const Move no_move = { ttt::no_move, ttt::no_move };

namespace console
{

class HumanPlayer : public Player
{
public:
    explicit HumanPlayer( Game const& game ) : game( game ) {}
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

namespace tree {

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

} // namespace tree 
} // namespace minimax 

namespace montecarlo
{

using Player = ::montecarlo::Player< Move, State >;
using NodeAllocator = ::montecarlo::NodeAllocator< Move, State >;

} // namespace montecarlo

namespace alphazero {

using Value = ::alphazero::Value< Move, State >;
using Node = ::Node< Value >;
using NodeAllocator = ::NodeAllocator< Value >;

const size_t G = 4 * 81;
const size_t P = 81;

class Player : public ::alphazero::Player< Move, State, G, P >
{
public:
    using base_type = ::alphazero::Player< Move, State, G, P >;
    using base_type::base_type;
private:
    size_t move_to_policy_index( Move const& ) const override;
    std::array< float, G > serialize_state(
        Game const& ) const override;
};

namespace training {
using Position = ::alphazero::training::Position< G, P >;
using Selfplay = ::alphazero::training::SelfPlay< Move, State, G, P >;
} // namespace training

namespace libtorch {
using InferenceService = ::libtorch::InferenceService< G, P >;
} // namespace libtorch
} // namespace alphazero
} // namespace uttt

std::ostream& operator<<( std::ostream&, uttt::Game const& );

std::ostream& operator<<(
    std::ostream&, TaggedDispatch< uttt::State, PlayerIndex > const& );

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
