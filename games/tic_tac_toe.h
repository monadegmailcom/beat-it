#include "../minimax-tree.h"
#include "../montecarlo.h"
#include "../alphazero.h"
#include "../libtorch_util.h"

#include <array>
#include <iostream>

// forward decl so we do not have to include libtorch_helper.h here
namespace torch {
namespace jit {
struct Module;
}}

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
using Player = ::Player< Move >;

extern const std::array< Move, 3 > wins[8];

Symbol player_index_to_symbol( PlayerIndex );

const Move no_move = 9;
const State empty_state = {
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty };

namespace minimax {

double score( State const& state );

using PlayerFactory = ::PlayerFactory< Move >;

class Player : public ::minimax::Player< Move, State >
{
public:
    Player( Game const& game, unsigned depth, unsigned seed )
    : ::minimax::Player< Move, State >( game, depth, seed ) {}
    double score( Game const& game ) const override
    { return minimax::score( game.get_state() ); };
};

namespace tree {

using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::minimax::tree::NodeAllocator< Move, State >;

class Player : public ::minimax::tree::Player< Move, State >
{
public:
    Player( Game const& game, unsigned depth, unsigned seed,
        NodeAllocator& allocator )
    : ::minimax::tree::Player< Move, State >( game, depth, seed, allocator ) {}
    double score( Game const& game ) const override
    { return minimax::score( game.get_state() ); };
};

} // namespace tree {

} // namespace minimax {

namespace montecarlo
{

using Player = ::montecarlo::Player< Move, State >;
using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::montecarlo::NodeAllocator< Move, State >;

} // namespace montecarlo

namespace alphazero {

using NodeAllocator = ::alphazero::NodeAllocator< Move, State >;

const size_t G = 3 * 9;
const size_t P = 9;

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
namespace async {
using Player = ::libtorch::async::Player< BasePlayer >;
} // namespace async {
namespace sync {
using Player = ::libtorch::sync::Player< BasePlayer >;
} // namespace sync {
} // namespace libtorch {

} // namespace alphazero {

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
} // namespace ttt

std::ostream& operator<<( std::ostream&, ttt::Game const& );

template<>
struct GameState< ttt::Move, ttt::State >
{
    static void next_valid_move(
        std::optional< ttt::Move >&, PlayerIndex, ttt::State const& );

    static void get_valid_moves(
        std::vector< ttt::Move >& moves, PlayerIndex, ttt::State const& state );

    static ttt::State apply(
        ttt::Move const&, PlayerIndex, ttt::State const& );

    static GameResult result( PlayerIndex, ttt::State const& );
};