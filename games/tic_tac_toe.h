#include "../minimax-tree.h"
#include "../montecarlo.h"
#include "../alphazero.h"
#include "../libtorch_util.h"

#include <array>
#include <iostream>
#include <utility>

// forward decl so we do not have to include libtorch_helper.h here
namespace torch::jit {
struct Module;
}

namespace ttt
{

enum class Symbol : char
{
    Empty = ' ',
    Player1 = 'X',
    Player2 = 'O'
};

inline std::ostream& operator<<( std::ostream& os, Symbol symbol )
{
    os << std::to_underlying(symbol);
    return os;
}

using Move = uint8_t;
using State = std::array< Symbol, 9 >;
using Game = ::Game< Move, State >;
using Player = ::Player< Move >;

extern const std::array< std::array< Move, 3 >, 8 > wins;

Symbol player_index_to_symbol( PlayerIndex );

const Move no_move = 9;
const State empty_state = {
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty,
    Symbol::Empty, Symbol::Empty, Symbol::Empty };

using PlayerFactory = ::PlayerFactory< Move >;

namespace minimax {

double score( State const& state );

using PlayerFactory = ::PlayerFactory< Move >;

class Player : public ::minimax::Player< Move, State >
{
public:
    using ::minimax::Player< Move, State >::Player;
    double score( Game const& game ) const override
    { return minimax::score( game.get_state() ); };
};

namespace tree {

class Player : public ::minimax::tree::Player< Move, State >
{
public:
    using ::minimax::tree::Player< Move, State >::Player;
    double score( Game const& game ) const override
    { return minimax::score( game.get_state() ); };
};

} // namespace tree
} // namespace minimax

namespace montecarlo
{

using Value = ::montecarlo::Value< Move, State >;
using Node = ::Node< Value >;
using Player = ::montecarlo::Player< Move, State >;
using PlayerFactory = ::PlayerFactory< Move >;

} // namespace montecarlo

namespace alphazero {

using Value = ::alphazero::Value< Move, State >;
using Node = ::Node< Value >;

const size_t G = 3 * 9;
const size_t P = 9;

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
