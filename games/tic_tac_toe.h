#include "../minimax-tree.h"
#include "../montecarlo.h"
#include "../alphazero.h"

#include <array>
#include <iostream>

// forward decl so we do not have to include libtorch_helper.h here
namespace libtorch {
class InferenceManager;
} // namespace libtorch {

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

using Data = ::minimax::Data< Move >;
using PlayerFactory = ::PlayerFactory< Move >;

class Player : public ::minimax::Player< Move, State >
{
public:
    Player( Game const& game, unsigned depth, Data& data ) 
    : ::minimax::Player< Move, State >( game, depth, data ) {}
    double score( Game const& game ) const override 
    { return minimax::score( game.get_state() ); };
};

namespace tree {

using Data = ::minimax::tree::Data< Move, State >;
using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::minimax::tree::NodeAllocator< Move, State >;

class Player : public ::minimax::tree::Player< Move, State >
{
public:
    Player( Game const& game, unsigned depth, Data& data ) 
    : ::minimax::tree::Player< Move, State >( game, depth, data ) {}
    double score( Game const& game ) const override 
    { return minimax::score( game.get_state() ); };
};

} // namespace tree {

} // namespace minimax {

namespace montecarlo 
{

using Data = ::montecarlo::Data< Move, State >;
using Player = ::montecarlo::Player< Move, State >;
using PlayerFactory = ::PlayerFactory< Move >;
using NodeAllocator = ::montecarlo::NodeAllocator< Move, State >;

} // namespace montecarlo 

namespace alphazero {

using NodeAllocator = ::alphazero::NodeAllocator< Move, State >;

const size_t G = 3 * 9;
const size_t P = 9;

namespace libtorch {

class Player : public ::alphazero::Player< Move, State, G, P >
{
public:
    Player( 
        Game const& game, 
        float c_base,
        float c_init,
        size_t simulations,
        NodeAllocator& allocator,
        ::libtorch::InferenceManager& inference_manager )
    : ::alphazero::Player< Move, State, G, P >( game, c_base, c_init, simulations, allocator),
      inference_manager( inference_manager ) {}
protected:
    ::libtorch::InferenceManager& inference_manager;

    std::pair< float, std::array< float, P >> predict( std::array< float, G > const& ) override;
    std::array< float, G > serialize_state( Game const& ) const override;
    size_t move_to_policy_index( Move const& ) const override;
};

} // namespace libtorch {

namespace training {

using Position = ::alphazero::training::Position< G, P >;
using SelfPlay = ::alphazero::training::SelfPlay< Move, State, G, P >;

} // namespace training {
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