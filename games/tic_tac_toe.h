#include "../minimax-tree.h"
#include "../montecarlo.h"
#include "../alphazero.h"

#include <torch/script.h> // Main LibTorch header for loading models
#include <torch/torch.h>

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

const size_t G = 2 * 9;
const size_t P = 9;

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
        std::array< float, G >& ) const override;
};

using Player = ::alphazero::Player< Move, State, G, P >;

namespace libtorch {

struct Data : public ttt::alphazero::Data
{
    Data( std::mt19937& g, NodeAllocator& allocator, std::string const& model_path );
    
    float predict( Game const&, std::array< float, P >& policies ) override;

    torch::jit::script::Module module; // The loaded TorchScript model
    torch::Device device = torch::kCPU;  // Device to run inference on (CPU or CUDA)
}; 

} // namespace libtorch

namespace training {

using SelfPlay = ::alphazero::training::SelfPlay< Move, State, G, P >;
using Position = ::alphazero::training::Position< G, P >;

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