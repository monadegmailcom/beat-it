#pragma once

#include "player.h"
#include "game.h"
#include "node.h"

#include <random>

namespace alphazero
{

namespace detail {

template< typename MoveT, typename StateT >
struct Value
{
    Value( Game< MoveT, StateT > const& game, MoveT const& move )
    : game( game ), move( move ), game_result( game.result()) {}

    Game< MoveT, StateT > game;
    MoveT move; // the previous move resulting in this game
    const GameResult game_result; // the cached game result

    size_t visits = 0;
    float nn_policy = 0.0; // probality of choosing this move
    // 1 for win, 0.5 for draw, 0 for loss
    float nn_value = 0.0; // value of the game state
    float nn_value_sum = 0.0;
};

} // namespace detail {

template< typename MoveT, typename StateT >
using NodeAllocator = ::NodeAllocator< detail::Value< MoveT, StateT > >;

// G game state size, P target policy size
template< typename MoveT, typename StateT, size_t G, size_t P >
struct Data
{
    Data( std::mt19937& g, NodeAllocator< MoveT, StateT >& allocator )
    : g( g ), allocator( allocator ) {}

    virtual ~Data() = default;

    // predict game state value and policy vector from nn
    // promise: policies contain probability distribution of moves
    virtual float predict( 
        Game< MoveT, StateT > const&,
        std::array< float, P >& policies ) = 0;

    // promise: return index of move in policy_vector
    virtual size_t move_to_policy_index( MoveT const& ) const = 0;

    // promise: write serialized game into passed arrays
    virtual void serialize_game( 
        Game< MoveT, StateT > const&,
        std::array< float, G >& game_state_player1,
        std::array< float, G >& game_state_player2 ) const = 0;

    std::mt19937& g;
    NodeAllocator< MoveT, StateT >& allocator;
};

namespace detail {

template< typename MoveT, typename StateT, size_t G, size_t P >
float nn_eval( 
    Node< Value< MoveT, StateT >>& node, 
    Data< MoveT, StateT, G, P >& data )
{
    auto& value = node.get_value();
    std::array< float, P > policies;
    value.nn_value = data.predict( value.game, policies ); 
    for (auto& child : node.get_children())
        // todo: convert logits to probabilities with softmax?
        child.get_value().nn_policy = policies[
            data.move_to_policy_index( child.get_value().move )]; 
    return value.nn_value;
}

float game_result_2_score( GameResult, PlayerIndex );

// upper confidence bound
template< typename MoveT, typename StateT >
float ucb( Value< MoveT, StateT > const& value, 
    size_t parent_visits, float c_base, float c_init )
{
    const float c = 
        std::log( (parent_visits + c_base + 1) / c_base) + c_init;
    float q = 0.0;
    if (value.visits != 0)
        q = value.nn_value_sum / value.visits;
    const float p = value.nn_policy;
        
    return q + c * p * std::sqrt( parent_visits / (value.visits + 1));
}

template< typename MoveT, typename StateT >
Node< Value< MoveT, StateT >>& select( 
    Node< Value< MoveT, StateT >>& node, 
    float c_base, float c_init )
{  
    return *std::ranges::max_element( 
        node.get_children(),
        [c_base, c_init, 
         parent_visits = node.get_value().visits]
        (auto const& a, auto const& b)
        { 
            return ucb( a.get_value(), parent_visits, c_base, c_init ) 
                 < ucb( b.get_value(), parent_visits, c_base, c_init ); 
        });
}

template< typename MoveT, typename StateT >
void add_children( Node< Value< MoveT, StateT >>& node )
{
    auto& value = node.get_value();
    for (MoveT const& move : value.game)
    {
        auto child = new 
            (node.get_allocator().allocate()) 
            Node( 
                Value( value.game.apply( move ), move ), 
                node.get_allocator());                            

        node.get_children().push_front( *child );
    }
}

template< typename MoveT, typename StateT, size_t G, size_t P >
float simulation( 
    Node< Value< MoveT, StateT >>& node, 
    Data< MoveT, StateT, G, P >& data, 
    float c_base, float c_init)
{
    auto& value = node.get_value();
    ++value.visits;

    float nn_value;

    if (value.game_result != GameResult::Undecided)
        nn_value = game_result_2_score( 
            value.game_result, value.game.current_player_index());
    else if (value.visits == 1) // nn eval on first visit
    {   
        add_children( node );
        nn_value = nn_eval< MoveT, StateT >( node, data );
    }
    else 
        // recursively simulate the selected child node
        // negate sign of return value because its from the opponent's perspective
        nn_value = -simulation( 
            select( node, c_base, c_init ), data, c_base, c_init );

    value.nn_value_sum += nn_value;

    return nn_value;
}

} // namespace detail {

template< typename MoveT, typename StateT, size_t G, size_t P >
class Player : public ::Player< MoveT >
{
public:
    Player( 
        Game< MoveT, StateT > const& game, 
        float c_base,
        float c_init,
        size_t simulations,
        Data< MoveT, StateT, G, P >& data )
    : data( data ), 
      root( new (data.allocator.allocate()) 
            Node< detail::Value< MoveT, StateT >>( 
                detail::Value< MoveT, StateT >( game, MoveT()), data.allocator ),
            [&data](auto ptr) { deallocate( data.allocator, ptr ); }
          ),
      c_base( c_base ), c_init( c_init ), simulations( simulations )
    {
        if (!simulations)
            throw std::invalid_argument( "simulations must be > 0" );
    }

    MoveT choose_move() override
    {
        for (size_t i = simulations; i != 0; --i)
            simulation( *root, this->data, c_base, c_init );
        auto itr = choose_best_move();
        
        if (itr == root->get_children().end())
            throw std::runtime_error( "no move choosen" );

        auto new_root = &*itr;
        root->get_children().erase( itr );
        root.reset( new_root );

        return root->get_value().move;
    }

    void apply_opponent_move( MoveT const& move ) override
    {
        auto itr = 
            std::ranges::find_if( 
                root->get_children(), 
                [move](auto const& node)
                { return node.get_value().move == move; } );
        Node< detail::Value< MoveT, StateT >>* new_root = nullptr;

        if (itr == root->get_children().end())
            new_root = new (this->data.allocator.allocate()) 
                   Node< detail::Value< MoveT, StateT >>( 
                        detail::Value< MoveT, StateT >(
                            root->get_value().game.apply( move ), move), 
                        this->data.allocator );
        else
        {
            root->get_children().erase( itr );
            new_root = &*itr;
        }

        root.reset( new_root );
    }
protected:
    // Apple clang version 17.0.0 (clang-1700.0.13.3) produces strange errors 
    // when i specify the return value iterator type explicitly:
    // Node< ValueT > member boost::intrusive::list< Node > children has incomplete type
    auto choose_best_move()
    {
        // choose child with most visits
        return std::ranges::max_element( root->get_children(),
            [](auto const& a, auto const& b)
            { return a.get_value().visits < b.get_value().visits; } );
    }

    Data< MoveT, StateT, G, P >& data;
    NodePtr< detail::Value< MoveT, StateT > > root;
    float c_base;
    float c_init;
    size_t simulations;
};

namespace training {

template< size_t G, size_t P >
struct Position
{
    std::array< float, G > game_state_player1;
    std::array< float, G > game_state_player2;
    std::array< float, P > target_policy;
    float target_value = 0.0f;
};

template< typename MoveT, typename StateT, size_t G, size_t P >
class SelfPlay : public alphazero::Player< MoveT, StateT, G, P >
{
public:
    SelfPlay( 
        Game< MoveT, StateT > const& game, 
        float c_base,
        float c_init,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        size_t simulations,
        size_t opening_moves,
        Data< MoveT, StateT, G, P >& data,
        std::vector< Position< G, P >>& positions )
    : Player< MoveT, StateT, G, P >( game, c_base, c_init, simulations, data ),
      dirichlet_alpha( dirichlet_alpha ), opening_moves( opening_moves ), 
      dirichlet_epsilon( dirichlet_epsilon ), positions( positions ) {}

    void run( Game< MoveT, StateT > const& game )
    {
        GameResult game_result;
        for (game_result = game.result();
             game_result == GameResult::Undecided;
             game_result = this->root->get_value().game_result)
            choose_move();

        float target_value = 0.0f;
        if (game_result == GameResult::Player1Win)
            target_value = -1.0f;
        else if (game_result == GameResult::Player2Win)
            target_value = 1.0f;

        for (auto& position : positions)
            position.target_value = target_value;
    }
protected:
    void add_dirichlet_noise() override 
    {
        std::gamma_distribution< float > gamma_dist(
            this->dirichlet_alpha, 1.0f );
               
        for (auto& child : this->root->get_children())
        {
            float& policy = child.get_value().nn_policy;
            policy *= 1.0 - dirichlet_epsilon;
            policy += gamma_dist( this->data.g ) * dirichlet_epsilon;
        }
    }

    MoveT choose_move() override
    {
        // add dirichlet noise on the very first move
        if (this->root->get_children().empty())
        {    
            simulation( *this->root, this->data, this->c_base, this->c_init );
            --this->simulations;
            add_dirichlet_noise();
        }

        for (size_t i = this->simulations; i != 0; --i)
            simulation( *this->root, this->data, this->c_base, this->c_init );
        
        auto itr = this->root->get_children().begin();
        if (++move_count <= opening_moves)
            itr = choose_opening_move();
        else
            itr = this->choose_best_move();

        if (itr == this->root->get_children().end())
            throw std::runtime_error( "no move choosen" );

        append_training_data();

        auto new_root = &*itr;
        this->root->get_children().erase( itr );
        this->root.reset( new_root );

        return this->root->get_value().move;
    }

    void append_training_data()
    {
        positions.emplace_back();
        Position< G, P >& position = positions.back();

        // append serialized game state
        this->data.serialize( this->root->get_value().game, position.game_state );
        
        // append new target policies        
        for (size_t policy_index = 0; policy_index < P; ++policy_index)
        {
            auto child_itr = std::ranges::find_if( 
                this->root->get_children(), 
                [this, policy_index](auto const& child)
                { return policy_index == this->data.move_to_policy_index( child.get_value().move ); } );
            if (child_itr == this->root->get_children().end())
                position.target_policy[policy_index] = 0.0f;
            else
                position.target_policy[policy_index] = static_cast< float >( 
                    child_itr->get_value().visits ) / (this->root->get_value().visits - 1);
        }
    }

    auto choose_opening_move()
    {
        // sample from children in opening phase by visit distribution
        // so we are more versatile in the opening
        size_t visits = 0;
        for (auto const& child : this->root->get_children())
            visits += child.get_value().visits;
        int r = this->data.g() % visits;
        for (auto itr = this->root->get_children().begin(); 
                itr != this->root->get_children().end(); ++itr)
        {
            r -= itr->get_value().visits;
            if (r < 0)
                return itr;
        }

        return this->root->get_children().end();
    }

    size_t opening_moves;
    float dirichlet_alpha;
    float dirichlet_epsilon;
    std::vector< Position< G, P >>& positions;
    size_t move_count = 0;
};

} // namespace training

} // namespace alphazero