#pragma once

#include "player.h"
#include "game.h"
#include "node.h"

#include <random>

namespace alphazero
{

namespace detail 
{
    template< typename MoveT, typename StateT >
    struct Value
    {
        Value( Game< MoveT, StateT > const& game, MoveT const& move )
        : game( game ), move( move ), game_result( game.result()) {}

        Game< MoveT, StateT > game;
        MoveT move; // the previous move resulting in this game
        const GameResult game_result; // the cached game result

        size_t visits = 0;
        // policy probalities (priors) of choosing this move
        float nn_policy = 0.0; 
        // 1 for win, 0.5 for draw, 0 for loss
        // value sum of the game outcomes: -1 loss, 0 draw, 1 win from the perspective
        // of the current player
        float nn_value_sum = 0.0;
    };

    float game_result_2_score( GameResult, PlayerIndex );

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
    virtual void serialize_state( 
        Game< MoveT, StateT > const&,
        std::array< float, G >& game_state_player1,
        std::array< float, G >& game_state_player2 ) const = 0;

    std::mt19937& g;
    NodeAllocator< MoveT, StateT >& allocator;
};

template< typename MoveT, typename StateT, size_t G, size_t P >
class Player : public ::Player< MoveT >
{
public:
    using GameType = Game< MoveT, StateT >;
    using DataType = Data< MoveT, StateT, G, P >;

    Player( 
        GameType const& game, 
        float c_base,
        float c_init,
        size_t simulations,
        DataType& data )
    : data( data ), 
      root( new (data.allocator.allocate()) 
            NodeType( ValueType( game, MoveT()), data.allocator ),
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
            simulation( *root );
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
    
        NodeType* new_root = nullptr;
        if (itr == root->get_children().end())
            new_root = new 
                (data.allocator.allocate()) 
                NodeType( ValueType( root->get_value().game.apply( move ), move), 
                    data.allocator );
        else
        {
            new_root = &*itr;
            root->get_children().erase( itr );
        }

        root.reset( new_root );
    }
protected:
    using ValueType = detail::Value< MoveT, StateT >;
    using NodeType = Node< ValueType >;
    
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

    // upper confidence bound
    float ucb( ValueType const& value, size_t parent_visits )
    {
        const float c = 
            std::log( (parent_visits + c_base + 1) / c_base) + c_init;
        float q = 0.0;
        if (value.visits != 0)
            // switch sign because value is from the child's perspective
            q = -value.nn_value_sum / value.visits;
        const float p = value.nn_policy;
            
        return q + c * p * std::sqrt( parent_visits / (value.visits + 1));
    }

    NodeType& select( NodeType& node )
    {  
        return *std::ranges::max_element( 
            node.get_children(),
            [this, parent_visits = node.get_value().visits]
            (auto const& a, auto const& b)
            { 
                return   ucb( a.get_value(), parent_visits ) 
                       < ucb( b.get_value(), parent_visits ); 
            });
    }

    // promise: node is expanded
    float nn_eval( NodeType& node )
    {
        auto& value = node.get_value();

        // if node not expanded, do so
        if (node.get_children().empty())
            for (MoveT const& move : value.game)
            {
                auto child = new 
                    (node.get_allocator().allocate()) 
                    Node( ValueType( value.game.apply( move ), move ), 
                          node.get_allocator());                            

                node.get_children().push_front( *child );
            }

        std::array< float, P > policies;
        const float nn_value = data.predict( value.game, policies ); 

        // convert logits of legal moves to probabilities with softmax 
        // and save in children
        float policy_sum = 0.0f;
        for (auto& child : node.get_children())
        {
            const float p = std::exp( policies[
                data.move_to_policy_index( child.get_value().move )]); 
            child.get_value().nn_policy = p;
            policy_sum += p;
        }
        // normalize
        for (auto& child : node.get_children())
            child.get_value().nn_policy /= policy_sum;

        return nn_value;
    }

    float simulation( NodeType& node )
    {
        auto& value = node.get_value();
        ++value.visits;

        // set target value
        float nn_value;
        // from game result if game is decided
        if (value.game_result != GameResult::Undecided)
            nn_value = detail::game_result_2_score( 
                value.game_result, value.game.current_player_index());
        // from nn if node is not yet expanded
        else if (node.get_children().empty())
            nn_value = nn_eval( node );
        // or recursivly otherwise
        else 
            // recursively simulate the selected child node
            // negate sign of return value because its from the opponent's perspective
            nn_value = -simulation( select( node ));

        value.nn_value_sum += nn_value;

        return nn_value;
    }

    DataType& data;
    NodePtr< ValueType > root;
    float c_base;
    float c_init;
    const size_t simulations;
};

namespace training {

template< size_t G, size_t P >
struct Position
{
    std::array< float, G > game_state_player1;
    std::array< float, G > game_state_player2;
    std::array< float, P > target_policy;
    float target_value = 0.0f;
    float current_player = 0.0f; // cast from PlayerIndex 0 and 1, player next to move
};

template< typename MoveT, typename StateT, size_t G, size_t P >
class SelfPlay : public alphazero::Player< MoveT, StateT, G, P >
{
public:
    using Base = alphazero::Player< MoveT, StateT, G, P >;
    using PositionType = Position< G, P >; 

    SelfPlay( 
        Base::GameType const& game, 
        float c_base,
        float c_init,
        float dirichlet_alpha,
        float dirichlet_epsilon,
        size_t simulations,
        size_t opening_moves,
        Base::DataType& data,
        std::vector< PositionType>& positions )
    : Base( game, c_base, c_init, simulations, data ),
      opening_moves( opening_moves ), dirichlet_epsilon( dirichlet_epsilon ), 
      gamma_dist( dirichlet_alpha, 1.0f ), positions( positions ) {}

    void run( Base::GameType const& game )
    {
        GameResult game_result;
        for (game_result = game.result();
             game_result == GameResult::Undecided;
             game_result = this->root->get_value().game_result)
            choose_move();

        for (auto& position : positions)
            position.target_value = detail::game_result_2_score(
                game_result, 
                position.current_player < 0.5f ? Player1 : Player2 );
    }
protected:
    // promise: root node is expanded
    void add_dirichlet_noise() 
    {
        // expand node if not expanded
        if (this->root->get_children().empty())
            this->nn_eval( *this->root );

        // add noise for root node
        for (auto& child : this->root->get_children())
        {
            float& policy = child.get_value().nn_policy;
            policy *= 1.0 - dirichlet_epsilon;
            policy += gamma_dist( this->data.g ) * dirichlet_epsilon;
        }
    }

    MoveT choose_move() override
    {
        // add dirichlet noise to root node before first simulation
        add_dirichlet_noise();

        // root node is expanded now
        for (size_t i = this->simulations; i != 0; --i)
            this->simulation( *this->root );
        
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
        PositionType& position = positions.back();

        position.current_player = static_cast< float >( 
            this->root->get_value().game.current_player_index());

        // append serialized game state
        this->data.serialize_state( 
            this->root->get_value().game, 
            position.game_state_player1,
            position.game_state_player2 );
        
        size_t sum_visits = 0;
        for (auto const& child : this->root->get_children())
            sum_visits += child.get_value().visits;

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
                    child_itr->get_value().visits ) / sum_visits;
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
    float dirichlet_epsilon;
    std::gamma_distribution< float > gamma_dist;
    std::vector< PositionType >& positions;
    size_t move_count = 0;
};

} // namespace training

} // namespace alphazero