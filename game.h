#pragma once

#include <cstdint>
#include <boost/iterator/iterator_facade.hpp>
#include <optional>
#include <vector>

enum PlayerIndex
{
    Player1 = 0,
    Player2
};

PlayerIndex toggle( PlayerIndex );

enum GameResult : char
{
    Draw = 0,
    Player1Win,
    Player2Win,
    Undecided
};

// Helper base class to provide a default get_valid_moves implementation
template<typename DerivedGameState, typename MoveT, typename StateT>
struct GameStateBase
{
    static void get_valid_moves( std::vector< MoveT >& moves, PlayerIndex player_index, StateT const& state )
    {
        moves.clear();
        for (std::optional< MoveT > move;;)
        {
            // Call next_valid_move on the actual specialized GameState
            DerivedGameState::next_valid_move( move, player_index, state );
            if (move)
                moves.push_back(*move);
            else
                break;
        }
    }
};

// for each game specialize game state
template< typename MoveT, typename StateT >
struct GameState : public GameStateBase<GameState<MoveT, StateT>, MoveT, StateT>
{
    // Specializations are expected to provide these:
    static void next_valid_move( std::optional< MoveT >&, PlayerIndex, StateT const& );
    static StateT apply( MoveT const&, PlayerIndex, StateT const& );
    static GameResult result( PlayerIndex, StateT const& state );
    // get_valid_moves is inherited from GameStateDefaultGetValidMovesProvider by default.
    // Specializations can provide their own get_valid_moves to override it.
};

template< typename MoveT, typename StateT >
class Game
{
public:
    Game( PlayerIndex player_index, StateT const& state ) 
        : player_index( player_index ), state( state ) {}
    
    PlayerIndex current_player_index() const { return player_index; }
    GameResult result() const 
    { return GameState< MoveT, StateT >::result( player_index, state ); }

    // require: move has to be a valid move
    // promise: reset next valid move
    Game apply( MoveT const& move ) const
    { 
        return Game( 
            toggle( player_index ), 
            GameState< MoveT, StateT >::apply( move, player_index, state )); 
    }
    StateT const& get_state() const { return state; }

    class MoveItr : public boost::iterator_facade<
                        MoveItr, // Derived class
                        MoveT,                  // Value type
                        boost::single_pass_traversal_tag, // Iterator category
                        const MoveT&            // Reference type
                    >    
    {
    public:
        explicit MoveItr( Game const& game ) : game( game ) {}

        // Dereference operator
        const MoveT& dereference() const { return *move; }

        // Pre-increment operator
        void increment() 
        {
            GameState<MoveT, StateT>::next_valid_move(
                move, game.current_player_index(), game.get_state());
        }

        // Equality comparison
        bool equal(const MoveItr& other) const { return move == other.move; }
    private:
        friend class boost::iterator_core_access;
        friend class Game;
        std::optional< MoveT > move; // unset optional represents end of iteration
        Game< MoveT, StateT > const& game;
    };

    MoveItr begin() const
    {
        MoveItr itr( *this );
        return ++itr;
    }

    MoveItr end() const
    {
        return MoveItr( *this);
    }
private:
    PlayerIndex player_index;
    StateT state;
};
