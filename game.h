#include <cstdint>
#include <memory>
#include <vector>

enum PlayerIndex
{
    Player1 = 0,
    Player2
};

PlayerIndex toggle( PlayerIndex );

template< typename MoveT >
class UndecidedGame;

template< typename MoveT >
class Player 
{
public:
    Player( PlayerIndex index ) : index( index ) {}
    virtual ~Player() {}
    PlayerIndex get_index() const { return index; }
    // promise: returns iterator to move if valid_moves is not empty
    virtual std::vector< MoveT >::const_iterator choose( 
        std::vector< MoveT > const& valid_moves ) = 0;
private:
    PlayerIndex index;
};

class Game
{
public:
    virtual ~Game() {}
    
    virtual PlayerIndex current_player_index() const = 0;
};

template< typename MoveT >
class UndecidedGame : public Game
{
public: 
    UndecidedGame( Player< MoveT > const& player )
    : player( player ) {}
    Player< MoveT > const& next_to_make_a_move() const { return player; }
    PlayerIndex current_player_index() const override
    {
        return player.get_index();
    }
    virtual std::vector< MoveT > const& valid_moves() const = 0;

    // require: move iterator has to be in valid_moves()
    // require: returned game's player has toggled index
    virtual std::unique_ptr< Game > apply( std::vector< MoveT >::const_iterator) const = 0;
protected:
    Player< MoveT > const& player;
};

class DrawnGame : public Game
{
public:
    DrawnGame( PlayerIndex current_player_index ) 
        : player_index( current_player_index ) {}
    PlayerIndex current_player_index() const override { return player_index; }
private:
    PlayerIndex player_index;
};

class WonGame : public Game
{
public:
    WonGame( PlayerIndex winner ) : winner_index( winner ) {}
    PlayerIndex current_player_index() const override { return winner_index; }

    PlayerIndex winner() const
    {
        return winner_index;
    }
private:
    PlayerIndex winner_index;
};