#include <cstdint>

enum PlayerIndex
{
    Player1 = 0,
    Player2
};

PlayerIndex toggle( PlayerIndex index )
{
    return PlayerIndex( (index + 1) % 2);
    // which one is faster?
    //return index == Player1 ? Player2 : Player1;
}

class Game
{
public:
    Game( PlayerIndex index) : index( index )
    {}  

    virtual ~Game() {}
    
    PlayerIndex current_player_index() const
    {
        return index;
    }
private:
    PlayerIndex index;
};

class DrawnGame : public Game
{
public:
    DrawnGame( PlayerIndex index ) : Game( index )
    {}
};

class UndecidedGame : public Game
{
public: 
    UndecidedGame( PlayerIndex index ) : Game( index )
    {}
};

class WonGame : public Game
{
public:
    WonGame( PlayerIndex index ) : Game( index )
    {}

    PlayerIndex winner() const
    {
        return current_player_index();
    }
};