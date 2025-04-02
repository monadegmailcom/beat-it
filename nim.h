#include "game.h"

namespace nim {

struct Move
{
    std::vector< size_t >::const_iterator heap_itr;
    size_t count;
};

bool operator==( Move const& lhs, Move const& rhs );

// the nim game, the player who takes the last object looses
class Game : public UndecidedGame< Move >
{
public:
    // require: at least one heap and heaps are not empty
    Game( Player< Move > const& player, Player< Move > const& opponent, std::vector< size_t > heaps );
    std::vector< size_t > const& get_heaps() const { return heaps; }

    std::vector< Move > const& valid_moves() const override
    {
        return moves;
    }
    std::unique_ptr< ::Game > apply( std::vector< Move >::const_iterator) const override;
private:
    Player< Move > const& opponent;
    std::vector< size_t > heaps;
    std::vector< Move > moves;
};

} // namespace nim {
