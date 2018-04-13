#pragma once

#include <vector>

template<class T>
class BarycentersContainer
{
    public:
        using Real                      = T;
        using CoordinateContainer       = std::vector<Real>;

    public:
                BarycentersContainer(size_t n, size_t dim):
                    n_(n), dim_(dim), coordinates(n * (dim + 1))    {}

        Real    coordinate(size_t i, size_t c) const                { return coordinates[i * (dim_ + 1) + c]; }
        Real&   coordinate(size_t i, size_t c)                      { return coordinates[i * (dim_ + 1) + c]; }

        Real    weight(size_t i) const                              { return coordinate(i, dim_); }
        Real&   weight(size_t i)                                    { return coordinate(i, dim_); }

        Real    distance(size_t u, size_t v) const;

        size_t  dimension() const                                   { return dim_; }
        size_t  size() const                                        { return n_; }

    private:
        size_t              n_;
        size_t              dim_;
        CoordinateContainer coordinates;
};

template<class T>
typename BarycentersContainer<T>::Real
BarycentersContainer<T>::
distance(size_t u, size_t v) const
{
    Real sq_dist = 0;
    for (size_t i = 0; i < dimension(); ++i)
    {
        Real diff = coordinate(u, i) - coordinate(v, i);
        sq_dist += diff*diff;
    }

    // intersection happens whenever lower weight point appears
    if (sq_dist <= std::abs(weight(u) - weight(v)))
        return std::max(-weight(u), -weight(v));

    Real result;
    result  = (weight(u) - weight(v))/sq_dist;
    result += 1.;
    result *= result;
    result *= sq_dist / 4;
    result -= weight(u);

    return result;
}
