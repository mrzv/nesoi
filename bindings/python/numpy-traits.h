#pragma once

#include <pybind11/numpy.h>
namespace py = pybind11;

template<class Real_>
struct NumPyTraits
{
    using Real  = Real_;
    using Array = py::array_t<Real>;

    struct PointHandle
    {
        size_t i;
        bool        operator==(const PointHandle& other) const          { return i == other.i; }
        bool        operator!=(const PointHandle& other) const          { return !(*this == other); }
        bool        operator<(const PointHandle& other) const           { return i < other.i; }
        bool        operator>(const PointHandle& other) const           { return i > other.i; }
    };
    struct PointType { size_t i; };

    using Coordinate    = Real;
    using DistanceType  = Real;

                    NumPyTraits(const Array& a):
                        a_(a)                                           { dim_ = a_.shape()[1]; }

    DistanceType    distance(PointHandle p1, PointHandle p2) const      { return sqrt(sq_distance(p1, p2)); }
    DistanceType    sq_distance(PointHandle p1, PointHandle p2) const
    {
        Real sq_dist = 0;
        for (unsigned i = 0; i < dim_; ++i)
        {
            Real x = coordinate(p1, i);
            Real y = coordinate(p2, i);
            sq_dist += (x-y)*(x-y);
        }
        return sq_dist;
    }
    unsigned        dimension() const                                   { return dim_; }
    Real            coordinate(PointHandle h, unsigned i) const         { return *a_.data(h.i, i); }

    size_t          id(PointHandle h) const                             { return h.i; }

    PointHandle     handle(size_t i) const                              { return PointHandle { i }; }
    PointHandle     handle(PointType p) const                           { return PointHandle { p.i }; }

    Array           a_;
    unsigned        dim_;
};
