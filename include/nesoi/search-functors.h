#pragma once

#include <algorithm>
#include <limits>

namespace nesoi
{

template<class NN>
struct HandleDistance
{
    typedef             typename NN::PointHandle                                    PointHandle;
    typedef             typename NN::DistanceType                                   DistanceType;
    typedef             typename NN::HDContainer                                    HDContainer;

                        HandleDistance()                                            {}
                        HandleDistance(PointHandle pp, DistanceType dd):
                            p(pp), d(dd)                                            {}
    bool                operator<(const HandleDistance& other) const                { return d < other.d; }

    PointHandle         p;
    DistanceType        d;
};

template<class HandleDistance>
struct NNRecord
{
    typedef         typename HandleDistance::PointHandle                            PointHandle;
    typedef         typename HandleDistance::DistanceType                           DistanceType;

                    NNRecord()                                                      { result.d = std::numeric_limits<DistanceType>::infinity(); }
    DistanceType    operator()(PointHandle p, DistanceType d)                       { if (d < result.d) { result.p = p; result.d = d; } return result.d; }
    HandleDistance  result;
};

template<class HandleDistance>
struct rNNRecord
{
    typedef         typename HandleDistance::PointHandle                            PointHandle;
    typedef         typename HandleDistance::DistanceType                           DistanceType;
    typedef         typename HandleDistance::HDContainer                            HDContainer;

                    rNNRecord(DistanceType r_): r(r_)                               {}
    DistanceType    operator()(PointHandle p, DistanceType d)
    {
        if (d <= r)
            result.push_back(HandleDistance(p,d));
        return r;
    }

    DistanceType    r;
    HDContainer     result;
};

template<class HandleDistance>
struct kNNRecord
{
    typedef         typename HandleDistance::PointHandle                            PointHandle;
    typedef         typename HandleDistance::DistanceType                           DistanceType;
    typedef         typename HandleDistance::HDContainer                            HDContainer;

                    kNNRecord(unsigned k_): k(k_)                                   {}
    DistanceType    operator()(PointHandle p, DistanceType d)
    {
        if (result.size() < k)
        {
            result.push_back(HandleDistance(p,d));
            std::push_heap(result.begin(), result.end());
            if (result.size() < k)
                return std::numeric_limits<DistanceType>::infinity();
        } else if (d < result[0].d)
        {
            std::pop_heap(result.begin(), result.end());
            result.back() = HandleDistance(p,d);
            std::push_heap(result.begin(), result.end());
        }
        return result[0].d;
    }

    unsigned        k;
    HDContainer     result;
};

}
