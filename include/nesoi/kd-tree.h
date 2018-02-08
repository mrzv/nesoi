#pragma once

#include "search-functors.h"

namespace nesoi
{
    // Traits_ provides Coordinate, DistanceType, PointType, dimension(), distance(p1,p2), coordinate(p,i)
    template< class Traits_ >
    class KDTree
    {
        public:
            using Traits            = Traits_;
            using HandleDistance    = nesoi::HandleDistance<KDTree>;

            using Point             = typename Traits::PointType;
            using PointHandle       = typename Traits::PointHandle;
            using Coordinate        = typename Traits::Coordinate;
            using DistanceType      = typename Traits::DistanceType;
            using HandleContainer   = std::vector<PointHandle>;
            using HDContainer       = std::vector<HandleDistance>;
            using Result            = HDContainer;

        public:
                            KDTree(const Traits& traits):
                                traits_(traits)                             {}

                            KDTree(const Traits& traits, HandleContainer&& handles);

            template<class Range>
                            KDTree(const Traits& traits, const Range& range);

            template<class Range>
            void            init(const Range& range);

            HandleDistance  find(PointHandle q) const;
            Result          findR(PointHandle q, DistanceType r) const;     // all neighbors within r
            Result          findK(PointHandle q, size_t k) const;           // k nearest neighbors

            HandleDistance  find(const Point& q) const                      { return find(traits().handle(q)); }
            Result          findR(const Point& q, DistanceType r) const     { return findR(traits().handle(q), r); }
            Result          findK(const Point& q, size_t k) const           { return findK(traits().handle(q), k); }

            template<class ResultsFunctor>
            void            search(PointHandle q, ResultsFunctor& rf) const;

            const Traits&   traits() const                                  { return traits_; }

        private:
            void            init();

            using HCIterator = typename HandleContainer::iterator;
            using KDTreeNode = std::tuple<HCIterator, HCIterator, size_t>;

            struct CoordinateComparison;
            struct OrderTree;

        private:
            Traits          traits_;
            HandleContainer tree_;
    };
}

#include "kd-tree.hpp"
