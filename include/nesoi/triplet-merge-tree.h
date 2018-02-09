#pragma once

#include <vector>
#include <cstdint>
#if !defined(NESOI_NO_PARALLEL)
#include <atomic>
#endif

namespace nesoi
{

template<class Value_, class Vertex_ = std::uint32_t>
class TripletMergeTree
{
    public:
        using Vertex    = Vertex_;
        using Value     = Value_;

        struct Edge
        {
            Vertex  through;
            Vertex  to;

            bool    operator==(const Edge& other) const     { return through == other.through && to == other.to; }
            bool    operator!=(const Edge& other) const     { return through != other.through || to != other.to; }
        };

#if !defined(NESOI_NO_PARALLEL)
        using AtomicEdge = std::atomic<Edge>;
#else
        using AtomicEdge = Edge;
#endif

        using Function  = std::vector<Value>;
        using Tree      = std::vector<AtomicEdge>;

    public:
                    TripletMergeTree()                      {}
                    TripletMergeTree(size_t size, bool negate = false):
                        negate_(negate),
                        function_(size),
                        tree_(size)                         { for (auto& e : tree_) e = dummy(); }

        // no copy because of std::atomic<...> in tree_
                            TripletMergeTree(const TripletMergeTree&)   = delete;
                            TripletMergeTree(TripletMergeTree&&)        = default;
        TripletMergeTree&   operator=(const TripletMergeTree&)          = delete;
        TripletMergeTree&   operator=(TripletMergeTree&&)               = default;

        bool        cmp(Vertex u, Vertex v) const;

        void        add(Vertex x, Value v);
        void        link(Vertex u, Vertex s, Vertex v)      { tree_[u] = Edge {s,v}; }
        bool        cas_link(Vertex u,
                             Vertex os, Vertex ov,
                             Vertex s,  Vertex v)           { auto op = Edge {os,ov}; auto p = Edge {s,v}; return tree_[u].compare_exchange_weak(op, p); }

        Edge        repair(Vertex u);
        void        repair();

        void        merge(Vertex u, Vertex v);
        void        merge(Vertex u, Vertex s, Vertex v);
        Vertex      representative(Vertex u, Vertex a) const;

        size_t      size() const                            { return tree_.size(); }
        bool        contains(const Vertex& u) const         { return (*this)[u] != dummy(); }

        bool        negate() const                          { return negate_; }
        void        set_negate(bool negate)                 { negate_ = negate; }

        template<class F>
        void        traverse_persistence(const F& f) const;

        Edge        dummy() const                           { return Edge { static_cast<Vertex>(-1), static_cast<Vertex>(-1)}; }
        Edge        operator[](Vertex u) const              { return tree_[u]; }
        Value       value(Vertex u) const                   { return function_[u]; }

        template<class F>
        void        for_each_vertex(const F& f) const;

    private:
        bool        compare_exchange(AtomicEdge& e, AtomicEdge expected, AtomicEdge desired)
        {
#if !defined(NESOI_NO_PARALLEL)
            return e.compare_exchange_weak(expected, desired);
#else
            if (e != expected)
                return false;
            e = desired;
            return true;
#endif
        }

    private:
        bool        negate_;
        Function    function_;
        Tree        tree_;
};

}

#include "triplet-merge-tree.hpp"
