#pragma once

#include <vector>
#include <cstdint>
#include <atomic>

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
        };

        using Function  = std::vector<Value>;
        using Tree      = std::vector<std::atomic<Edge>>;

    public:
                    TripletMergeTree(size_t size, bool negate = false):
                        negate_(negate),
                        function_(size),
                        tree_(size)                         { for (auto& e : tree_) e = dummy(); }

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
        bool        contains(const Vertex& x) const         { return tree_[x] != dummy(); }

        bool        negate() const                          { return negate_; }
        void        set_negate(bool negate)                 { negate_ = negate; }

        template<class F>
        void        traverse_persistence(const F& f);

        Edge        dummy() const                           { return Edge { static_cast<Vertex>(-1), static_cast<Vertex>(-1)}; }
        Edge        parent(Vertex u) const                  { return tree_[u]; }

    private:
        template<class F>
        void        for_each_vertex(const F& f) const;

    private:
        bool        negate_;
        Function    function_;
        Tree        tree_;
};

}

#include "triplet-merge-tree.hpp"
