#pragma once

#include <vector>
#include <utility>
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
        using AtomicEdge   = std::atomic<Edge>;
#else
        using AtomicEdge   = Edge;
#endif

        using Function     = std::vector<Value>;
        using Tree         = std::vector<AtomicEdge>;
        using IndexArray   = std::vector<Vertex>;
        using IndexDiagram = std::vector<std::pair<Vertex, Vertex>>;

    public:
                    TripletMergeTree()                      {}
                    TripletMergeTree(size_t size, bool negate = false):
                        negate_(negate),
                        function_(size),
                        cache_(size),
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
                             Vertex s,  Vertex v)
#if !defined(NESOI_NO_PARALLEL)
        { auto op = Edge {os,ov}; auto p = Edge {s,v}; return tree_[u].compare_exchange_weak(op, p); }
#else
        { tree_[u] = Edge {s,v}; return true; }             // NB: this is not technically CAS, but it's Ok in serial
#endif


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
        void        for_each_vertex(const F& f) const       { for_each_vertex(size(), f); }

        template<class F>
        void        for_each_vertex(Vertex n, const F& f) const;

        void        compute_mt(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* values, bool negate);

        Function    simplify(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* values, Value epsilon, bool negate, bool squash_root);
        Function    simplify(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* values, Value epsilon, Value level_value, bool negate);

        IndexDiagram noise_diagram_points(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* val_ptr, Value epsilon, bool negate, bool squash_root);
        IndexDiagram noise_diagram_points_ls(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* val_ptr, Value epsilon, Value level_value, bool negate);

    private:


        Vertex      dummy_vertex() const                    { return static_cast<Vertex>(-1); }
        Vertex      dummy_vertex_2() const                  { return static_cast<Vertex>(-2); }

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

        void        cache_all_reps(Value epsilon, bool squash_root);
        void        cache_all_reps(Value epsilon, Value level_value);
        Vertex      simplification_repr(Vertex u, Value epsilon, bool squash_root);
        void        cache_simplification_repr(Vertex u, Value epsilon, Value level_value);



    private:
        bool        negate_;
        Function    function_;
        IndexArray  cache_;
        Tree        tree_;
};

}

#include "triplet-merge-tree.hpp"
