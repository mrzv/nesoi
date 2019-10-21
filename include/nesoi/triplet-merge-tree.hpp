#include "parallel.h"

template<class Value, class Vertex>
bool
nesoi::TripletMergeTree<Value, Vertex>::
cmp(Vertex u, Vertex v) const
{
    Value uval = function_[u];
    Value vval = function_[v];
    if (negate_)
        return uval > vval || (uval == vval && u > v);
    else
        return uval < vval || (uval == vval && u < v);
}

template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
add(Vertex x, Value v)
{
    function_[x] = v;
    link(x,x,x);
}

template<class Value, class Vertex>
typename nesoi::TripletMergeTree<Value, Vertex>::Vertex
nesoi::TripletMergeTree<Value, Vertex>::
representative(Vertex u, Vertex a) const
{
    Edge sv = tree_[u];
    Vertex s = sv.through;
    Vertex v = sv.to;
    while (!cmp(a, s) && s != v)
    {
        u = v;
        sv = tree_[u];
        s  = sv.through;
        v  = sv.to;
    }
    return u;
}

template<class Value, class Vertex>
typename nesoi::TripletMergeTree<Value, Vertex>::Edge
nesoi::TripletMergeTree<Value, Vertex>::
repair(Vertex u)
{
    Vertex s, v, ov;
    do
    {
        Edge sov = tree_[u];
        s  = sov.through;
        ov = sov.to;
        v = representative(u, s);
        if (u == v) return Edge {s,v};
    } while (!cas_link(u,s,ov,s,v));

    return Edge {s,v};
}

template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
repair()
{
    for_each_vertex([&](Vertex u) { repair(u); });
}

template<class Value, class Vertex>
template<class F>
void
nesoi::TripletMergeTree<Value, Vertex>::
for_each_vertex(Vertex n, const F& f) const
{
    for_each(n, f);
}

template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
merge(Vertex u, Vertex v)
{
    if (cmp(u, v))
        merge(v, v, u);
    else
        merge(u, u, v);
}

template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
merge(Vertex u, Vertex s, Vertex v)
{
    while(true)
    {
        u = representative(u, s);
        v = representative(v, s);
        if (u == v)
            break;

        Edge   su  = tree_[u];
        Vertex s_u = su.through,
               u_  = su.to;
        Edge   sv  = tree_[v];
        Vertex s_v = sv.through,
               v_  = sv.to;

        // check that s_u and s_v haven't changed since running representative
        if (s_u != u_ && !cmp(s, s_u))
            continue;
        if (s_v != v_ && !cmp(s, s_v))
            continue;

        if (cmp(v, u))
        {
            std::swap(u, v);
            std::swap(s_u, s_v);
            std::swap(u_, v_);
        }

        bool success = cas_link(v, s_v, v_, s, u);
        if (success)
        {
            if (v == v_)
                break;

            s = s_v;
            v = v_;
        } // else: rinse and repeat
    }
}

template<class Value, class Vertex>
template<class F>
void
nesoi::TripletMergeTree<Value, Vertex>::
traverse_persistence(const F& f) const
{
    for (Vertex u = 0; u < size(); ++u)
    {
        Edge   sv = tree_[u];
        Vertex s  = sv.through,
               v  = sv.to;
        if (u != s || u == v) f(u, s, v);
    }
}
