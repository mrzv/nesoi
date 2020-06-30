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

template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
compute_mt(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* val_ptr, bool negate)
{
    for(size_t v = 0; v < size(); ++v) {
        add(v, val_ptr[v]);
    }
    for(const auto& e : edges) {
        merge(std::get<0>(e), std::get<1>(e));
    }
    repair();
}


template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
cache_all_reps(Value epsilon)
{
    for(auto& v : cache_)
        v = static_cast<Vertex>(-1);

    for(Vertex u = 0; u < size(); ++u) {
        cache_[u] = simplification_repr(u, epsilon);
    }
}

template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
cache_all_reps(Value epsilon, Value level_value)
{
    for(auto& v : cache_)
        v = static_cast<Vertex>(-1);

    for(Vertex u = 0; u < size(); ++u) {
        cache_simplification_repr(u, epsilon, level_value);
        std::cerr << "LOOP u = " << u << ", cache_ = " << cache_[u] << std::endl;
    }
}


template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
cache_simplification_repr(Vertex u, Value epsilon, Value level_value)
{
    std::cerr << "enter simplification_repr, u  = " << u << ", epsilon = " << epsilon << ", level_value = " << level_value << std::endl;
    if (cache_[u] != static_cast<Vertex>(-1))
        return;

    Edge    sv = tree_[u];
    Vertex  s = sv.through, v = sv.to, result;

    std::vector<Vertex> intermediate;
    bool definitely_immovable = false;
    bool passed_movable = false;

    while(true) {
        bool crossed_level;
        if (negate())
            crossed_level = (function_[u] >= level_value) && (level_value >= function_[s]);
        else {
            crossed_level = (function_[u] <= level_value) && (level_value <= function_[s]);
        }

        passed_movable = passed_movable || (crossed_level && (fabs(function_[s] - function_[u]) < epsilon) && (u != v));

        // either persistent or root
        definitely_immovable = ((fabs(function_[s] - function_[u]) >= epsilon) || (u == v));
        if (definitely_immovable) {
            // if u was root, keep it
            if (intermediate.empty()) {
                intermediate.push_back(u);
            }
            break;
        } else {
            // store branch (u, s) in intermediate
            // and continue recursively with v
            intermediate.push_back(u);
            intermediate.push_back(s);

            u = v;
            sv = tree_[u];
            s = sv.through;
            v = sv.to;
        }
    }

    if (intermediate.empty())
        throw std::runtime_error("OHO");

    for(Vertex v : intermediate) {
        if (passed_movable) {
            // last saddle with maximal value will be at the end of
            // intermediate
            cache_[v] = intermediate.back();
        } else {
            // all vertices remain unchanged
            cache_[v] = v;
        }
        std::cerr << "INSIDE v = " << v << ", passed_movable = " << passed_movable << ", cache_v = " << cache_[v] << std::endl;
    }

    std::cerr << "EXIT simplification_repr, u  = " << u << ", epsilon = " << epsilon << ", level_value = " << level_value << std::endl;
}

template<class Value, class Vertex>
Vertex
nesoi::TripletMergeTree<Value, Vertex>::
simplification_repr(Vertex u, Value epsilon)
{
    if (cache_[u] != static_cast<Vertex>(-1))
        return cache_[u];

    Edge    sv = tree_[u];
    Vertex  s = sv.through, v = sv.to, result;

    if (u == v) {                                                // root
        result = u;
    } else if (fabs(function_[s] - function_[u]) >= epsilon) {   // persistent
        result = u;
    } else {
        Vertex vr = simplification_repr(v, epsilon);
        if (vr == v)                                             // terminal, use saddle
            result = s;
        else                                                     // otherwise use parent's rep
            result = vr;
    }
    cache_[u] = result;

    return result;
}


template<class Value, class Vertex>
typename nesoi::TripletMergeTree<Value, Vertex>::
Function
nesoi::TripletMergeTree<Value, Vertex>::
simplify(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* val_ptr, Value epsilon, bool negate)
{
    Function simplified = function_;
    compute_mt(edges, val_ptr, negate);
    cache_all_reps(epsilon);

    for_each_vertex([&simplified, this](Vertex u) {
        simplified[u] = this->function_[this->cache_[u]];
    });

    return simplified;
}

template<class Value, class Vertex>
typename nesoi::TripletMergeTree<Value, Vertex>::
Function
nesoi::TripletMergeTree<Value, Vertex>::
simplify(const std::vector<std::tuple<Vertex,Vertex>>& edges, Value* val_ptr, Value epsilon, Value level_value, bool negate)
{
    Function simplified = function_;
    compute_mt(edges, val_ptr, negate);
    cache_all_reps(epsilon, level_value);

    for_each_vertex([&simplified, this](Vertex u) {
        simplified[u] = this->function_[this->cache_[u]];
    });

    return simplified;
}


