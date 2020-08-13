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
compute_mt(const std::vector<std::tuple<Vertex,Vertex>>& edges, const int64_t* const labels, const Value* const val_ptr, bool negate)
{
    set_negate(negate);

    for(size_t v = 0; v < size(); ++v) {
        add(v, val_ptr[v]);
    }
    if (labels) {
        for(const auto& e : edges) {
            Vertex u = std::get<0>(e), v = std::get<1>(e);
            if (labels[u] == labels[v])
                merge(u, v);
        }
    } else {
        for(const auto& e : edges)
            merge(std::get<0>(e), std::get<1>(e));
    }
    repair();
}


template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
cache_all_reps(Value epsilon, bool squash_root)
{
    for(auto& v : cache_)
        v = static_cast<Vertex>(-1);

    for(Vertex u = 0; u < size(); ++u) {
        cache_[u] = simplification_repr(u, epsilon, squash_root);
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
    }
}


template<class Value, class Vertex>
void
nesoi::TripletMergeTree<Value, Vertex>::
cache_simplification_repr(Vertex u, Value epsilon, Value level_value)
{
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
        throw std::runtime_error("No intermediate nodes found");

    for(Vertex v : intermediate) {
        if (passed_movable) {
            // last saddle with maximal value will be at the end of
            // intermediate
            cache_[v] = intermediate.back();
        } else {
            // all vertices remain unchanged
            cache_[v] = v;
        }
    }
}


template<class Value, class Vertex>
Vertex
nesoi::TripletMergeTree<Value, Vertex>::
simplification_repr(Vertex u, Value epsilon, bool squash_root)
{
    if (cache_[u] != dummy_vertex())
        return cache_[u];

    Edge    sv = tree_[u];
    Vertex  s = sv.through, v = sv.to, result;

    if (u == v) {                                                // root
        if (squash_root && value(u) <= epsilon)
            result = dummy_vertex_2();
        else
            result = u;
    } else if (fabs(value(s) - value(u)) >= epsilon) {   // persistent
        result = u;
    } else {
        Vertex vr = simplification_repr(v, epsilon, squash_root);
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
simplify(const std::vector<std::tuple<Vertex,Vertex>>& edges, const int64_t* labels, const Value* const val_ptr, Value epsilon, bool negate, bool squash_root)
{

    if (squash_root && !negate) {
        std::cerr << "squash_root requires negate=True" << std::endl;
        throw std::runtime_error("squash_root requires negate=True");
    }

    set_negate(negate);

    Function simplified = function_;

    compute_mt(edges, labels, val_ptr, negate);

    cache_all_reps(epsilon, squash_root);

    if (squash_root)
        for_each_vertex([&simplified, this](Vertex u) {
            if (this->cache_[u] != dummy_vertex_2())
                simplified[u] = this->function_[this->cache_[u]];
            else
                simplified[u] = 0;
        });
    else
        for_each_vertex([&simplified, this](Vertex u) {
            simplified[u] = this->function_[this->cache_[u]];
        });

    return simplified;
}

template<class Value, class Vertex>
typename nesoi::TripletMergeTree<Value, Vertex>::
Function
nesoi::TripletMergeTree<Value, Vertex>::
simplify(const std::vector<std::tuple<Vertex,Vertex>>& edges, const Value* const val_ptr, Value epsilon, Value level_value, bool negate)
{
    set_negate(negate);

    Function simplified = function_;
    compute_mt(edges, nullptr, val_ptr, negate);
    cache_all_reps(epsilon, level_value);

    for_each_vertex([&simplified, this](Vertex u) {
        simplified[u] = this->function_[this->cache_[u]];
    });

    return simplified;
}

        //Diagram     diagram(const std::vector<std::tuple<Vertex,Vertex>>& edges, const int64_t* const labels, const Value* const values, bool negate);
        //Diagram     noisy_part_of_diagram(const std::vector<std::tuple<Vertex,Vertex>>& edges, const int64_t* const labels, const Value* const values, Value epsilon, bool negate);

template<class Value, class Vertex>
typename nesoi::TripletMergeTree<Value, Vertex>::
Diagram
nesoi::TripletMergeTree<Value, Vertex>::
diagram(const std::vector<std::tuple<Vertex,Vertex>>& edges, const int64_t* const labels, const Value* const val_ptr, bool negate, bool squash_root)
{
    if (squash_root && !negate) {
        throw std::runtime_error("negate=false and squash_root=true");
    }

    set_negate(negate);
    compute_mt(edges, labels, val_ptr, negate);

    Diagram diagram;

    Value root_death = 0;
    if (!squash_root) {
        root_death = negate ?  -std::numeric_limits<Value>::infinity() : std::numeric_limits<Value>::infinity();
    }

    traverse_persistence(
            [this, &diagram, squash_root, negate, root_death](Vertex u, Vertex s, Vertex v)
            {
                Value birth = this->value(u);
                Value death = (u == s) ? root_death : this->value(s);
                diagram.emplace_back(birth, death);
            });

   return diagram;
}


template<class Value, class Vertex>
typename nesoi::TripletMergeTree<Value, Vertex>::
Pairings
nesoi::TripletMergeTree<Value, Vertex>::
pairings(const std::vector<std::tuple<Vertex,Vertex>>& edges,
                      const int64_t* const labels,
                      const Value* const val_ptr,
                      bool negate,
                      bool squash_root,
                      Value epsilon)
{
    if (squash_root && !negate) {
        throw std::runtime_error("negate=false and squash_root=true");
    }

    set_negate(negate);
    compute_mt(edges, labels, val_ptr, negate);

    IndexDiagram noisy_pairs, important_pairs;
    IndexArray noisy_essential, essential;

    Value root_death = 0;
    if (!squash_root) {
        root_death = negate ?  -std::numeric_limits<Value>::infinity() : std::numeric_limits<Value>::infinity();
    }

    traverse_persistence(
            [this, &noisy_pairs, &important_pairs, &noisy_essential, &essential, root_death, epsilon, squash_root](Vertex u, Vertex s, Vertex v)
            {
                Value birth = this->value(u);
                Value death = (u == s) ? root_death : this->value(s);

                if (u == s) {
                    // root
                    if (squash_root && (fabs(birth-root_death) < epsilon))
                        noisy_essential.emplace_back(u);
                    else
                        essential.emplace_back(u);
                } else {
                    if (fabs(birth - death) < epsilon)
                        noisy_pairs.emplace_back(u, s);
                    else
                        important_pairs.emplace_back(u, s);
                }
            });

   return std::make_tuple(noisy_pairs, important_pairs, noisy_essential, essential);
}

template<class Value, class Vertex>
size_t
nesoi::TripletMergeTree<Value, Vertex>::
n_components(const std::vector<std::tuple<Vertex,Vertex>>& edges, const int64_t* const labels)
{
    std::vector<Value> values(size(), Value(0));

    compute_mt(edges, labels, &values[0], false);

    size_t result = 0;

    traverse_persistence(
            [this, &result](Vertex u, Vertex s, Vertex v)
            {
                result += (u == s && s == v);
            });

   return result;
}
