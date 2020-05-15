#include <queue>
#include <stack>

#if !defined(NESOI_NO_PARALLEL)
#include <thread>
#include <future>
#endif

#include <iostream>

template<class T>
nesoi::KDTree<T>::
KDTree(const Traits& traits, HandleContainer&& handles):
    traits_(traits), tree_(std::move(handles))
{ init(); }

template<class T>
template<class Range>
nesoi::KDTree<T>::
KDTree(const Traits& traits, const Range& range):
    traits_(traits)
{
    init(range);
}

template<class T>
template<class Range>
void
nesoi::KDTree<T>::
init(const Range& range)
{
    tree_.reserve(std::distance(std::begin(range), std::end(range)));
    for (PointHandle h : range)
        tree_.push_back(h);
    init();
}

template<class T>
void
nesoi::KDTree<T>::
init()
{
    if (tree_.empty())
        return;

    HCIterator  b = tree_.begin(),
                e = tree_.end();
    size_t      i = 0;

#if defined(NESOI_NO_PARALLEL)
    sort_all(b,e,i);
#else
    unsigned threads = std::thread::hardware_concurrency();
    std::cerr << "Building k-d tree using " << threads << " threads" << std::endl;
    sort_all_threads(b,e,i,threads);
#endif
}

#if !defined(NESOI_NO_PARALLEL)
template<class T>
void
nesoi::KDTree<T>::
sort_all_threads(HCIterator b, HCIterator e, size_t i, unsigned threads)
{
    if (threads == 1)
    {
        sort_all(b,e,i);
        return;
    }

    HCIterator m = sort(b,e,i);

    size_t next_i = (i + 1) % traits().dimension();

    std::vector<std::future<void>> handles;
    if (b < m - 1)
        handles.emplace_back(std::async(std::launch::async,
                                        [this,b,m,e,next_i,threads]()
                                        {
                                            sort_all_threads(b,m,next_i,threads/2);
                                        }));
    if (e - m > 2)
        handles.emplace_back(std::async(std::launch::async,
                                        [this,b,m,e,next_i,threads]()
                                        {
                                            sort_all_threads(m+1,e,next_i,threads - threads/2);
                                        }));
}
#endif


template<class T>
void
nesoi::KDTree<T>::
sort_all(HCIterator b, HCIterator e, size_t i)
{
    std::queue<KDTreeNode> q;
    q.push(KDTreeNode(b,e,i));
    while (!q.empty())
    {
        HCIterator b, e; size_t i;
        std::tie(b,e,i) = q.front();
        q.pop();

        HCIterator m = sort(b,e,i);

        size_t next_i = (i + 1) % traits().dimension();

        // Replace with a size condition instead?
        if (b < m - 1)  q.push(KDTreeNode(b,   m, next_i));
        if (e - m > 2)  q.push(KDTreeNode(m+1, e, next_i));
    }
}

template<class T>
typename nesoi::KDTree<T>::HCIterator
nesoi::KDTree<T>::
sort(HCIterator b, HCIterator e, size_t i)
{
    HCIterator m = b + (e - b)/2;
    CoordinateComparison cmp(i, traits());
    std::nth_element(b,m,e, cmp);
    return m;
}

template<class T>
template<class ResultsFunctor>
void
nesoi::KDTree<T>::
search(PointHandle q, ResultsFunctor& rf) const
{
    typedef         typename HandleContainer::const_iterator        HCIterator;
    typedef         std::tuple<HCIterator, HCIterator, size_t>      KDTreeNode;

    if (tree_.empty())
        return;

    DistanceType    D  = std::numeric_limits<DistanceType>::infinity();

    std::queue<KDTreeNode>  nodes;
    nodes.push(KDTreeNode(tree_.begin(), tree_.end(), 0));

    while (!nodes.empty())
    {
        HCIterator b, e; size_t i;
        std::tie(b,e,i) = nodes.front();
        nodes.pop();

        CoordinateComparison cmp(i, traits());
        i = (i + 1) % traits().dimension();

        HCIterator m = b + (e - b)/2;
        DistanceType dist = traits().distance(q, *m);
        D = rf(*m, dist);

        // we are really searching w.r.t L_\infty ball; could prune better with an L_2 ball
        Coordinate diff = cmp.diff(q, *m);     // diff returns signed distance
        if (e > m + 1 && diff >= -D)
            nodes.push(KDTreeNode(m+1, e, i));

        if (b < m && diff <= D)
            nodes.push(KDTreeNode(b,   m, i));
    }
}

template<class T>
typename nesoi::KDTree<T>::HandleDistance
nesoi::KDTree<T>::
find(PointHandle q) const
{
    nesoi::NNRecord<HandleDistance> nn;
    search(q, nn);
    return nn.result;
}

template<class T>
typename nesoi::KDTree<T>::Result
nesoi::KDTree<T>::
findR(PointHandle q, DistanceType r) const
{
    nesoi::rNNRecord<HandleDistance> rnn(r);
    search(q, rnn);
    std::sort(rnn.result.begin(), rnn.result.end());
    return rnn.result;
}

template<class T>
typename nesoi::KDTree<T>::Result
nesoi::KDTree<T>::
findK(PointHandle q, size_t k) const
{
    nesoi::kNNRecord<HandleDistance> knn(k);
    search(q, knn);
    std::sort(knn.result.begin(), knn.result.end());
    return knn.result;
}


template<class T>
struct nesoi::KDTree<T>::CoordinateComparison
{
                CoordinateComparison(size_t i, const Traits& traits):
                    i_(i), traits_(traits)                              {}

    bool        operator()(PointHandle p1, PointHandle p2) const        { return coordinate(p1) < coordinate(p2); }
    Coordinate  diff(PointHandle p1, PointHandle p2) const              { return coordinate(p1) - coordinate(p2); }

    Coordinate  coordinate(PointHandle p) const                         { return traits_.coordinate(p, i_); }
    size_t      axis() const                                            { return i_; }

    private:
        size_t          i_;
        const Traits&   traits_;
};
