#pragma once

#if !defined(NESOI_NO_PARALLEL)
#include <vector>
#include <thread>
#include <future>
#endif

namespace nesoi
{

template<class T, class F>
void for_each(T n, const F& f)
{
#if defined(NESOI_NO_PARALLEL)
    for (T u = 0; u < n; ++u)
        f(u);
#else
    unsigned threads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> handles;
    for (unsigned i = 0; i < threads; ++i)
    {
        handles.emplace_back(std::async(std::launch::async,
                                        [i,threads,n,&f]()
                                        {
                                            size_t chunk = n / threads;
                                            T b = chunk*i,
                                              e = (i == threads - 1 ? n : chunk*(i+1));
                                            for (T u = b; u < e; ++u)
                                                f(u);
                                        }));
    }

    // handles' destructors make sure that everything runs
#endif
}

}
