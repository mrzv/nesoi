#include <iostream>
#include <nesoi/triplet-merge-tree.h>

using TripletMergeTree  = nesoi::TripletMergeTree<int>;
using Vertex            = TripletMergeTree::Vertex;

int main()
{
    TripletMergeTree tmt(10, false);

    for(int i = 0; i < 5; ++i) {
        tmt.add(i, i);
    }

    for(int i = 5; i < tmt.size(); ++i) {
        tmt.add(i, 4 - 3 *i);
    }

    for(int i = 0; i < tmt.size() - 1; ++i) {
        tmt.merge(i, i+1);
    }

//    tmt.add(0,2);
//    tmt.add(1,4);
//    tmt.add(2,1);
//    tmt.add(3,5);
//    tmt.add(4,0);
//
//    tmt.merge(0,1);
//    tmt.merge(1,2);
//    tmt.merge(2,3);
//    tmt.merge(3,4);

    tmt.repair();
    tmt.traverse_persistence([](Vertex u, Vertex s, Vertex v) { std::cout << u << ' ' << s << ' ' << v << std::endl; });

    std::cout << "----" << std::endl;

    for (Vertex u = 0; u < tmt.size(); ++u)
    {
        auto e = tmt[u];
        std::cout << u << " -> " << e.through << ' ' << e.to << std::endl;
    }

    for (Vertex u = 1; u < tmt.size()-1; ++u)
    {
        auto rep_m = tmt.representative(u, u-1);
        auto rep_p = tmt.representative(u, u+1);
        std::cout << u << " at " << u -1 << ": " << rep_m << "; at " << u + 1 << ": " << rep_p << std::endl;
    }
}
