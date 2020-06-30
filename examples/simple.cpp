#include <iostream>
#include <nesoi/triplet-merge-tree.h>

using TripletMergeTree  = nesoi::TripletMergeTree<int>;
using Vertex            = TripletMergeTree::Vertex;

int main()
{
    TripletMergeTree tmt(5, true);

    tmt.add(0,2);
    tmt.add(1,4);
    tmt.add(2,1);
    tmt.add(3,5);
    tmt.add(4,0);

    tmt.merge(0,1);
    tmt.merge(1,2);
    tmt.merge(2,3);
    tmt.merge(3,4);

    tmt.repair();
    tmt.traverse_persistence([](Vertex u, Vertex s, Vertex v) { std::cout << u << ' ' << s << ' ' << v << std::endl; });

    std::cout << "----" << std::endl;

    for (Vertex u = 0; u < tmt.size(); ++u)
    {
        auto e = tmt[u];
        std::cout << u << ' ' << e.through << ' ' << e.to << std::endl;
    }
}
