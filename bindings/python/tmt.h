#pragma once

#include <nesoi/triplet-merge-tree.h>

using PyTMT  = nesoi::TripletMergeTree<std::uint32_t, std::uint32_t>;
using Vertex = PyTMT::Vertex;
using Degree = PyTMT::Value;
