#include "../include/opts.hpp"

#include <cstdio>
#include <cstdlib>
#include <cxxopts.hpp>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace viewer {
namespace internal {
void add_common_opts(cxxopts::Options& options) {
    // clang-format off
    // @formatter:off

    options.add_options()
    ("file", "npz file storing octree data", cxxopts::value<std::string>())
    ("bg", "background brightness 0-1", cxxopts::value<float>()->default_value("0.0"))
    ("s,step_size", "step size epsilon added to computed cube size", cxxopts::value<float>()->default_value("1e-4"))
    ("e,stop_thresh", "early stopping threshold (on remaining intensity)", cxxopts::value<float>()->default_value("1e-2"))
    ("a,sigma_thresh", "sigma threshold (skip cells with < sigma)", cxxopts::value<float>()->default_value("1e-2"))
    ("model_path", "model path", cxxopts::value<std::string>()->default_value(""))
    ("c,max_tree_capacity", "max capacity of octree", cxxopts::value<size_t>()->default_value("20000000"))
    ("x,split_batch_size", "max number of splits performed per batch", cxxopts::value<int>()->default_value("4096"))
    ("n,nerf_batch_size", "max number of nerf evals performed per batch", cxxopts::value<int>()->default_value("4096"))
    ("v,samples_per_voxel", "number of guided_samples per voxel", cxxopts::value<int>()->default_value("8"))
    ("b,bounds_only", "only load bounds and scale")
    ("y,appearance_embedding", "appearance embedding to use", cxxopts::value<int>()->default_value("-1"))
    ("z,max_guided_samples", "max guided_samples to use per ray", cxxopts::value<int>()->default_value("128"))

    ("help", "Print this help message");
    // @formatter:on
    // clang-format on
}

cxxopts::ParseResult parse_options(cxxopts::Options& options,
                                   int argc,
                                   char* argv[]) {
    options.parse_positional({"file"});
    cxxopts::ParseResult args = options.parse(argc, argv);
    if (args.count("help")) {
        printf("%s\n", options.help().c_str());
        std::exit(0);
    }
    return args;
}

viewer::RenderOptions render_options_from_args(cxxopts::ParseResult& args) {
    viewer::RenderOptions options;

    options.background_brightness = args["bg"].as<float>();
    if (args.count("grid")) {
        options.show_grid = true;
        options.grid_max_depth = args["grid"].as<int>();
    }

    options.step_size = args["step_size"].as<float>();
    options.stop_thresh = args["stop_thresh"].as<float>();
    options.sigma_thresh = args["sigma_thresh"].as<float>();
    options.split_batch_size = args["split_batch_size"].as<int>();
    options.nerf_batch_size = args["nerf_batch_size"].as<int>();
    options.samples_per_corner = args["samples_per_voxel"].as<int>();
    options.appearance_embedding = args["appearance_embedding"].as<int>();
    options.max_guided_samples = args["max_guided_samples"].as<int>();

    return options;
}

}  // namespace internal
}  // namespace viewer