/*
 * precision_test.cpp

 *
 *  Created on: Jul 13, 2016
 *      Author: Claudio Sanhueza
 *      Contact: csanhuezalobos@gmail.com
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <map>
#include <random>
#include <limits> // for std::numeric_limits

#include "../../../scikitplot/cexternals/_annoy/src/kissrandom.h"
#include "../../../scikitplot/cexternals/_annoy/src/annoylib.h"

using namespace Annoy;
int precision(int f=40, int n=1000000, int seed=0){
	// Declare once at the top of precision()
	// std::default_random_engine generator;
	// Seed C++ RNG
	// std::default_random_engine generator(std::random_device{}());
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);
	std::uniform_int_distribution<int> distr_node(0, n-1); // for selecting random nodes

    //=========================================================
    // Build index AnnoyIndexSingleThreadedBuildPolicy, AnnoyIndexMultiThreadedBuildPolicy
    //=========================================================
	// Use deterministic single-threaded Annoy (bit reproducible)

	// copy/move initialization:  "-std=c++14",  # non-copyable (because of std::atomic)
	// T t = T(args);
	// Index t = Index(f);      // can trigger copy/move
	// auto t = Index(f);       // same issue
	// AnnoyIndex<int, double, Angular, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy> t =
	// auto t =
	// 	AnnoyIndex<int, double, Angular, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy>(f);

	// direct initialization:
	// using Index = AnnoyIndex<int, double, Angular, Kiss32Random, AnnoyIndexSingleThreadedBuildPolicy>;
	// Index t(f);     // C++14 OK
	AnnoyIndex<int, double, Angular, Kiss32Random, AnnoyIndexMultiThreadedBuildPolicy> t{f};

	// Seed Annoy internal RNG
	t.set_seed(seed);

	std::cout << "Building index ... be patient !!" << std::endl;
	std::cout << "\"Trees that are slow to grow bear the best fruit\" (Moliere)" << std::endl;

	for (int i = 0; i < n; ++i) {
		// double *vec = (double *) malloc( f * sizeof(double) );
		// for(int z=0; z<f; ++z){
		// 	vec[z] = (distribution(generator));
		// }
		// t.add_item(i, vec);

		// Avoid memory leaks, Faster allocation than calling malloc millions of times.
		// vec.data() returns double* pointing to the internal contiguous storage of the vector.
		std::vector<double> vec(f);
		std::generate(vec.begin(), vec.end(), [&]() { return distribution(generator); });
		t.add_item(i, vec.data());  // <- Pass pointer to contiguous storage

		std::cout << "Loading objects ...\t object: "<< (i + 1)
		          << "\tProgress:"<< std::fixed << std::setprecision(2)
				  << (double) i / (double)(n + 1) * 100.0 << "%"
				  << "\r" << std::flush;

	}
	std::cout << std::endl;
	std::cout << "Building index num_trees = 2 * num_features ...";

	// std::chrono::high_resolution_clock::time_point t_start, t_end;
	// t_start = std::chrono::high_resolution_clock::now();
	// ...
	// t_end = std::chrono::high_resolution_clock::now();
    using clock = std::chrono::high_resolution_clock;

    auto t_start = clock::now();
	t.build(2 * f);
    auto t_end = clock::now();
	// build_secs
	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t_end - t_start ).count();

	std::cout << " Done in " << duration << " secs." << std::endl;
	std::cout << "Saving index ...";
	t.save("precision.tree");
	std::cout << " Done" << std::endl;

    //=========================================================
    // Precision testing
    //=========================================================

	int K=10;
	int prec_n = 1000;
	std::vector<int> limits = {10, 100, 1000, 10000};
	std::map<int, double> prec_sum;
	std::map<int, double> time_sum;
	//init precision and timers map
	for(std::vector<int>::iterator it = limits.begin(); it!=limits.end(); ++it){
		prec_sum[(*it)] = 0.0;
		time_sum[(*it)] = 0.0;
	}

	std::vector<int> closest;
	// doing the work
	for(int i=0; i < prec_n; ++i){
        // progress bar every 10%
		// int division will drop the remainder automatically — just like prec_n // 10 in Python.
		// if ((i+1) % (prec_n / 2) == 0 || i == prec_n-1) {
		// 	// std::cout << std::endl;
		// 	std::cout << "\rProgress: " << (i+1) << "/" << prec_n << " ("
		// 		<< std::fixed << std::setprecision(1)
		// 		// << (100.0 * (i+1) / prec_n) << "%)"
		// 		<< (double)(i+1)/prec_n*100 << "%)"
		// 		<< "\r" << std::flush;
		// 		// << std::endl;
		// }

		//select a random node
		// int j = rand() % n;
		int j = distr_node(generator);  // no redeclaration
		std::cout << "\nFinding nbs for " << j << std::endl;

        // ground-truth nearest neighbors, getting the K closest
		t.get_nns_by_item(j, K, n, &closest, nullptr);

		std::vector<int> toplist;
		std::vector<int> intersection;

        // for (auto limit : limits) {
		for(std::vector<int>::iterator limit = limits.begin(); limit!=limits.end(); ++limit){

			// t_start = std::chrono::high_resolution_clock::now();
            t_start = clock::now();
			// (size_t)-1         // bit trick, legacy style
			// std::numeric_limits<size_t>::max()  // explicit, readable
			// (size_t)-1  = 0xFFFFFFFF = 4294967295 = 2^N - 1 = 2^32 - 1
			// (size_t)-1  = 0xFFFFFFFFFFFFFFFF = 18446744073709551615 = 2^N - 1 = 2^64 - 1
			// t.get_nns_by_item(j, (*limit), (size_t) -1, &toplist, nullptr); //search_k defaults to "n_trees * n" if not provided.
			// t.get_nns_by_item(j, (*limit), std::numeric_limits<size_t>::max(), &toplist, nullptr);
			t.get_nns_by_item(j, (*limit), -1, &toplist, nullptr);
			// t_end = std::chrono::high_resolution_clock::now();
            t_end = clock::now();
			// long long ms =
			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
				t_end - t_start
			).count();

			//intersecting results
            // std::sort(closest.begin(), closest.end());
            // std::sort(toplist.begin(), toplist.end());
            // intersection.resize(std::min(closest.size(), toplist.size()));

			std::sort(closest.begin(), closest.end(), std::less<int>());
			std::sort(toplist.begin(), toplist.end(), std::less<int>());
			intersection.resize(std::max(closest.size(), toplist.size()));

			// std::vector<int>::iterator it_set =
			auto it_set = std::set_intersection(
				closest.begin(), closest.end(),
				toplist.begin(), toplist.end(),
				intersection.begin()
			);
			intersection.resize(it_set - intersection.begin());

			// storing metrics
			// int found = intersection.size();
			// double hitrate = found / (double) K;
			double hitrate = double(intersection.size()) / double(K);

			prec_sum[(*limit)] += hitrate;
			time_sum[(*limit)] += duration;

			//deallocate memory
			// vector<int>().swap(intersection);
			// vector<int>().swap(toplist);
			toplist.clear();
			intersection.clear();
		}
		// print resulting metrics
        // for (auto limit : limits) {
		for(std::vector<int>::iterator limit = limits.begin(); limit!=limits.end(); ++limit){
			// multiply by 1e-04 to convert milliseconds to seconds (approx).
			// You could instead divide by 1000.0 for clarity:
			// << (time_sum[(*limit)] / (i + 1)) / 1000.0
			std::cout << "limit: " << (*limit) << "\tprecision: "
			          << std::fixed << std::setprecision(2)
					  << (100.0 * prec_sum[(*limit)] / (i + 1))
					  << "% \tavg. time: "
					  << std::fixed << std::setprecision(6)
					  << (time_sum[(*limit)] / (i + 1)) * 1e-04
					  << "s" << std::endl;
		}
		// vector<int>().swap(closest);
		closest.clear();

	}
	// std::cout << std::endl;  // flush after loop
	std::cout << "\nDone" << std::endl;
	return 0;
}

//=========================================================
// CLI helper functions
//=========================================================

void help(){
    std::cout << "Annoy Precision C++ example\n"
              << "\"f: num_features, n: num_nodes\"\n"
              << "Usage:\n"
              << "  ./precision            → defaults\n"
              << "  ./precision f          → set f\n"
              << "  ./precision f n        → set f, n\n"
              << "  ./precision f n seed   → set f, n, seed\n"
	          << std::endl;
}

void feedback(int f, int n, int seed){
    std::cout << "Runing precision example with:\n"
              << "  seed         : " << seed << "\n"
              << "  num. features: " << f << "\n"
			  << "  num. nodes   : " << n
	          << std::endl;
}

int main(int argc, char **argv) {
	// constexpr int RNG_SEED = 0;   // deterministic default seed
	int f, n, seed;
	f = 40;
	// n = 1'000'000;  // valid in C++14 and later
	n = 100000;
	seed = 0;

    if (argc == 1) {
        // No arguments → run with defaults
        feedback(f, n, seed);
        precision(f, n, seed);
        return EXIT_SUCCESS;
    }
    // Expect: ./precision f n seed
    if (argc >= 2) {
        f = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        n = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        seed = std::atoi(argv[3]);
    }
    if (argc > 4) {
        help();
		feedback(f, n, seed);
        return EXIT_FAILURE;
    }

    feedback(f, n, seed);
    precision(f, n, seed);
    return EXIT_SUCCESS;
}
