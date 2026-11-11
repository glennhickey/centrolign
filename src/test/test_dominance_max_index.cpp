#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <limits>
#include <stdexcept>

#include "centrolign/dominance_max_index.hpp"

using namespace std;
using namespace centrolign;

// checks every query against a brute force scan of the points' current values
template<typename Key, typename Value, size_t NumVal, typename UIntSize>
void check_query(const DominanceMaxIndex<Key, Value, NumVal, UIntSize>& index,
                 const vector<Key>& keys, const vector<array<Value, NumVal>>& values,
                 size_t prefix, Key bound) {

    array<Value, NumVal> max_values;
    array<UIntSize, NumVal> max_ranks;
    index.prefix_max(prefix, bound, max_values, max_ranks);

    for (size_t s = 0; s < NumVal; ++s) {
        Value expected = numeric_limits<Value>::lowest();
        for (size_t r = 0; r < prefix; ++r) {
            if (keys[r] < bound) {
                expected = max(expected, values[r][s]);
            }
        }
        bool correct = (max_values[s] == expected);
        if (expected == numeric_limits<Value>::lowest()) {
            correct = correct && max_ranks[s] == keys.size();
        }
        else {
            correct = correct && max_ranks[s] < prefix && keys[max_ranks[s]] < bound && values[max_ranks[s]][s] == expected;
        }
        if (!correct) {
            cerr << "failed query with prefix " << prefix << ", bound " << bound << ", slot " << s << ": got value " << max_values[s] << " at rank " << max_ranks[s] << ", expected " << expected << '\n';
            cerr << "points:\n";
            for (size_t r = 0; r < keys.size(); ++r) {
                cerr << r << '\t' << keys[r];
                for (auto v : values[r]) {
                    cerr << '\t' << v;
                }
                cerr << '\n';
            }
            exit(1);
        }
    }
}

// interleaves random updates with random queries, including bounds and prefixes past the ends
template<typename Key, typename Value, size_t NumVal, typename UIntSize, class Generator>
void random_test(size_t size, Key min_key, Key max_key, Generator& gen) {

    uniform_int_distribution<Key> key_distr(min_key, max_key);
    uniform_int_distribution<int> val_distr(-20, 20);
    uniform_int_distribution<int> op_distr(0, 2);
    uniform_int_distribution<size_t> rank_distr(0, size == 0 ? 0 : size - 1);
    uniform_int_distribution<size_t> prefix_distr(0, size);
    uniform_int_distribution<Key> bound_distr(min_key, max_key + 1);

    vector<Key> keys(size);
    for (auto& key : keys) {
        key = key_distr(gen);
    }
    vector<array<Value, NumVal>> values(size);
    for (auto& vals : values) {
        vals.fill(numeric_limits<Value>::lowest());
    }

    DominanceMaxIndex<Key, Value, NumVal, UIntSize> index(keys);
    if (index.size() != size) {
        cerr << "index has size " << index.size() << ", expected " << size << '\n';
        exit(1);
    }

    for (size_t step = 0; step < 4 * size + 10; ++step) {
        if (size != 0 && op_distr(gen) == 0) {
            size_t rank = rank_distr(gen);
            array<Value, NumVal> update;
            for (size_t s = 0; s < NumVal; ++s) {
                update[s] = val_distr(gen);
                values[rank][s] = max(values[rank][s], update[s]);
            }
            index.update(rank, update);
        }
        else {
            check_query(index, keys, values, prefix_distr(gen), bound_distr(gen));
        }
    }
}

int main(int argc, char* argv[]) {

    {
        // a small example worked out by hand
        vector<int> keys{3, 1, 4, 1, 5};
        DominanceMaxIndex<int, double, 2, uint32_t> index(keys);
        index.update(0, {1.0, 7.0});
        index.update(1, {2.0, -1.0});
        index.update(3, {0.5, 3.0});
        index.update(4, {9.0, 9.0});
        index.update(1, {1.5, 4.0}); // only raises the second value

        array<double, 2> max_values;
        array<uint32_t, 2> max_ranks;
        index.prefix_max(4, 4, max_values, max_ranks);
        if (max_values != array<double, 2>{2.0, 7.0} || max_ranks != array<uint32_t, 2>{1, 0}) {
            cerr << "failed hand-worked query 1\n";
            exit(1);
        }
        index.prefix_max(4, 3, max_values, max_ranks);
        if (max_values != array<double, 2>{2.0, 4.0} || max_ranks != array<uint32_t, 2>{1, 1}) {
            cerr << "failed hand-worked query 2\n";
            exit(1);
        }
        index.prefix_max(5, 1, max_values, max_ranks);
        if (max_values[0] != numeric_limits<double>::lowest() || max_ranks != array<uint32_t, 2>{5, 5}) {
            cerr << "failed hand-worked query 3\n";
            exit(1);
        }
    }

    {
        // too many points for the rank type
        vector<int> keys(255, 0);
        bool threw = false;
        try {
            DominanceMaxIndex<int, double, 1, uint8_t> index(keys);
        }
        catch (const runtime_error&) {
            threw = true;
        }
        if (!threw) {
            cerr << "did not reject more points than the rank type can hold\n";
            exit(1);
        }
    }

    random_device rd;
    default_random_engine gen(rd());

    uniform_int_distribution<size_t> small_size_distr(0, 40);
    uniform_int_distribution<size_t> large_size_distr(0, 600);
    for (size_t rep = 0; rep < 2000; ++rep) {
        size_t size = rep % 4 == 0 ? large_size_distr(gen) : small_size_distr(gen);
        // few distinct keys, so that many points tie
        random_test<uint32_t, float, 3, uint32_t>(size, 0, 10, gen);
        // many distinct keys, some negative
        random_test<int64_t, double, 1, uint16_t>(size, -1000, 1000, gen);
    }

    cerr << "passed all tests!" << endl;
}
