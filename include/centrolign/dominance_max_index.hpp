#ifndef centrolign_dominance_max_index_hpp
#define centrolign_dominance_max_index_hpp

#include <vector>
#include <array>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "centrolign/utility.hpp"

namespace centrolign {

/*
 * A static set of points, each with a rank (its position in the order the points were given in)
 * and a key, and each holding an array of values that can only increase. Answers dominance max
 * queries: for each slot of the value array, the max over the points whose rank is below a prefix
 * length and whose key is below a bound, along with the rank of a point that has it.
 *
 * Implemented as a Fenwick tree over rank, each of whose nodes holds a Fenwick tree over the keys
 * of the points it covers. Queries and updates take O(log^2 n) time, and the structure takes
 * O(n log n) memory, about (n log n) / 2 entries.
 */
template<typename Key, typename Value, size_t NumVal, typename UIntSize = uint32_t>
class DominanceMaxIndex {
public:

    // keys[i] is the key of the point with rank i
    DominanceMaxIndex(const std::vector<Key>& keys);
    DominanceMaxIndex() = default;
    DominanceMaxIndex(const DominanceMaxIndex& other) = default;
    DominanceMaxIndex(DominanceMaxIndex&& other) = default;
    ~DominanceMaxIndex() = default;
    DominanceMaxIndex& operator=(const DominanceMaxIndex& other) = default;
    DominanceMaxIndex& operator=(DominanceMaxIndex&& other) = default;

    // raise each value of the point at this rank to the given one, if it is greater
    inline void update(UIntSize rank, const std::array<Value, NumVal>& values);

    // for each value slot, the max value among the points with rank < prefix and key < bound, and the
    // rank of a point that has it. slots with no such point (or where all of them still have the initial
    // value) get the lowest value and the rank size()
    inline void prefix_max(size_t prefix, const Key& bound, std::array<Value, NumVal>& max_values,
                           std::array<UIntSize, NumVal>& max_ranks) const;

    inline size_t size() const;
    size_t memory_size() const;

private:

    // a point's position in an outer node, ordered by key and then rank
    struct Entry {
        Key key;
        UIntSize rank;
        inline bool operator<(const Entry& other) const {
            return key < other.key || (key == other.key && rank < other.rank);
        }
    };

    // an inner Fenwick tree node
    struct Node {
        std::array<Value, NumVal> value;
        std::array<UIntSize, NumVal> rank;
    };

    static inline size_t lowbit(size_t i);
    // where the inner arrays of outer node i (1-based) begin
    static inline size_t offset(size_t i);

    // the key of each point, by rank
    std::vector<Key> point_key;
    // the entries of each outer node, contiguously by node
    std::vector<Entry> entries;
    // the inner Fenwick trees, parallel to the entries
    std::vector<Node> nodes;

    static_assert(std::is_integral<UIntSize>::value && std::is_unsigned<UIntSize>::value,
                  "UIntSize must be an unsigned integer");
};




/*
 * Template implementations
 */

template<typename Key, typename Value, size_t NumVal, typename UIntSize>
inline size_t DominanceMaxIndex<Key, Value, NumVal, UIntSize>::lowbit(size_t i) {
    return i & (~i + 1);
}

template<typename Key, typename Value, size_t NumVal, typename UIntSize>
inline size_t DominanceMaxIndex<Key, Value, NumVal, UIntSize>::offset(size_t i) {
    // the sum of lowbit(j) for j < i, which works out to m + sum_{k >= 1} 2^(k - 1) floor(m / 2^k) for m = i - 1
    size_t m = i - 1;
    size_t total = m;
    for (size_t k = 1; (m >> k) != 0; ++k) {
        total += (m >> k) << (k - 1);
    }
    return total;
}

template<typename Key, typename Value, size_t NumVal, typename UIntSize>
DominanceMaxIndex<Key, Value, NumVal, UIntSize>::DominanceMaxIndex(const std::vector<Key>& keys) : point_key(keys) {

    size_t n = keys.size();
    if (n >= size_t(std::numeric_limits<UIntSize>::max())) {
        throw std::runtime_error("DominanceMaxIndex integer size is too small to represent its ranks.");
    }
    if (n == 0) {
        return;
    }

    size_t total = offset(n + 1);
    entries.resize(total);
    Node empty;
    empty.value.fill(std::numeric_limits<Value>::lowest());
    empty.rank.fill(UIntSize(n));
    nodes.resize(total, empty);

    // add the points in key order so that each outer node's entries come out sorted
    std::vector<UIntSize> order(n);
    for (size_t r = 0; r < n; ++r) {
        order[r] = r;
    }
    std::stable_sort(order.begin(), order.end(), [&](UIntSize a, UIntSize b) {
        return keys[a] < keys[b];
    });
    std::vector<size_t> cursor(n + 1);
    for (size_t i = 1, off = 0; i <= n; off += lowbit(i), ++i) {
        cursor[i] = off;
    }
    for (UIntSize r : order) {
        for (size_t i = size_t(r) + 1; i <= n; i += lowbit(i)) {
            auto& entry = entries[cursor[i]++];
            entry.key = keys[r];
            entry.rank = r;
        }
    }
}

template<typename Key, typename Value, size_t NumVal, typename UIntSize>
inline void DominanceMaxIndex<Key, Value, NumVal, UIntSize>::update(UIntSize rank, const std::array<Value, NumVal>& values) {

    Entry target;
    target.key = point_key[rank];
    target.rank = rank;

    size_t n = point_key.size();
    for (size_t i = size_t(rank) + 1; i <= n; i += lowbit(i)) {
        size_t begin = offset(i);
        size_t m = lowbit(i);
        // the point's (1-based) position among this outer node's entries
        size_t pos = std::lower_bound(entries.begin() + begin, entries.begin() + begin + m, target) - (entries.begin() + begin) + 1;
        for (size_t j = pos; j <= m; j += lowbit(j)) {
            auto& node = nodes[begin + j - 1];
            for (size_t s = 0; s < NumVal; ++s) {
                if (values[s] > node.value[s]) {
                    node.value[s] = values[s];
                    node.rank[s] = rank;
                }
            }
        }
    }
}

template<typename Key, typename Value, size_t NumVal, typename UIntSize>
inline void DominanceMaxIndex<Key, Value, NumVal, UIntSize>::prefix_max(size_t prefix, const Key& bound,
                                                                         std::array<Value, NumVal>& max_values,
                                                                         std::array<UIntSize, NumVal>& max_ranks) const {

    max_values.fill(std::numeric_limits<Value>::lowest());
    max_ranks.fill(UIntSize(point_key.size()));

    for (size_t i = prefix; i > 0; i -= lowbit(i)) {
        size_t begin = offset(i);
        size_t m = lowbit(i);
        // the number of this outer node's entries with key < bound
        auto first = entries.begin() + begin;
        size_t len = std::partition_point(first, first + m, [&](const Entry& entry) { return entry.key < bound; }) - first;
        for (size_t j = len; j > 0; j -= lowbit(j)) {
            const auto& node = nodes[begin + j - 1];
            for (size_t s = 0; s < NumVal; ++s) {
                if (node.value[s] > max_values[s]) {
                    max_values[s] = node.value[s];
                    max_ranks[s] = node.rank[s];
                }
            }
        }
    }
}

template<typename Key, typename Value, size_t NumVal, typename UIntSize>
inline size_t DominanceMaxIndex<Key, Value, NumVal, UIntSize>::size() const {
    return point_key.size();
}

template<typename Key, typename Value, size_t NumVal, typename UIntSize>
size_t DominanceMaxIndex<Key, Value, NumVal, UIntSize>::memory_size() const {
    return point_key.capacity() * sizeof(Key) + entries.capacity() * sizeof(Entry) + nodes.capacity() * sizeof(Node);
}

}

#endif /* centrolign_dominance_max_index_hpp */
