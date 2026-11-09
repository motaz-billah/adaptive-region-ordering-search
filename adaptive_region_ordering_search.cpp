#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Adaptive Region Ordering Search (AROS)
// C++17 port of the Python prototype.
// The input vector must remain alive and unchanged while this object is used.

struct AdaptiveSearchConfig {
    double decay = 0.95;          // Exponential history decay.
    std::size_t window = 128;     // Number of recent successful queries to retain.
    double alpha = 0.50;          // Weight assigned to recent observations.
    double gamma = 0.50;          // Weight assigned to long-term history vs. trend.
    double beta = 0.25;           // Trend strength.
};

struct SearchResult {
    std::optional<std::size_t> index;
    std::size_t comparisons = 0;

    bool found() const noexcept { return index.has_value(); }
};

template <typename T>
class AdaptiveRegionOrderingSearch {
public:
    static constexpr std::size_t REGION_COUNT = 4;

    explicit AdaptiveRegionOrderingSearch(
        const std::vector<T>& data,
        const AdaptiveSearchConfig& config = AdaptiveSearchConfig{})
        : data_(data), config_(config) {
        validateConfig();
        buildRegions();
        order_ = {0, 1, 2, 3};
        weights_.fill(0.0);
        predicted_.fill(1.0 / static_cast<double>(REGION_COUNT));
    }

    SearchResult search(const T& target) {
        SearchResult result;

        for (std::size_t region : order_) {
            const Region& range = regions_[region];

            for (std::size_t index = range.begin; index < range.end; ++index) {
                ++result.comparisons;

                if (data_[index] == target) {
                    result.index = index;
                    ++successes_;
                    learn(region);
                    return result;
                }
            }
        }

        ++misses_;
        return result;
    }

    std::size_t successes() const noexcept { return successes_; }
    std::size_t misses() const noexcept { return misses_; }
    std::size_t orderUpdates() const noexcept { return orderUpdates_; }

    const std::array<std::size_t, REGION_COUNT>& regionOrder() const noexcept {
        return order_;
    }

    const std::array<double, REGION_COUNT>& predictedProbabilities() const noexcept {
        return predicted_;
    }

    const std::array<double, REGION_COUNT>& decayedWeights() const noexcept {
        return weights_;
    }

private:
    struct Region {
        std::size_t begin = 0;
        std::size_t end = 0; // Exclusive.
    };

    const std::vector<T>& data_;
    AdaptiveSearchConfig config_;
    std::array<Region, REGION_COUNT> regions_{};
    std::array<std::size_t, REGION_COUNT> order_{};
    std::array<double, REGION_COUNT> weights_{};
    std::array<double, REGION_COUNT> predicted_{};
    std::array<double, REGION_COUNT> previousHistory_{};
    bool hasPreviousHistory_ = false;
    std::deque<std::size_t> recent_;
    std::size_t successes_ = 0;
    std::size_t misses_ = 0;
    std::size_t orderUpdates_ = 0;

    void validateConfig() const {
        const auto finite = [](double value) { return std::isfinite(value); };

        if (!finite(config_.decay) || config_.decay <= 0.0 || config_.decay >= 1.0 ||
            config_.window == 0 ||
            !finite(config_.alpha) || config_.alpha < 0.0 || config_.alpha > 1.0 ||
            !finite(config_.gamma) || config_.gamma < 0.0 || config_.gamma > 1.0 ||
            !finite(config_.beta) || config_.beta < 0.0) {
            throw std::invalid_argument("Invalid adaptive-search parameter.");
        }
    }

    void buildRegions() {
        const std::size_t n = data_.size();

        // Equivalent to floor(i * n / REGION_COUNT), while reducing overflow risk.
        for (std::size_t i = 0; i < REGION_COUNT; ++i) {
            const std::size_t begin = (n / REGION_COUNT) * i + ((n % REGION_COUNT) * i) / REGION_COUNT;
            const std::size_t next = i + 1;
            const std::size_t end = (n / REGION_COUNT) * next + ((n % REGION_COUNT) * next) / REGION_COUNT;
            regions_[i] = {begin, end};
        }
    }

    static std::array<double, REGION_COUNT> normalize(
        const std::array<double, REGION_COUNT>& values) {
        std::array<double, REGION_COUNT> result{};
        double sum = 0.0;

        for (std::size_t i = 0; i < REGION_COUNT; ++i) {
            result[i] = std::max(0.0, values[i]);
            sum += result[i];
        }

        if (!(sum > 0.0)) {
            result.fill(1.0 / static_cast<double>(REGION_COUNT));
            return result;
        }

        for (double& value : result) {
            value /= sum;
        }
        return result;
    }

    double expectedCost(const std::array<std::size_t, REGION_COUNT>& candidate,
                        const std::array<double, REGION_COUNT>& probabilities) const {
        double prefixSize = 0.0;
        double totalCost = 0.0;

        for (std::size_t region : candidate) {
            const double size = static_cast<double>(
                regions_[region].end - regions_[region].begin);

            if (size > 0.0) {
                totalCost += probabilities[region] *
                             (prefixSize + (size + 1.0) / 2.0);
            }
            prefixSize += size;
        }

        return totalCost;
    }

    std::array<std::size_t, REGION_COUNT> findBestOrder(
        const std::array<double, REGION_COUNT>& probabilities) const {
        std::array<std::size_t, REGION_COUNT> candidate{0, 1, 2, 3};
        std::array<std::size_t, REGION_COUNT> best = candidate;
        double bestCost = std::numeric_limits<double>::infinity();

        // Enumerate all 4! = 24 possible region orders.
        do {
            const double candidateCost = expectedCost(candidate, probabilities);
            // Strict comparison preserves the first lexicographic order on ties,
            // matching the Python prototype's permutation/min behavior.
            if (candidateCost < bestCost) {
                bestCost = candidateCost;
                best = candidate;
            }
        } while (std::next_permutation(candidate.begin(), candidate.end()));

        return best;
    }

    std::array<double, REGION_COUNT> predict() {
        const auto history = normalize(weights_);
        std::array<double, REGION_COUNT> trend = history;

        if (hasPreviousHistory_) {
            std::array<double, REGION_COUNT> rawTrend{};
            for (std::size_t i = 0; i < REGION_COUNT; ++i) {
                rawTrend[i] = history[i] +
                              config_.beta * (history[i] - previousHistory_[i]);
            }
            trend = normalize(rawTrend);
        }

        std::array<double, REGION_COUNT> recentCounts{};
        for (std::size_t region : recent_) {
            recentCounts[region] += 1.0;
        }
        const auto recentProbabilities = normalize(recentCounts);

        std::array<double, REGION_COUNT> longTerm{};
        std::array<double, REGION_COUNT> combined{};

        for (std::size_t i = 0; i < REGION_COUNT; ++i) {
            longTerm[i] = config_.gamma * history[i] +
                          (1.0 - config_.gamma) * trend[i];
            combined[i] = config_.alpha * recentProbabilities[i] +
                          (1.0 - config_.alpha) * longTerm[i];
        }

        predicted_ = normalize(combined);
        return predicted_;
    }

    void learn(std::size_t successfulRegion) {
        // Save normalized history before applying the next decay/update.
        previousHistory_ = normalize(weights_);
        hasPreviousHistory_ = true;

        for (double& weight : weights_) {
            weight *= config_.decay;
        }
        weights_[successfulRegion] += 1.0;

        recent_.push_back(successfulRegion);
        if (recent_.size() > config_.window) {
            recent_.pop_front();
        }

        const auto newOrder = findBestOrder(predict());
        if (newOrder != order_) {
            order_ = newOrder;
            ++orderUpdates_;
        }
    }
};

// --------------------------- Correctness tests ---------------------------

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
    }
}

void runCorrectnessTests() {
    // Empty and single-element arrays.
    {
        const std::vector<int> empty;
        AdaptiveRegionOrderingSearch<int> searchEmpty(empty);
        const auto miss = searchEmpty.search(10);
        require(!miss.found() && miss.comparisons == 0, "empty-array search");

        const std::vector<int> one{42};
        AdaptiveRegionOrderingSearch<int> searchOne(one);
        const auto hit = searchOne.search(42);
        require(hit.found() && *hit.index == 0 && hit.comparisons == 1,
                "single-element hit");
        const auto oneMiss = searchOne.search(-1);
        require(!oneMiss.found() && oneMiss.comparisons == 1,
                "single-element miss");
    }

    // All existing values must be found at a valid matching position.
    {
        std::vector<int> values(250);
        std::iota(values.begin(), values.end(), 0);
        std::mt19937 rng(28);
        std::shuffle(values.begin(), values.end(), rng);

        AdaptiveRegionOrderingSearch<int> search(values);
        for (int target : values) {
            const auto result = search.search(target);
            require(result.found(), "existing value was not found");
            require(values[*result.index] == target, "returned index has wrong value");
            require(result.comparisons >= 1 && result.comparisons <= values.size(),
                    "comparison count is outside valid bounds");
        }
    }

    // A miss scans all elements and does not change the learned model.
    {
        const std::vector<int> values{19, -4, 8, 100, 7, 0};
        AdaptiveRegionOrderingSearch<int> search(values);
        const auto result = search.search(999);
        require(!result.found(), "missing value incorrectly found");
        require(result.comparisons == values.size(), "miss did not scan all elements");
        require(search.successes() == 0 && search.misses() == 1,
                "success/miss counters are incorrect");
    }

    // Duplicates: any returned matching position is valid (not necessarily first).
    {
        const std::vector<int> values{5, 1, 5, 9, 1, 5, 7};
        AdaptiveRegionOrderingSearch<int> search(values);
        for (int target : {1, 5, 7, 9}) {
            const auto result = search.search(target);
            require(result.found(), "duplicate test value was not found");
            require(values[*result.index] == target, "duplicate test returned wrong value");
        }
    }

    // Probabilities remain normalized and region order remains a permutation.
    {
        std::vector<int> values(400);
        std::iota(values.begin(), values.end(), 0);
        AdaptiveRegionOrderingSearch<int> search(values);

        for (int i = 0; i < 2'000; ++i) {
            const std::size_t index = static_cast<std::size_t>((i * 37) % values.size());
            (void)search.search(values[index]);

            const auto& probabilities = search.predictedProbabilities();
            double sum = 0.0;
            for (double p : probabilities) {
                require(p >= 0.0, "negative predicted probability");
                sum += p;
            }
            require(std::abs(sum - 1.0) < 1e-9, "probabilities do not sum to one");

            auto order = search.regionOrder();
            std::sort(order.begin(), order.end());
            require(order == std::array<std::size_t, 4>{0, 1, 2, 3},
                    "region order is not a valid permutation");
        }
    }

    std::cout << "[PASS] All correctness tests passed.\n";
}

// ------------------------------- Demo ------------------------------------

void runDemo() {
    constexpr std::size_t N = 20'000;
    constexpr std::size_t QUERY_COUNT = 5'000;

    std::vector<int> data(N);
    std::iota(data.begin(), data.end(), 0);
    std::mt19937 rng(20261109);
    std::shuffle(data.begin(), data.end(), rng);

    // Build regions using the same boundary rule as the search class.
    std::array<std::pair<std::size_t, std::size_t>, 4> bounds{};
    for (std::size_t r = 0; r < 4; ++r) {
        bounds[r] = {r * N / 4, (r + 1) * N / 4};
    }

    // This workload deliberately places 70% of successful queries in region 4.
    std::discrete_distribution<int> chooseRegion({0.1, 0.1, 0.1, 0.7});
    std::vector<int> queries;
    queries.reserve(QUERY_COUNT);

    for (std::size_t q = 0; q < QUERY_COUNT; ++q) {
        const std::size_t region = static_cast<std::size_t>(chooseRegion(rng));
        std::uniform_int_distribution<std::size_t> chooseIndex(
            bounds[region].first, bounds[region].second - 1);
        const std::size_t index = chooseIndex(rng);
        queries.push_back(data[index]);
    }

    AdaptiveRegionOrderingSearch<int> adaptive(data);
    std::size_t adaptiveComparisons = 0;
    std::size_t linearComparisons = 0;

    for (int target : queries) {
        const auto adaptiveResult = adaptive.search(target);
        adaptiveComparisons += adaptiveResult.comparisons;

        for (int value : data) {
            ++linearComparisons;
            if (value == target) {
                break;
            }
        }
    }

    const double adaptiveAverage =
        static_cast<double>(adaptiveComparisons) / QUERY_COUNT;
    const double linearAverage =
        static_cast<double>(linearComparisons) / QUERY_COUNT;
    const double comparisonSpeedup =
        adaptiveAverage > 0.0 ? linearAverage / adaptiveAverage : 0.0;

    std::cout << "\n=== Adaptive Region Ordering Search (AROS) ===\n";
    std::cout << "Array size: " << N << '\n';
    std::cout << "Queries: " << QUERY_COUNT << '\n';
    std::cout << "Workload: 70% of successful queries target region 4\n\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Linear average comparisons:   " << linearAverage << '\n';
    std::cout << "AROS average comparisons:     " << adaptiveAverage << '\n';
    std::cout << "Comparison-count ratio:       " << comparisonSpeedup << "x\n";
    std::cout << "Successful queries:           " << adaptive.successes() << '\n';
    std::cout << "Misses:                       " << adaptive.misses() << '\n';
    std::cout << "Region-order updates:         " << adaptive.orderUpdates() << '\n';

    std::cout << "Final region order:            ";
    for (std::size_t region : adaptive.regionOrder()) {
        std::cout << (region + 1) << ' ';
    }
    std::cout << "\nPredicted region probabilities: ";
    for (double p : adaptive.predictedProbabilities()) {
        std::cout << p << ' ';
    }
    std::cout << "\n\nNote: this demo compares comparison counts, not total wall-clock time.\n";
}

int main(int argc, char* argv[]) {
    try {
        runCorrectnessTests();

        if (argc > 1 && std::string(argv[1]) == "--test") {
            return 0;
        }

        runDemo();
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
