/*
  Stockfish NNUE Python Bindings
  Python bindings for extracting NNUE activations and evaluations from Stockfish
*/

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <tuple>
#include <vector>

#include "position.h"
#include "bitboard.h"
#include "types.h"
#include "evaluate.h"
#include "nnue/network.h"
#include "nnue/nnue_accumulator.h"
#include "nnue/nnue_architecture.h"

namespace py = pybind11;

namespace Stockfish {

// Forward declarations to satisfy -Wmissing-declarations
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, float, float>
get_activations_and_eval(const std::string& fen);
float get_evaluation(const std::string& fen);
py::dict get_network_info();

// Global network instance
static std::unique_ptr<Eval::NNUE::Networks> g_networks = nullptr;

// Initialize the networks
void init_networks() {
    if (g_networks == nullptr) {
        // Initialize Stockfish
        Bitboards::init();
        Position::init();
        
        // Load the default networks
        Eval::NNUE::EvalFile evalFileBig;
        evalFileBig.defaultName = EvalFileDefaultNameBig;
        evalFileBig.current = "";
        
        Eval::NNUE::EvalFile evalFileSmall;
        evalFileSmall.defaultName = EvalFileDefaultNameSmall;
        evalFileSmall.current = "";
        
        auto networkBig = Eval::NNUE::NetworkBig(evalFileBig, Eval::NNUE::EmbeddedNNUEType::BIG);
        auto networkSmall = Eval::NNUE::NetworkSmall(evalFileSmall, Eval::NNUE::EmbeddedNNUEType::SMALL);
        
        // Load the networks from default location
        networkBig.load("", EvalFileDefaultNameBig);
        networkSmall.load("", EvalFileDefaultNameSmall);
        
        g_networks = std::make_unique<Eval::NNUE::Networks>(
            std::move(networkBig), 
            std::move(networkSmall)
        );
    }
}

// Main function to extract activations and evaluation with intermediate layers
std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, py::array_t<float>, float, float> 
get_activations_and_eval(const std::string& fen) {
    
    // Initialize networks if not already done
    init_networks();
    
    // Create position from FEN
    StateInfo si;
    Position pos;
    pos.set(fen, false, &si);
    
    // Create accumulator stack and caches
    Eval::NNUE::AccumulatorStack accumulators;
    auto caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(*g_networks);
    
    // Determine which network to use
    bool useSmallNet = Eval::use_smallnet(pos);
    
    // Evaluate the position to populate the accumulator
    Value finalEval = Eval::evaluate(*g_networks, pos, accumulators, *caches, VALUE_ZERO);
    
    // Get the accumulator state
    const auto& accumulatorState = accumulators.latest();
    
    // Extract accumulator values (main hidden layer)
    Eval::NNUE::IndexType accSize = useSmallNet 
        ? Eval::NNUE::TransformedFeatureDimensionsSmall 
        : Eval::NNUE::TransformedFeatureDimensionsBig;
    
    // Create numpy arrays for accumulator (main hidden layer)
    auto accumulation_white = py::array_t<float>(accSize);
    auto accumulation_black = py::array_t<float>(accSize);
    
    auto acc_white_ptr = accumulation_white.mutable_unchecked<1>();
    auto acc_black_ptr = accumulation_black.mutable_unchecked<1>();
    
    // Copy accumulator data
    if (useSmallNet) {
        const auto& acc = accumulatorState.acc<Eval::NNUE::TransformedFeatureDimensionsSmall>();
        for (Eval::NNUE::IndexType i = 0; i < accSize; ++i) {
            acc_white_ptr(i) = static_cast<float>(acc.accumulation[WHITE][i]);
            acc_black_ptr(i) = static_cast<float>(acc.accumulation[BLACK][i]);
        }
    } else {
        const auto& acc = accumulatorState.acc<Eval::NNUE::TransformedFeatureDimensionsBig>();
        for (Eval::NNUE::IndexType i = 0; i < accSize; ++i) {
            acc_white_ptr(i) = static_cast<float>(acc.accumulation[WHITE][i]);
            acc_black_ptr(i) = static_cast<float>(acc.accumulation[BLACK][i]);
        }
    }
    
    // Create numpy array for PSQT values (explicit ShapeContainer for older pybind11)
    py::array::ShapeContainer psqt_shape{
        static_cast<py::ssize_t>(2),
        static_cast<py::ssize_t>(Eval::NNUE::PSQTBuckets)
    };
    auto psqt_values = py::array_t<float>(psqt_shape);
    auto psqt_ptr = psqt_values.mutable_unchecked<2>();
    
    // Copy PSQT data
    if (useSmallNet) {
        const auto& acc = accumulatorState.acc<Eval::NNUE::TransformedFeatureDimensionsSmall>();
        for (int color = 0; color < 2; ++color) {
            for (Eval::NNUE::IndexType bucket = 0; bucket < Eval::NNUE::PSQTBuckets; ++bucket) {
                psqt_ptr(color, bucket) = static_cast<float>(acc.psqtAccumulation[color][bucket]);
            }
        }
    } else {
        const auto& acc = accumulatorState.acc<Eval::NNUE::TransformedFeatureDimensionsBig>();
        for (int color = 0; color < 2; ++color) {
            for (Eval::NNUE::IndexType bucket = 0; bucket < Eval::NNUE::PSQTBuckets; ++bucket) {
                psqt_ptr(color, bucket) = static_cast<float>(acc.psqtAccumulation[color][bucket]);
            }
        }
    }
    
    // Convert evaluation to centipawns
    float finalEvalCp = static_cast<float>(finalEval) / 100.0f;
    
    // For now, return the same value for both positional and PSQT components
    // In a full implementation, you might want to separate these
    float psqtEvalCp = finalEvalCp;
    
    // Extract intermediate layer activations
    // We need to manually transform the accumulator and propagate through layers
    py::array_t<float> layer1_out;
    py::array_t<float> layer2_out;
    
    const int bucket = (pos.count<ALL_PIECES>() - 1) / 4;
    
    if (useSmallNet) {
        constexpr int L1 = Eval::NNUE::TransformedFeatureDimensionsSmall;
        constexpr int L2 = Eval::NNUE::L2Small;
        constexpr int L3 = Eval::NNUE::L3Small;
        
        const auto& acc = accumulatorState.acc<L1>();
        
        // Transform accumulator to uint8_t (clip to [0, 127])
        alignas(64) std::uint8_t transformedFeatures[L1 * 2];
        const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
        
        for (int p = 0; p < 2; ++p) {
            const int offset = (L1 / 2) * p;
            for (Eval::NNUE::IndexType i = 0; i < L1 / 2; ++i) {
                // Clip and scale int16_t to uint8_t [0, 127]
                auto val0 = acc.accumulation[perspectives[p]][i];
                auto val1 = acc.accumulation[perspectives[p]][i + L1 / 2];
                
                // Clamp to [0, 127]
                val0 = std::max<std::int16_t>(0, std::min<std::int16_t>(127, val0));
                val1 = std::max<std::int16_t>(0, std::min<std::int16_t>(127, val1));
                
                transformedFeatures[offset + i] = static_cast<std::uint8_t>((val0 * val1) / 128);
            }
        }
        
        // Now propagate through network layers to extract intermediate activations
        const auto& net = g_networks->small.get_network(bucket);
        
        alignas(64) std::int32_t fc_0_out[L2 + 1];
        alignas(64) std::uint8_t ac_sqr_0_out[L2 * 2];
        alignas(64) std::uint8_t ac_0_out[L2];
        alignas(64) std::int32_t fc_1_out[L3];
        alignas(64) std::uint8_t ac_1_out[L3];
        
        // Propagate through layers
        net.fc_0.propagate(transformedFeatures, fc_0_out);
        net.ac_sqr_0.propagate(fc_0_out, ac_sqr_0_out);
        net.ac_0.propagate(fc_0_out, ac_0_out);
        
        // Concatenate for layer 1 output
        layer1_out = py::array_t<float>(L2 * 2);
        auto l1_ptr = layer1_out.mutable_unchecked<1>();
        for (int i = 0; i < L2; ++i) {
            l1_ptr(i) = static_cast<float>(ac_sqr_0_out[i]);
            l1_ptr(i + L2) = static_cast<float>(ac_0_out[i]);
        }
        
        // Copy ac_0_out to second half of ac_sqr_0_out (as done in propagate)
        std::memcpy(ac_sqr_0_out + L2, ac_0_out, L2 * sizeof(std::uint8_t));
        
        // Continue propagation
        net.fc_1.propagate(ac_sqr_0_out, fc_1_out);
        net.ac_1.propagate(fc_1_out, ac_1_out);
        
        // Copy layer 2 activations
        layer2_out = py::array_t<float>(L3);
        auto l2_ptr = layer2_out.mutable_unchecked<1>();
        for (int i = 0; i < L3; ++i) {
            l2_ptr(i) = static_cast<float>(ac_1_out[i]);
        }
        
    } else {
        constexpr int L1 = Eval::NNUE::TransformedFeatureDimensionsBig;
        constexpr int L2 = Eval::NNUE::L2Big;
        constexpr int L3 = Eval::NNUE::L3Big;
        
        const auto& acc = accumulatorState.acc<L1>();
        
        // Transform accumulator to uint8_t (clip to [0, 127])
        alignas(64) std::uint8_t transformedFeatures[L1 * 2];
        const Color perspectives[2] = {pos.side_to_move(), ~pos.side_to_move()};
        
        for (int p = 0; p < 2; ++p) {
            const int offset = (L1 / 2) * p;
            for (Eval::NNUE::IndexType i = 0; i < L1 / 2; ++i) {
                // Clip and scale int16_t to uint8_t [0, 127]
                auto val0 = acc.accumulation[perspectives[p]][i];
                auto val1 = acc.accumulation[perspectives[p]][i + L1 / 2];
                
                // Clamp to [0, 127]
                val0 = std::max<std::int16_t>(0, std::min<std::int16_t>(127, val0));
                val1 = std::max<std::int16_t>(0, std::min<std::int16_t>(127, val1));
                
                transformedFeatures[offset + i] = static_cast<std::uint8_t>((val0 * val1) / 128);
            }
        }
        
        // Now propagate through network layers to extract intermediate activations
        const auto& net = g_networks->big.get_network(bucket);
        
        alignas(64) std::int32_t fc_0_out[L2 + 1];
        alignas(64) std::uint8_t ac_sqr_0_out[L2 * 2];
        alignas(64) std::uint8_t ac_0_out[L2];
        alignas(64) std::int32_t fc_1_out[L3];
        alignas(64) std::uint8_t ac_1_out[L3];
        
        // Propagate through layers
        net.fc_0.propagate(transformedFeatures, fc_0_out);
        net.ac_sqr_0.propagate(fc_0_out, ac_sqr_0_out);
        net.ac_0.propagate(fc_0_out, ac_0_out);
        
        // Concatenate for layer 1 output
        layer1_out = py::array_t<float>(L2 * 2);
        auto l1_ptr = layer1_out.mutable_unchecked<1>();
        for (int i = 0; i < L2; ++i) {
            l1_ptr(i) = static_cast<float>(ac_sqr_0_out[i]);
            l1_ptr(i + L2) = static_cast<float>(ac_0_out[i]);
        }
        
        // Copy ac_0_out to second half of ac_sqr_0_out (as done in propagate)
        std::memcpy(ac_sqr_0_out + L2, ac_0_out, L2 * sizeof(std::uint8_t));
        
        // Continue propagation
        net.fc_1.propagate(ac_sqr_0_out, fc_1_out);
        net.ac_1.propagate(fc_1_out, ac_1_out);
        
        // Copy layer 2 activations
        layer2_out = py::array_t<float>(L3);
        auto l2_ptr = layer2_out.mutable_unchecked<1>();
        for (int i = 0; i < L3; ++i) {
            l2_ptr(i) = static_cast<float>(ac_1_out[i]);
        }
    }
    
    return std::make_tuple(
        accumulation_white,
        accumulation_black, 
        psqt_values,
        layer1_out,
        layer2_out,
        finalEvalCp,
        psqtEvalCp
    );
}

// Simple function to get just the evaluation
float get_evaluation(const std::string& fen) {
    init_networks();
    
    StateInfo si;
    Position pos;
    pos.set(fen, false, &si);
    
    Eval::NNUE::AccumulatorStack accumulators;
    auto caches = std::make_unique<Eval::NNUE::AccumulatorCaches>(*g_networks);
    
    Value finalEval = Eval::evaluate(*g_networks, pos, accumulators, *caches, VALUE_ZERO);
    return static_cast<float>(finalEval) / 100.0f;
}

// Get network architecture information
py::dict get_network_info() {
    py::dict info;
    info["TransformedFeatureDimensionsBig"] = Eval::NNUE::TransformedFeatureDimensionsBig;
    info["TransformedFeatureDimensionsSmall"] = Eval::NNUE::TransformedFeatureDimensionsSmall;
    info["PSQTBuckets"] = Eval::NNUE::PSQTBuckets;
    info["L2Big"] = Eval::NNUE::L2Big;
    info["L3Big"] = Eval::NNUE::L3Big;
    info["L2Small"] = Eval::NNUE::L2Small;
    info["L3Small"] = Eval::NNUE::L3Small;
    return info;
}

} // namespace Stockfish

PYBIND11_MODULE(stockfish_nnue, m) {
    m.doc() = "Stockfish NNUE Python bindings";
    
    m.def("get_activations_and_eval", &Stockfish::get_activations_and_eval,
          "Get NNUE activations and evaluation for a position",
          py::arg("fen"));
    
    m.def("get_evaluation", &Stockfish::get_evaluation,
          "Get NNUE evaluation for a position",
          py::arg("fen"));
    
    m.def("get_network_info", &Stockfish::get_network_info,
          "Get network architecture information");
}