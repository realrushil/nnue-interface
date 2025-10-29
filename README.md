# nnue-interface

Python bindings for extracting Stockfish NNUE neural network activations and evaluations.

## Features

✨ **Key Capabilities:**
- Extract NNUE **accumulator activations** (hidden layer 0): 3072 dimensions (Big network) or 128 dimensions (Small network)
- Extract **intermediate layer activations** (layers 1-2): For deep network analysis
- Extract **PSQT values**: Piece-square table contributions  
- Get **final evaluations** in centipawns
- **Cross-platform**: Works on Linux, macOS, and Windows
- **Fast**: Compiled C++ extension via pybind11
- **ML-Ready**: All outputs as float32 numpy arrays

## Installation

### From PyPI (recommended)
```bash
pip install nnue-interface
```

### From Source
```bash
git clone https://github.com/yourusername/nnue-interface.git
cd nnue-interface
pip install -e .
```

**Requirements:**
- Python 3.8+
- C++17 compatible compiler (GCC, Clang, MSVC)
- CMake 3.15+
- NumPy 1.19+

## Quick Start

```python
import nnue_interface
import numpy as np

# Extract all NNUE activations and evaluation for a position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

acc_white, acc_black, psqt, layer1, layer2, eval_final, eval_psqt = \
    nnue_interface.get_activations_and_eval(fen)

print(f"Evaluation: {eval_final:.2f} cp")
print(f"Accumulator shape: {acc_white.shape}")  # (3072,) for Big network
print(f"Layer 1 shape: {layer1.shape}")         # (30,) 
print(f"Layer 2 shape: {layer2.shape}")         # (32,)
```

## API Reference

### `get_activations_and_eval(fen: str) -> tuple`

Extract all NNUE activations and evaluation for a given position.

**Parameters:**
- `fen` (str): FEN notation of the chess position

**Returns:**
- `acc_white` (ndarray): White perspective accumulator, shape (3072,) or (128,)
- `acc_black` (ndarray): Black perspective accumulator, shape (3072,) or (128,)
- `psqt` (ndarray): PSQT values, shape (2, 8)
- `layer1` (ndarray): First hidden layer activations, shape (30,) or (15×2)
- `layer2` (ndarray): Second hidden layer activations, shape (32,)
- `eval_final` (float): Final evaluation in centipawns
- `eval_psqt` (float): PSQT-only evaluation in centipawns

### `get_evaluation(fen: str) -> float`

Get only the final evaluation for a position (faster if you don't need activations).

**Parameters:**
- `fen` (str): FEN notation

**Returns:**
- Evaluation in centipawns (float)

### `get_network_info() -> dict`

Get information about the NNUE network architecture.

**Returns:**
```python
{
    'TransformedFeatureDimensionsBig': 3072,
    'TransformedFeatureDimensionsSmall': 128,
    'L2Big': 15,
    'L3Big': 32,
    'L2Small': 15,
    'L3Small': 32,
    'PSQTBuckets': 8,
}
```

## Examples

### Using Activations for Machine Learning

```python
import nnue_interface
import numpy as np

# Collect training data
positions = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    # ... more positions
]

X_acc = []  # Accumulator features
X_layers = []  # Intermediate layer features
y = []  # Target evaluations

for fen in positions:
    acc_w, acc_b, psqt, layer1, layer2, eval_final, _ = \
        nnue_interface.get_activations_and_eval(fen)
    
    # Combine features
    features = np.concatenate([acc_w, acc_b, psqt.flatten(), layer1, layer2])
    
    X_layers.append(features)
    y.append(eval_final)

X = np.array(X_layers)
y = np.array(y)

# Now use X and y for ML training (scikit-learn, PyTorch, TensorFlow, etc.)
```

### Analyzing Network Activations

```python
import nnue_interface

fen = "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"

acc_w, acc_b, psqt, layer1, layer2, eval_final, _ = \
    nnue_interface.get_activations_and_eval(fen)

print(f"Position evaluation: {eval_final:.2f} cp")
print(f"\nAccumulator statistics (White):")
print(f"  Mean: {acc_w.mean():.2f}")
print(f"  Std:  {acc_w.std():.2f}")
print(f"  Min:  {acc_w.min():.2f}")
print(f"  Max:  {acc_w.max():.2f}")

print(f"\nLayer 1 sparsity: {(layer1 == 0).sum() / layer1.size * 100:.1f}%")
print(f"Layer 2 sparsity: {(layer2 == 0).sum() / layer2.size * 100:.1f}%")
```

## Architecture

The NNUE network consists of:

1. **Input Layer (Accumulator)**: 3072 or 128 dimensions
   - Efficiently updatable representation of board state
   - Updated incrementally as moves are made

2. **Layer 1**: 
   - FC layer (sparse input) + SqrClippedReLU + ClippedReLU
   - Output: 15 dims × 2 (concatenated)

3. **Layer 2**:
   - FC layer + ClippedReLU
   - Output: 32 dims

4. **Output Layer**:
   - FC layer → single scalar evaluation
   - Also uses PSQT values for final output

All intermediate activations use int8/uint8 quantization for efficiency, converted to float32 for Python.

## Performance

- **Speed**: ~50-200 µs per position (depending on CPU)
- **Memory**: ~2 MB for network weights
- **No dependencies**: Only NumPy required at runtime

## Building from Source

### Linux / macOS
```bash
git clone https://github.com/realrushil/nnue-interface.git
cd nnue-interface
pip install -e .
```

### Windows (MSYS2 UCRT64)
```bash
git clone https://github.com/realrushil/nnue-interface.git
cd nnue-interface
pip install -e .
```

### Build Documentation

CMake is used for cross-platform builds. To manually build:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Testing

```bash
pip install pytest numpy
pytest tests/
```

## License

GPL-3.0-or-later (same as Stockfish)

## Citation

If you use this in research, please cite Stockfish:

```bibtex
@software{stockfish,
  title = {Stockfish},
  url = {https://stockfishchess.org/},
  author = {Tord Romstad and Marco Costalba and Joona Kiiski and Gary Linscott},
}
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## Troubleshooting

### Build fails on Linux with missing dependencies
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake python3-dev

# Fedora
sudo dnf install gcc gcc-c++ cmake python3-devel
```

### Import error on Windows
Make sure you're using the correct Python version (64-bit). The wheel must match your Python installation:
```bash
python -c "import struct; print('64-bit' if struct.calcsize('P') == 8 else '32-bit')"
```

## Resources

- [Stockfish Official](https://stockfishchess.org/)
- [NNUE Research](https://github.com/official-stockfish/Stockfish/tree/master/src/nnue)
- [Chess Programming Wiki - NNUE](https://www.chessprogramming.org/NNUE)

## Authors

- Created with Stockfish source code
- Python bindings by Rushil Saraf

---

**Have questions?** Open an issue on GitHub!
