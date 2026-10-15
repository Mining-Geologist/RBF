"""Apply minimal Linux GCC compatibility fix to Jizai 1.0.0 source checkout.

std::reduce permits the binary operation in either operand order. The upstream
lambda assumes an integer accumulator is always the first argument. Replacing
std::reduce with std::accumulate guarantees a left fold.
"""
from pathlib import Path

source = Path("jizai-src/include/jizai/geostats/variogram_set.hpp")
text = source.read_text()
before = "return std::reduce(variograms_.begin(), variograms_.end(), Index{0},"
after = "return std::accumulate(variograms_.begin(), variograms_.end(), Index{0},"
if text.count(before) != 1:
    raise RuntimeError("Jizai upstream changed; refusing an unverified patch")
source.write_text(text.replace(before, after))
print(f"Patched {source}: std::reduce -> std::accumulate")

# Upstream ScalFMM's header defines global lambda variables with external linkage.
# C++17 inline variables allow one shared definition across translation units.
scalfmm = Path("jizai-src/third_party/scalfmm/include/scalfmm/tree/io.hpp")
source_text = scalfmm.read_text()
for name in ("print_map", "print_with_map"):
    old = f"    auto {name} = []("
    new = f"    inline auto {name} = []("
    if source_text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one ScalFMM {name} definition")
    source_text = source_text.replace(old, new)
scalfmm.write_text(source_text)
print("Patched ScalFMM header: inline global lambda variables")
