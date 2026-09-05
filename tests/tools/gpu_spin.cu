// Occupy a GPU's SMs so another process has to contend for them.
//
// The multi-GPU NaNs (F4 in docs/multigpu-plan.md) only ever appeared against a genuinely
// saturated peer GPU, and the leading suspects -- the CPU-side spin barriers in modupMGPU
// and the cross-device polling kernels in PeerUtils.cu -- are exactly the constructs that
// break when the notifying side cannot get SMs. A matmul loop is a poor stressor for that:
// it yields between launches. This launches one long-running block per SM instead, each
// spinning on the clock, so the GPU has no free slot to schedule a peer's polling kernel
// into.
//
//   nvcc -O2 -arch=native -o gpu_spin gpu_spin.cu
//   ./gpu_spin 0 2 3          # spin on these devices until killed
//
// Sends SIGINT/SIGTERM-safe: the kernels are bounded (10 s each) and relaunched in a loop,
// so killing the process never leaves a GPU stuck in an unkillable kernel.

#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <vector>

namespace {
volatile sig_atomic_t stop_requested = 0;
void on_signal(int) { stop_requested = 1; }
} // namespace

__global__ void spin(long long cycles) {
	const long long start = clock64();
	// Volatile accumulator so the compiler cannot hoist the loop away.
	volatile long long acc = 0;
	while (clock64() - start < cycles)
		acc += 1;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		std::fprintf(stderr, "usage: %s <device> [device ...]\n", argv[0]);
		return 2;
	}
	std::signal(SIGINT, on_signal);
	std::signal(SIGTERM, on_signal);

	std::vector<int> devices;
	for (int i = 1; i < argc; ++i)
		devices.push_back(std::atoi(argv[i]));

	std::vector<cudaStream_t> streams(devices.size());
	std::vector<int> blocks(devices.size());
	for (size_t i = 0; i < devices.size(); ++i) {
		if (cudaSetDevice(devices[i]) != cudaSuccess) {
			std::fprintf(stderr, "cannot select device %d\n", devices[i]);
			return 1;
		}
		cudaDeviceProp prop{};
		cudaGetDeviceProperties(&prop, devices[i]);
		// One block per SM, sized so a block fills the SM: nothing else gets scheduled.
		blocks[i] = prop.multiProcessorCount;
		cudaStreamCreate(&streams[i]);
		std::printf("spinning on device %d (%s, %d SMs)\n", devices[i], prop.name, blocks[i]);
	}
	std::fflush(stdout);

	// ~10 s per launch at any plausible clock; relaunched until we are asked to stop.
	const long long cycles = 10LL * 1000 * 1000 * 1000;
	while (!stop_requested) {
		for (size_t i = 0; i < devices.size(); ++i) {
			cudaSetDevice(devices[i]);
			spin<<<blocks[i], 256, 0, streams[i]>>>(cycles);
		}
		for (size_t i = 0; i < devices.size(); ++i) {
			cudaSetDevice(devices[i]);
			cudaStreamSynchronize(streams[i]);
		}
	}
	for (size_t i = 0; i < devices.size(); ++i) {
		cudaSetDevice(devices[i]);
		cudaStreamDestroy(streams[i]);
	}
	return 0;
}
