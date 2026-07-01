// pybind11 bindings for FIDESlib (CKKS GPU library, CAPS-UMU).
//
// Wraps the subset of the FIDESlib public API needed by the
// HerMiniRocket/PolyMiniRocket encrypted-inference pipeline:
// context setup, key generation, encoding, encrypt/decrypt, leveled ops,
// rotations, EvalChebyshevSeries, AccumulateSum and bootstrapping.
//
// Notes:
//  - All GPU-heavy calls release the GIL.
//  - Do NOT import this module together with `openfhe` (openfhe-python) in the
//    same process: this module embeds a patched OpenFHE 1.5.1 statically.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fideslib.hpp>

#include <bit>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace fideslib;

using CC	= CryptoContextImpl<DCRTPoly>;
using CtI	= CiphertextImpl<DCRTPoly>;
using Ct	= Ciphertext<DCRTPoly>; // std::shared_ptr<CtI>
using Pt	= Plaintext;			// std::shared_ptr<PlaintextImpl>
using PubK	= PublicKey<DCRTPoly>;
using PrivK = PrivateKey<DCRTPoly>;
using KP	= KeyPair<DCRTPoly>;
using Params = CCParams<CryptoContextCKKSRNS>;

namespace {
constexpr auto nogil = py::call_guard<py::gil_scoped_release>();

// Mirrors FIDESlib::CKKS::GetAccumulateRotationIndices (src/CKKS/AccumulateBroadcast.cu).
// The public AccumulateSum API uses bStep = 4 internally.
std::vector<int> AccumulateRotationIndices(int slots, int stride, int bStep) {
	std::vector<int> indices;
	int logbStep = std::bit_width(static_cast<uint32_t>(bStep)) - 1;
	for (int s = stride; s < stride * slots; s <<= logbStep) {
		for (int idx = s; idx < s * bStep && idx < stride * slots; idx += s) {
			indices.push_back(idx);
		}
	}
	return indices;
}
} // namespace

PYBIND11_MODULE(_core, m) {
	m.doc() = "Python bindings for FIDESlib (CKKS on GPU, interoperable with OpenFHE)";

	// ---- Enums ----

	py::enum_<PKESchemeFeature>(m, "PKESchemeFeature")
		.value("PKE", PKE)
		.value("KEYSWITCH", KEYSWITCH)
		.value("PRE", PRE)
		.value("LEVELEDSHE", LEVELEDSHE)
		.value("ADVANCEDSHE", ADVANCEDSHE)
		.value("MULTIPARTY", MULTIPARTY)
		.value("FHE", FHE)
		.value("SCHEMESWITCH", SCHEMESWITCH)
		.export_values();

	py::enum_<ScalingTechnique>(m, "ScalingTechnique")
		.value("FIXEDMANUAL", FIXEDMANUAL)
		.value("FIXEDAUTO", FIXEDAUTO)
		.value("FLEXIBLEAUTO", FLEXIBLEAUTO)
		.value("FLEXIBLEAUTOEXT", FLEXIBLEAUTOEXT)
		.export_values();

	py::enum_<KeySwitchTechnique>(m, "KeySwitchTechnique")
		.value("HYBRID", HYBRID)
		.export_values();

	py::enum_<SecretKeyDist>(m, "SecretKeyDist")
		.value("GAUSSIAN", GAUSSIAN)
		.value("UNIFORM_TERNARY", UNIFORM_TERNARY)
		.value("SPARSE_TERNARY", SPARSE_TERNARY)
		.value("SPARSE_ENCAPSULATED", SPARSE_ENCAPSULATED)
		.export_values();

	py::enum_<SecurityLevel>(m, "SecurityLevel")
		.value("HEStd_128_classic", HEStd_128_classic)
		.value("HEStd_192_classic", HEStd_192_classic)
		.value("HEStd_256_classic", HEStd_256_classic)
		.value("HEStd_128_quantum", HEStd_128_quantum)
		.value("HEStd_192_quantum", HEStd_192_quantum)
		.value("HEStd_256_quantum", HEStd_256_quantum)
		.value("HEStd_NotSet", HEStd_NotSet)
		.export_values();

	// ---- Keys ----

	py::class_<PublicKeyImpl<DCRTPoly>, PubK>(m, "PublicKey");
	py::class_<PrivateKeyImpl<DCRTPoly>, PrivK>(m, "PrivateKey");

	py::class_<KP>(m, "KeyPair")
		.def_readonly("publicKey", &KP::publicKey)
		.def_readonly("secretKey", &KP::secretKey);

	// ---- Plaintext ----

	py::class_<PlaintextImpl, Pt>(m, "Plaintext")
		.def("SetLength", &PlaintextImpl::SetLength, py::arg("length"))
		.def("SetSlots", &PlaintextImpl::SetSlots, py::arg("slots"))
		.def("GetLogPrecision", &PlaintextImpl::GetLogPrecision)
		.def("GetLevel", &PlaintextImpl::GetLevel)
		.def("GetCKKSPackedValue", &PlaintextImpl::GetCKKSPackedValue)
		.def("GetRealPackedValue", &PlaintextImpl::GetRealPackedValue)
		.def(
			"UnloadFromDevice",
			[](PlaintextImpl& pt) {
				// Free the GPU copy but keep the CPU-side encoding. Ops load
				// plaintexts to the device on first use and the buffer then
				// lives until the Plaintext is destroyed — for host-side
				// plaintext caches (he_benchmark_gpu pt_cache) that would pin
				// hundreds of GB of VRAM. Unloading after each use caps device
				// residency at one ct's worth; the next use re-uploads (~ms).
				if (pt.loaded && pt.gpu != 0 && pt.parent_context) {
					pt.parent_context->EvictDevicePlaintext(pt.gpu);
					pt.gpu	  = 0;
					pt.loaded = false;
				}
			},
			nogil)
		.def("__str__", [](const PlaintextImpl& pt) {
			std::ostringstream os;
			os << pt;
			return os.str();
		});

	// ---- Ciphertext ----

	py::class_<CtI, Ct>(m, "Ciphertext")
		.def("Clone", &CtI::Clone, nogil)
		.def("GetLevel", &CtI::GetLevel)
		.def("GetNoiseScaleDeg", &CtI::GetNoiseScaleDeg)
		.def("SetSlots", &CtI::SetSlots, py::arg("slots"));

	// ---- Parameters ----

	py::class_<Params>(m, "CCParams")
		.def(py::init<>())
		.def("SetMultiplicativeDepth", &Params::SetMultiplicativeDepth)
		.def("SetScalingModSize", &Params::SetScalingModSize)
		.def("SetBatchSize", &Params::SetBatchSize)
		.def("SetRingDim", &Params::SetRingDim)
		.def("SetScalingTechnique", &Params::SetScalingTechnique)
		.def("SetNumLargeDigits", &Params::SetNumLargeDigits)
		.def("SetFirstModSize", &Params::SetFirstModSize)
		.def("SetDigitSize", &Params::SetDigitSize)
		.def("SetKeySwitchTechnique", &Params::SetKeySwitchTechnique)
		.def("SetSecretKeyDist", &Params::SetSecretKeyDist)
		.def("SetSecurityLevel", &Params::SetSecurityLevel)
		.def("SetDevices", [](Params& p, std::vector<int> devices) { p.SetDevices(std::move(devices)); }, py::arg("devices"))
		.def("SetPlaintextAutoload", &Params::SetPlaintextAutoload)
		.def("SetCiphertextAutoload", &Params::SetCiphertextAutoload)
		.def("GetMultiplicativeDepth", &Params::GetMultiplicativeDepth)
		.def("GetBatchSize", &Params::GetBatchSize);

	// ---- CryptoContext ----

	py::class_<CC, std::shared_ptr<CC>>(m, "CryptoContext")
		// Setup
		.def("Enable", py::overload_cast<PKESchemeFeature>(&CC::Enable), py::arg("feature"))
		.def("GetCyclotomicOrder", &CC::GetCyclotomicOrder)
		.def("GetRingDimension", &CC::GetRingDimension)
		.def("SetDevices", &CC::SetDevices, py::arg("devices"))
		.def("SetAutoLoadPlaintexts", &CC::SetAutoLoadPlaintexts)
		.def("SetAutoLoadCiphertexts", &CC::SetAutoLoadCiphertexts)
		// Keys
		.def("KeyGen", &CC::KeyGen, nogil)
		.def("EvalMultKeyGen", &CC::EvalMultKeyGen, py::arg("privateKey"), nogil)
		.def("EvalRotateKeyGen", &CC::EvalRotateKeyGen, py::arg("privateKey"), py::arg("indexList"), nogil)
		// GPU loading
		.def("LoadContext", &CC::LoadContext, py::arg("publicKey"), nogil)
		.def("LoadPlaintext", &CC::LoadPlaintext, py::arg("plaintext"), nogil)
		.def("LoadCiphertext", &CC::LoadCiphertext, py::arg("ciphertext"), nogil)
		.def("Synchronize", &CC::Synchronize, nogil)
		// Encoding
		.def(
			"MakeCKKSPackedPlaintext",
			[](CC& cc, const std::vector<double>& value, size_t noiseScaleDeg, uint32_t level, uint32_t slots) {
				return cc.MakeCKKSPackedPlaintext(value, noiseScaleDeg, level, nullptr, slots);
			},
			py::arg("value"), py::arg("noiseScaleDeg") = 1, py::arg("level") = 0, py::arg("slots") = 0, nogil)
		.def(
			"MakeCKKSPackedPlaintextF64",
			[](CC& cc, py::array_t<double, py::array::c_style | py::array::forcecast> value, size_t noiseScaleDeg,
			   uint32_t level, uint32_t slots) {
				// Fast-path encode for float64 numpy vectors. The generic
				// overload above converts a Python list element-by-element
				// under the GIL (~15-20 ms for 2^16 slots), which serializes
				// multi-threaded encode pools to ~45 pts/s no matter the
				// thread count. Here the buffer copy is a memcpy; the GIL is
				// then released for the (expensive) encode itself.
				auto buf = value.request();
				if (buf.ndim != 1)
					throw std::runtime_error("MakeCKKSPackedPlaintextF64: expected a 1-D float64 array");
				const double* p = static_cast<const double*>(buf.ptr);
				std::vector<double> v(p, p + buf.shape[0]);
				py::gil_scoped_release release;
				return cc.MakeCKKSPackedPlaintext(v, noiseScaleDeg, level, nullptr, slots);
			},
			py::arg("value"), py::arg("noiseScaleDeg") = 1, py::arg("level") = 0, py::arg("slots") = 0)
		// Encrypt / Decrypt
		.def(
			"Encrypt", [](CC& cc, const PubK& pk, Pt pt) { return cc.Encrypt(pk, pt); }, py::arg("publicKey"),
			py::arg("plaintext"), nogil)
		.def(
			"Decrypt",
			[](CC& cc, const PrivK& sk, Ct& ct) {
				Pt pt;
				DecryptResult r = cc.Decrypt(sk, ct, &pt);
				if (!r.isValid)
					throw std::runtime_error("Decrypt: invalid result");
				return pt;
			},
			py::arg("privateKey"), py::arg("ciphertext"), nogil)
		// Add
		.def("EvalAdd", [](CC& cc, const Ct& a, const Ct& b) { return cc.EvalAdd(a, b); }, nogil)
		.def("EvalAdd", [](CC& cc, const Ct& a, Pt b) { return cc.EvalAdd(a, b); }, nogil)
		.def("EvalAdd", [](CC& cc, const Ct& a, double b) { return cc.EvalAdd(a, b); }, nogil)
		.def("EvalAdd", [](CC& cc, double a, const Ct& b) { return cc.EvalAdd(a, b); }, nogil)
		.def("EvalAddInPlace", [](CC& cc, Ct& a, const Ct& b) { cc.EvalAddInPlace(a, b); }, nogil)
		.def("EvalAddInPlace", [](CC& cc, Ct& a, Pt b) { cc.EvalAddInPlace(a, b); }, nogil)
		.def("EvalAddInPlace", [](CC& cc, Ct& a, double b) { cc.EvalAddInPlace(a, b); }, nogil)
		// Sub
		.def("EvalSub", [](CC& cc, const Ct& a, const Ct& b) { return cc.EvalSub(a, b); }, nogil)
		.def("EvalSub", [](CC& cc, const Ct& a, Pt b) { return cc.EvalSub(a, b); }, nogil)
		.def("EvalSub", [](CC& cc, const Ct& a, double b) { return cc.EvalSub(a, b); }, nogil)
		.def("EvalSub", [](CC& cc, double a, const Ct& b) { return cc.EvalSub(a, b); }, nogil)
		.def("EvalSubInPlace", [](CC& cc, Ct& a, const Ct& b) { cc.EvalSubInPlace(a, b); }, nogil)
		.def("EvalSubInPlace", [](CC& cc, Ct& a, double b) { cc.EvalSubInPlace(a, b); }, nogil)
		// Mult
		.def("EvalMult", [](CC& cc, const Ct& a, const Ct& b) { return cc.EvalMult(a, b); }, nogil)
		.def("EvalMult", [](CC& cc, const Ct& a, Pt b) { return cc.EvalMult(a, b); }, nogil)
		.def("EvalMult", [](CC& cc, const Ct& a, double b) { return cc.EvalMult(a, b); }, nogil)
		.def("EvalMult", [](CC& cc, double a, const Ct& b) { return cc.EvalMult(a, b); }, nogil)
		.def("EvalMultInPlace", [](CC& cc, Ct& a, Pt b) { cc.EvalMultInPlace(a, b); }, nogil)
		.def("EvalMultInPlace", [](CC& cc, Ct& a, double b) { cc.EvalMultInPlace(a, b); }, nogil)
		// Square / Negate
		.def("EvalSquare", &CC::EvalSquare, nogil)
		.def("EvalNegate", &CC::EvalNegate, nogil)
		// Rotations
		.def("EvalRotate", &CC::EvalRotate, py::arg("ciphertext"), py::arg("index"), nogil)
		.def("EvalRotateInPlace", &CC::EvalRotateInPlace, py::arg("ciphertext"), py::arg("index"), nogil)
		// Chebyshev
		.def(
			"EvalChebyshevSeries",
			[](CC& cc, const Ct& ct, std::vector<double> coeffs, double a, double b) {
				return cc.EvalChebyshevSeries(ct, coeffs, a, b);
			},
			py::arg("ciphertext"), py::arg("coefficients"), py::arg("a"), py::arg("b"), nogil)
		.def(
			"EvalChebyshevSeriesInPlace",
			[](CC& cc, Ct& ct, std::vector<double> coeffs, double a, double b) {
				cc.EvalChebyshevSeriesInPlace(ct, coeffs, a, b);
			},
			py::arg("ciphertext"), py::arg("coefficients"), py::arg("a"), py::arg("b"), nogil)
		// Rescale
		.def("Rescale", &CC::Rescale, nogil)
		.def("RescaleInPlace", &CC::RescaleInPlace, nogil)
		// Accumulation (replaces OpenFHE's EvalSum; needs rotation keys from
		// accumulate_rotation_indices(slots, stride)).
		.def("AccumulateSum", &CC::AccumulateSum, py::arg("ciphertext"), py::arg("slots"), py::arg("stride") = 1, nogil)
		.def(
			"AccumulateSumInPlace",
			[](CC& cc, Ct& ct, int slots, int stride) { cc.AccumulateSumInPlace(ct, slots, stride); },
			py::arg("ciphertext"), py::arg("slots"), py::arg("stride") = 1, nogil)
		// Bootstrapping
		.def("EvalBootstrapSetup", &CC::EvalBootstrapSetup, py::arg("levelBudget") = std::vector<uint32_t>{ 5, 4 },
			 py::arg("dim1") = std::vector<uint32_t>{ 0, 0 }, py::arg("slots") = 0, py::arg("correctionFactor") = 0,
			 py::arg("precompute") = true, py::arg("btsfirstboot") = false, nogil)
		.def("EvalBootstrapKeyGen", &CC::EvalBootstrapKeyGen, py::arg("privateKey"), py::arg("slots"), nogil)
		.def("EvalBootstrap", &CC::EvalBootstrap, py::arg("ciphertext"), py::arg("numIterations") = 1,
			 py::arg("precision") = 0, py::arg("prescaled") = false, nogil);

	// ---- Free functions ----

	m.def(
		"GenCryptoContext", [](Params& params) { return GenCryptoContext(params); }, py::arg("params"), nogil);

	m.def(
		"GetChebyshevCoefficients",
		[](py::function func, double a, double b, size_t degree) {
			std::function<double(double)> f = [&func](double x) { return func(x).cast<double>(); };
			return CC::GetChebyshevCoefficients(f, a, b, degree);
		},
		py::arg("func"), py::arg("a"), py::arg("b"), py::arg("degree"),
		"Chebyshev interpolation coefficients of a Python callable on [a, b].");

	m.def("accumulate_rotation_indices", &AccumulateRotationIndices, py::arg("slots"), py::arg("stride") = 1,
		  py::arg("bstep") = 4,
		  "Rotation indices required by AccumulateSum(ct, slots, stride). Pass them to EvalRotateKeyGen.");
}
