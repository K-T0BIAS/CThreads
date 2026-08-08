#pragma once

#include "pyThread.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace cthreads {

// OS detection used by the loader (and useful for callers / diagnostics).
#if defined(_WIN32)
inline constexpr const char* kHostOS = "windows";
#elif defined(__APPLE__)
inline constexpr const char* kHostOS = "macos";
#elif defined(__linux__)
inline constexpr const char* kHostOS = "linux";
#else
inline constexpr const char* kHostOS = "unknown";
#endif

// Loads cthreads_kernels (.dll / .so / .dylib) and resolves extern "C" symbols.
class KernelLib {
    void* _handle = nullptr;
    std::string _path;

public:
    KernelLib() = default;

    explicit KernelLib(const std::string& path) { load(path); }

    KernelLib(const KernelLib&) = delete;
    KernelLib& operator=(const KernelLib&) = delete;

    KernelLib(KernelLib&& other) noexcept
        : _handle(other._handle), _path(std::move(other._path)) {
        other._handle = nullptr;
    }

    KernelLib& operator=(KernelLib&& other) noexcept {
        if (this != &other) {
            close();
            _handle = other._handle;
            _path = std::move(other._path);
            other._handle = nullptr;
        }
        return *this;
    }

    ~KernelLib() { close(); }

    static const char* host_os() { return kHostOS; }

    bool loaded() const { return _handle != nullptr; }

    const std::string& path() const { return _path; }

    void load(const std::string& path) {
        close();
        _path = path;

#if defined(_WIN32)
        HMODULE mod = LoadLibraryA(path.c_str());
        if (!mod) {
            throw std::runtime_error(
                "KernelLib: LoadLibrary failed for '" + path +
                "' (GetLastError=" + std::to_string(GetLastError()) +
                ", os=" + kHostOS + ")"
            );
        }
        _handle = static_cast<void*>(mod);
#else
        // RTLD_NOW: fail fast on missing deps; RTLD_LOCAL: keep symbols private
        void* mod = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!mod) {
            const char* err = dlerror();
            throw std::runtime_error(
                std::string("KernelLib: dlopen failed for '") + path +
                "': " + (err ? err : "unknown") +
                " (os=" + kHostOS + ")"
            );
        }
        _handle = mod;
#endif
    }

    void close() {
        if (!_handle) {
            return;
        }
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(_handle));
#else
        dlclose(_handle);
#endif
        _handle = nullptr;
        _path.clear();
    }

    // Raw symbol lookup (nullptr if missing — check before cast).
    void* sym(const char* name) const {
        if (!_handle) {
            throw std::runtime_error("KernelLib: no library loaded");
        }
#if defined(_WIN32)
        return reinterpret_cast<void*>(
            GetProcAddress(static_cast<HMODULE>(_handle), name)
        );
#else
        dlerror(); // clear
        void* p = dlsym(_handle, name);
        return p;
#endif
    }

    // Typed lookup — throws if the export is missing.
    template <class Fn>
    Fn get(const char* name) const {
        void* p = sym(name);
        if (!p) {
            throw std::runtime_error(
                std::string("KernelLib: symbol not found: '") + name +
                "' in '" + _path + "' (os=" + kHostOS + ")"
            );
        }
        return reinterpret_cast<Fn>(p);
    }
};

// Process-wide kernel DLL (set after api.build() via load_kernels).
inline KernelLib& kernels() {
    static KernelLib lib;
    return lib;
}

inline void load_kernels(const std::string& path) {
    kernels().load(path);
}

// Look up extern "C" symbol `name` as Fn, bind args, return an unstarted CThread.
// Raises if the library is not loaded or the symbol is missing.
template <class Fn, class... Args>
std::unique_ptr<CThread> dispatch(KernelLib& lib, const char* name, Args... args) {
    if (!lib.loaded()) {
        throw std::runtime_error(
            std::string("dispatch: kernel library not loaded (os=") + kHostOS +
            ") — call load_kernels(BINARY_PATH) after build()"
        );
    }
    if (lib.sym(name) == nullptr) {
        throw std::runtime_error(
            std::string("dispatch: no compiled kernel for '") + name +
            "' in '" + lib.path() + "' (os=" + kHostOS + ")"
        );
    }
    Fn fn = lib.get<Fn>(name);
    return CThread::thread(fn, std::move(args)...);
}

template <class Fn, class... Args>
std::unique_ptr<CThread> dispatch(const char* name, Args... args) {
    return dispatch<Fn>(kernels(), name, std::move(args)...);
}

} // namespace cthreads
