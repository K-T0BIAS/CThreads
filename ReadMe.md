
![LOGO](./docs/__ressources/CTHREADS_03_1.svg)

----

CThreads solves pythons  thread concurrency issue by compiling python into C++.

The compilable python subset spans most python types aswell as user custom classes via the `@Threadable` class wrapper.

Code written using the cthread tool box can be multithreaded in a `GIL free` environment, which allows for true concurrency without the limitations of pythons `multiprocessing` library.

With a multitude of synchronization tools the python side main thread can be updated with low latency or even completely without blocking the C++ backend.

The internal math and linalg lib additionally provides most `math` and `tensor` functions that are commonly used in multithreading applications. On `SIMD` capable cpus the linalg module is further optimized for fast vector operation and thus provides a high performance linalg solution even outside of the multithreading usecase.

See [Math and linalg](./docs/guide/math_and_linalg.md) for package docs and NumPy benchmark numbers.
