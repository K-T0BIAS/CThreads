# Thread

Defines a wrapper for functions that have a clear defined execution policy and can be compiled to binary.

## Current codegen

Emits a declaration-only `.hpp` and an implementing `.cpp`.  
Annotated locals (`x: int = 1`) go in the `.cpp` body.  
Other body statements are left as `// unsupported ...` comments until expression lowering exists.

```py
@Thread
def move(p: Particle, dt: float) -> None:
    scale: float = 1.0
    steps: int = 10
```

`move.hpp`:

```cpp
#pragma once

#include "../__Threadable__/Particle.hpp"

void move(Particle& p, double dt);
```

`move.cpp`:

```cpp
#include "move.hpp"

void move(Particle& p, double dt) {
    double scale = 1.0;
    int steps = 10;
}
```
