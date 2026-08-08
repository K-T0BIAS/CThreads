# Threadable

Defines a class wrapper that enfores classes with predictable memory layouts. [Thread](../Thread/DOC.md) wrapped functions must operate on `Threadable` classes.

## Example

```py
@Threadable
class Particle:
    x: int
    y: int
    velocity: float
```

becomes

```c++
typedef struct {
    double x;
    double y;
    double velocity;
} Particle;
```

This object is then usable in a [Thread](../Thread/DOC.md) wrapped function like:

```py
@Thread
def move(p: Particle, dt: float):
    p.x += p.velocity * dt
```

which is then compiled to c++ code as:

```cpp
void move(Particle& p, double dt) {
    p.x += p.velocity * dt;
}
```

## Threadble DTypes

Thes python data types can be used in a Threadable class since they have well defined c++ equivalents

### Easy

```py
int
float
str
bool
arrays like (int[] <-> list[int])
```

### potentially allowed when types are well defined

```
dict -> unordered_map<>
set  -> unordered_set<>
```

### Not Allowed

```
class (Except other @Threadable ones)
Callable (except other @Thread wrapped ones)
```
