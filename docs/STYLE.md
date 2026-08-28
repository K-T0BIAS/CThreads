#### Style Guide

#### python rules 
1. variables should be typed when the type is defined (Any may be omitted). Types that would need a import for typehinting may also be ommited

```py
# Good
var_a1: str = "..."
var_b1: list[tuple[int, float]] = []

# Bad
var_a2 = "..."
var_b2 = [[1, 1.1]]
```

2. ` RULE 1` must be applied to function parameters except for `*args, **kwargs` and exceptions to `RULE 1`

3. Function return types must be type hinted (exceptions are the same as for  `RULE 1`)
4. Function names must not start with a capitalized letter
5. modules must conform to uniform naming conventions `a_b` or `aB`. convention may differ accross modules
6. Class names must be capitalized
7. Function docstrings must:
    7.1. Explain the purpose and behaviour of the function in simple, explicit and intuitive terms.
    7.2. If user facing must contain an example. (Optional for non user facing functions)
    7.3. Follow this shape:

```py
def example_fn() -> None:
    """
    Function that does x.
    [Optional] mutates data in some way y
    
    #### Args:
    - argName: type = explanation (default) [example for complex deps]

    #### Returns
    - type = explanation
    [Optional] Schema for nested or complex types

    [Optional]
    #### Example:
    ``py
    ... full code example + return schema
    ``
    [Only if technical terms where used in this doc string]
    #### Technical terms:
    - term: eplanation
    """
    pass
```
8. when using markdown like syntax in docstrings use `` not ```` 

#### global style rules

1. files should include one logical unit (i.e. one class [optinally helper classes + funcs if they are unique to this file])
2. functions should be self contained unless code is reused (i.e. a function should not be the only one calling another function. In that case they should be merged into one. [counter example cross product: calls vec.norm however the norm logic should be sepperate since .norm is also a user facing api])
3. complex logical sections should be commented (if needed line by line) [more comments are generally preferred over too little]
4. technical terms must be followed by short, concise explanations (see py docstring example [here](#python-rules)) a technical term is only considered such if it is redefined by the project- Globally used terms DO NOT need an explanation

#### Commenting rules

1. a comment must not use unexplained abreviations
2. a comment should be coherend and form a valid sentence (for trivial comments)
3. depending on the difficulty of a section the comments should decribe both function and reason (i.e. when doing abstract indexing in complex buffer structures (like in cuda programming) the comment should describe: waht is indexed, why / for what purpose and depending on complexity and context where the index comes from. Naturally this is not bound to the example but instead must be decided depending on the situation)
4. docstrings must follow the structure in `Python rules RULE 7` but must be adapted to the languages commenting structure
5. classes should have a docstring aswell as methods

#### Documentation rules

1. all documentation must be written in valid sentences
2. all documentation must be written in reasonably simple language
3. all documentation should be intuitive
4. all documentation should be self explanatory (when dependencies in docs arise they must be linked every section when used)
5. abreviations must be written out atleast once per section (when first used)

#### Additional rules for ai agents

1. the use of special utf characters is not permitted
```latex
Exmaple:
— should be - or depending on ctx ,. etc.
→ should be ->
```
