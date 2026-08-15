# defines the export header for the cthreads hpp
# includes the export macro to turn CTHREADS_API into a dll export macro based on the platform
EXPORT_HPP = (
    "#pragma once\n\n"
    "#ifndef CTHREADS_API\n"
    "#  if defined(_WIN32)\n"
    '#    define CTHREADS_API extern "C" __declspec(dllexport)\n'
    "#  else\n"
    '#    define CTHREADS_API extern "C"\n'
    "#  endif\n"
    "#endif\n"
)
