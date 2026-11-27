#pragma once

// Compatibility for inline logging sites in the packaged public headers.
// Logging compiled into libknowhere continues to use its own backend.
#include <iostream>
#define LOG(level) std::clog
#define DLOG(level) std::clog
