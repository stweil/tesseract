#include <math.h>
#include "tesstypes.h"

#if defined(TFLOAT)

namespace tesseract {

TFloat fabs(TFloat x) {
  return fabsf(x);
}

TFloat log2(TFloat x) {
  return log2f(x);
}

TFloat sqrt(TFloat x) {
  return sqrtf(x);
}

}

#endif
