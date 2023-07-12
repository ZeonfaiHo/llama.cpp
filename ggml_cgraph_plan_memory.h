#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_cgraph ggml_cgraph;
void ggml_cgraph_plan_memory(struct ggml_cgraph *cgraph);
void ggml_cgraph_plan_memory_naive(struct ggml_cgraph *cgraph);

#ifdef __cplusplus
}
#endif
