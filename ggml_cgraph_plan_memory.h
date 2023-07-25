#include "ggml.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_cgraph ggml_cgraph;
void ggml_cgraph_plan_memory(ggml_cgraph *cgraph, void ** intermediate_mem_buffer, size_t * buf_size);
void ggml_cgraph_schedule(ggml_cgraph * cgraph);

#ifdef __cplusplus
}
#endif
