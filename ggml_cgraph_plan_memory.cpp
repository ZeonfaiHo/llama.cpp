#include "ggml_cgraph_plan_memory.h"

#include <cstdlib>
#include <cstddef>
#include <unordered_map>
#include <algorithm>
#include <set>

struct ggml_mem_buffer_info {
    int id;
    size_t size;
    int begin;
    int end;
    size_t offset;
};


void ggml_cgraph_plan_memory(struct ggml_cgraph *cgraph) {
    int n_mem_buffer = 0;
    std::unordered_map<ggml_tensor *, int> tensor_to_mem_buffer_id;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_intermediate) {
            if (node->share_from == NULL) {
                tensor_to_mem_buffer_id.insert(
                    std::pair<ggml_tensor *, int>(node, n_mem_buffer));
                n_mem_buffer++;
            }
            else {
                tensor_to_mem_buffer_id.insert(
                    std::pair<ggml_tensor *, int>
                    (node, tensor_to_mem_buffer_id[node->share_from]));
            }
        }
    }

    ggml_mem_buffer_info *buf_infos = new ggml_mem_buffer_info[n_mem_buffer];
    for (int i = 0; i < n_mem_buffer; i++) {
        buf_infos[i].id = i;
        buf_infos[i].begin = (int) 1e9;
        buf_infos[i].end = -1;
    } 

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_intermediate) {
            int id = tensor_to_mem_buffer_id[node];
            buf_infos[id].size = node->data_size;
            buf_infos[id].begin = std::min(buf_infos[id].begin, i);
            buf_infos[id].end = std::max(buf_infos[id].end, i);
        }

        if (node->src0 && node->src0->is_intermediate) {
            int src0_id = tensor_to_mem_buffer_id[node->src0];
            buf_infos[src0_id].end = i;
        }
        if (node->src1 && node->src1->is_intermediate) {
            int src1_id = tensor_to_mem_buffer_id[node->src1];
            buf_infos[src1_id].end = i;
        }
    }

    // 对buf_infos进行排序，以begin和end分别作为第一第二关键字
    std::sort(buf_infos, buf_infos + n_mem_buffer, [](const ggml_mem_buffer_info & a, const ggml_mem_buffer_info & b) {
            if (a.begin != b.begin) {
                return a.begin < b.begin;
            }
            else {
                return a.end < b.end;
            }
        });

    size_t size_needed = 0;
    
    for (int i = 0; i < n_mem_buffer; i++) {
        std::set<std::pair<size_t, size_t>> mem_buffer_usage;
        for (int j = 0; j < i; j++) {
            if (buf_infos[j].end >= buf_infos[i].begin) {
                mem_buffer_usage.insert(
                    std::pair<size_t, size_t>
                    (buf_infos[j].offset, buf_infos[j].offset + buf_infos[j].size));
            }
        }

        size_t offset = 0;
        for (std::pair<size_t, size_t> usage : mem_buffer_usage) {
            if (usage.first - offset >= buf_infos[i].size) {
                break;
            }
            else {
                offset = usage.second;
            }
        }

        buf_infos[i].offset = offset;
        size_needed = std::max(size_needed, offset + buf_infos[i].size);
    }

    cgraph->mem_buffer = malloc(size_needed);
    cgraph->buf_size = size_needed;

    std::sort(buf_infos, buf_infos + n_mem_buffer, [](const ggml_mem_buffer_info & a, const ggml_mem_buffer_info & b) {
            return a.id < b.id;
        });
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_intermediate) {
            if (node->share_from != NULL) {
                node->data = node->share_from->data + node->share_offset;
            }
            else {
                const int id = tensor_to_mem_buffer_id[node];
                node->data = cgraph->mem_buffer + buf_infos[id].offset;
            }
        }
    }
}

void ggml_cgraph_plan_memory_naive(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_intermediate) {
            if (node->share_from != NULL) {
                node->data = node->share_from->data + node->share_offset;
            }
            else {
                node->data = malloc(node->data_size);
            }
        }
    }
}