#include "ggml_cgraph_plan_memory.h"

#include <cstdlib>
#include <cstddef>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <cmath>
#include <unordered_set>
#include <vector>

struct ggml_mem_buffer_info {
    int id;
    size_t size;
    int begin;
    int end;
    size_t offset;
    bool allocated;
};

void update_ref_count(ggml_tensor * tensor, std::unordered_map<ggml_tensor *, int>& ref_count, int delta) {
    if (tensor == NULL) {
        return;
    }
    
    if (ref_count.find(tensor) == ref_count.end()) {
        ref_count[tensor] = 0;
    }

    ref_count[tensor] += delta;
}

float calculate_benefit(ggml_tensor * tensor, std::unordered_map<ggml_tensor *, int>& ref_count, std::unordered_set<ggml_tensor *>& scheduled, float alpha) {
    float benefit = tensor->is_deferred && tensor->share_from != NULL ? (-tensor->data_size) : 0;
    // float benefit = 0;

    std::vector<ggml_tensor *> sources = { tensor->src0, tensor->src1 };

    for (int i = 0; i < GGML_MAX_OPT; i++) {
        sources.push_back(tensor->opt[i]);
    }

    for (ggml_tensor * src : sources) {
        if (src != NULL) {
            if (scheduled.find(src) == scheduled.end()) {
                return -2e18;
            }
            else 
            if (src->is_deferred) 
            {
                benefit += alpha * src->data_size / pow(alpha, ref_count[src]);
                // benefit += 1 / pow(alpha, ref_count[src]);
            }
        }
    }

    return benefit;
}

void ggml_cgraph_schedule(ggml_cgraph * cgraph) {

    const float alpha = 2.0;
    const int n_nodes = cgraph->n_nodes;

    std::unordered_map<ggml_tensor *, int> ref_count;
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * tensor = cgraph->nodes[i];
        update_ref_count(tensor->src0, ref_count, 1);
        update_ref_count(tensor->src1, ref_count, 1);
        for (int j = 0; j < GGML_MAX_OPT; j++) {
            update_ref_count(tensor->opt[j], ref_count, 1);
        }
    }

    ggml_tensor **sch = new ggml_tensor * [n_nodes];
    std::unordered_set<ggml_tensor *> scheduled;

    for (int i = 0; i < n_nodes; i++) {
        float best_benefit = -1e18;
        ggml_tensor * choice = NULL;
        for (int j = 0; j < n_nodes; j++) {
            ggml_tensor * tensor = cgraph->nodes[j];
            
            if (scheduled.find(tensor) == scheduled.end()) {
                float benefit = calculate_benefit(tensor, ref_count, scheduled, alpha);
                if (benefit > best_benefit) {
                    best_benefit = benefit;
                    choice = tensor;
                }
            }
        }

        GGML_ASSERT(choice != NULL);

        sch[i] = choice;
        scheduled.insert(choice);

        update_ref_count(choice->src0, ref_count, -1);
        update_ref_count(choice->src1, ref_count, -1);
        for (int j = 0; j < GGML_MAX_OPT; j++) {
            update_ref_count(choice->opt[j], ref_count, -1);
        }
    }

    for (int i = 0; i < n_nodes; i++) {
        cgraph->nodes[i] = sch[i];
    }

    delete[] sch;
}

void ggml_cgraph_plan_memory(ggml_cgraph *cgraph, void ** intermediate_mem_buffer, size_t * buf_size) {

    // 为与常量共享空间的张量分配空间
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->share_from != NULL 
            && node->share_from->data != NULL) {
            node->data = (char *) node->share_from->data + node->share_offset;
            node->is_deferred = false;
        }
    }

    int n_mem_buffer = 0;
    std::unordered_map<ggml_tensor *, int> tensor_to_mem_buffer_id;

    size_t mem_sum = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_deferred) {
            if (node->share_from == NULL) {
                tensor_to_mem_buffer_id[node] =  n_mem_buffer;
                n_mem_buffer++;

                mem_sum += node->data_size;
            }
            else {
                tensor_to_mem_buffer_id[node] = tensor_to_mem_buffer_id[node->share_from];
            }
        }
    }

    ggml_mem_buffer_info *buf_infos = new ggml_mem_buffer_info[n_mem_buffer];
    for (int i = 0; i < n_mem_buffer; i++) {
        buf_infos[i].id = i;
        buf_infos[i].begin = (int) 1e9;
        buf_infos[i].end = -1;
        buf_infos[i].allocated = false;
    } 

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_deferred) {
            int id = tensor_to_mem_buffer_id[node];
            buf_infos[id].size = node->data_size;
            buf_infos[id].begin = std::min(buf_infos[id].begin, i);
            buf_infos[id].end = std::max(buf_infos[id].end, i);
        }

        if (node->src0 && node->src0->is_deferred) {
            int src0_id = tensor_to_mem_buffer_id[node->src0];
            buf_infos[src0_id].end = i;
        }
        if (node->src1 && node->src1->is_deferred) {
            int src1_id = tensor_to_mem_buffer_id[node->src1];
            buf_infos[src1_id].end = i;
        }

        for (int j = 0; j < GGML_MAX_OPT; j++) {
            if (node->opt[j] != NULL && node->opt[j]->is_deferred) {
                int src_id = tensor_to_mem_buffer_id[node->opt[j]];
                buf_infos[src_id].end = i;
            }
        }
    }

    // 对buf_infos进行排序，以持续时间作为第一关键字
    std::sort(buf_infos, buf_infos + n_mem_buffer, [](const ggml_mem_buffer_info & a, const ggml_mem_buffer_info & b) {
            return a.end - a.begin > b.end - b.begin;
        });

    size_t size_needed = 0;
    size_t offset = 0;
    
    while (true) {
        bool finished = true;
        for (int i = 0; i < n_mem_buffer; i++) {
            if (!buf_infos[i].allocated) {
                finished = false;
                break;
            }
        }

        if (finished) break;

        for (int i = 0; i < n_mem_buffer; i++) {
            if (!buf_infos[i].allocated) {
                bool conflict = false;

                for (int j = 0; j < n_mem_buffer; j++) {
                    if (buf_infos[j].allocated && 
                        std::max(buf_infos[i].begin, buf_infos[j].begin) 
                        <= 
                        std::min(buf_infos[i].end, buf_infos[j].end) &&
                        std::max(offset, buf_infos[j].offset)
                        <
                        std::min(offset + buf_infos[i].size,
                                 buf_infos[j].offset + buf_infos[j].size)) {
                        
                        conflict = true;
                        break;
                    }
                }

                if (!conflict) {
                    buf_infos[i].offset = offset;
                    buf_infos[i].allocated = true;
                    size_needed = std::max(size_needed, buf_infos[i].offset + buf_infos[i].size);
                }
            }
        }

        size_t next = (size_t)1e18;

        for (int i = 0; i < n_mem_buffer; i++) {
            size_t tmp = buf_infos[i].offset + buf_infos[i].size;
            if (buf_infos[i].allocated && tmp > offset) {
                next = std::min(next, tmp);
            }
        }

        offset = next;
    }

    if (size_needed > *buf_size) {
        free(*intermediate_mem_buffer);
        *buf_size = size_needed + 1024;
        *intermediate_mem_buffer = malloc(*buf_size);
    }

    cgraph->mem_buffer = *intermediate_mem_buffer;
    cgraph->buf_size = size_needed;

    std::sort(buf_infos, buf_infos + n_mem_buffer, [](const ggml_mem_buffer_info & a, const ggml_mem_buffer_info & b) {
            return a.id < b.id;
        });
    
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_deferred) {
            if (node->share_from != NULL) {
                node->data = (char *) node->share_from->data + node->share_offset;
            }
            else {
                const int id = tensor_to_mem_buffer_id[node];
                node->data = (char *) cgraph->mem_buffer + buf_infos[id].offset;
            }
            node->is_deferred = false;
        }
    }

    delete[] buf_infos;
}

void ggml_cgraph_plan_memory_greedy(ggml_cgraph *cgraph) {
    // 为与常量共享空间的张量分配空间
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->share_from != NULL 
            && node->share_from->data != NULL) {
            node->data = (char *) node->share_from->data + node->share_offset;
            node->is_deferred = false;
        }
    }

    int n_mem_buffer = 0;
    std::unordered_map<ggml_tensor *, int> tensor_to_mem_buffer_id;

    size_t mem_sum = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_deferred) {
            if (node->share_from == NULL) {
                tensor_to_mem_buffer_id.insert(
                    std::pair<ggml_tensor *, int>(node, n_mem_buffer));
                n_mem_buffer++;

                mem_sum += node->data_size;
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
        if (node->is_deferred) {
            int id = tensor_to_mem_buffer_id[node];
            buf_infos[id].size = node->data_size;
            buf_infos[id].begin = std::min(buf_infos[id].begin, i);
            buf_infos[id].end = std::max(buf_infos[id].end, i);
        }

        if (node->src0 && node->src0->is_deferred) {
            int src0_id = tensor_to_mem_buffer_id[node->src0];
            buf_infos[src0_id].end = i;
        }
        if (node->src1 && node->src1->is_deferred) {
            int src1_id = tensor_to_mem_buffer_id[node->src1];
            buf_infos[src1_id].end = i;
        }

        for (int j = 0; j < GGML_MAX_OPT; j++) {
            if (node->opt[j] != NULL) {
                int src_id = tensor_to_mem_buffer_id[node->opt[j]];
                buf_infos[src_id].end = i;
            }
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
        if (node->is_deferred) {
            if (node->share_from != NULL) {
                node->data = (char *) node->share_from->data + node->share_offset;
            }
            else {
                const int id = tensor_to_mem_buffer_id[node];
                node->data = (char *) cgraph->mem_buffer + buf_infos[id].offset;
            }
            node->is_deferred = false;
        }
    }
}

void ggml_cgraph_plan_memory_naive(ggml_cgraph * cgraph) {
    size_t size_needed = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_deferred && node->share_from == NULL) {
            size_needed += node->data_size;
        }
    }

    cgraph->mem_buffer = malloc(size_needed);
    cgraph->buf_size = size_needed;

    size_t end = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor *node = cgraph->nodes[i];
        if (node->is_deferred) {
            if (node->share_from != NULL) {
                node->data = (char *) node->share_from->data + node->share_offset;
            }
            else {
                node->data = (char *) cgraph->mem_buffer + end;
                end += node->data_size;
            }
        }
    }
}
