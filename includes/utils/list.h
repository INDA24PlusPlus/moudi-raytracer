#pragma once

#include "common.h"

typedef struct list {
    void ** items;
    size_t capacity;
    size_t count;
    size_t item_size;
} List;

List new_list(size_t item_size);

void list_push(List * list, void * item);
void * list_pop(List * list);

void * list_at(List list, size_t index);
