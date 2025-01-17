#include "list.h"

#include "gpu.h"

#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>

List new_list(size_t item_size) {
    return (List) { .item_size = item_size, };
}

void list_push(List * list, void * item) {
    if (list->capacity == 0) {
        list->capacity = 1;
        list->items = (void **) malloc(list->capacity * list->item_size);
    } else if (list->capacity <= list->count) {
        list->capacity *= 2;
        list->items = (void **) realloc(list->items, list->capacity * list->item_size);
    }

    list->items[list->count++] = item;
}

void * list_pop(List * list) {
    if (list->count == 0) {
        puts("list_pop called on list with 0 items");
        exit(1);
    }

    return list->items[--list->count];
}

__device__ void * list_at(List list, size_t index) {
    if (!(0 <= index && index < list.count)) {
        printf("List[%d] index out of bounds: %d\n", list.count, index);
        asm("exit;");
    }

    return list.items[index];
}
