#include "list.h"
#include <stdlib.h>

List new_list(size_t item_size) {
    return (List) { .item_size = item_size, };
}

void list_push(List * list, void * item) {
    if (list->capacity == 0) {
        list->items = malloc(list->item_size);
        list->capacity = 1;
    } else if (list->capacity <= list->count) {
        list->capacity *= 2;
        list->items = realloc(list->items, list->capacity * list->item_size);
    }

    list->items[list->count++] = item;
}

void * list_pop(List * list) {
    if (list->count == 0) {
        println("list_pop called on list with 0 items");
        exit(1);
    }

    return list->items[--list->count];
}

void * list_at(List list, size_t index) {
    if (!(0 <= index && index < list.count)) {
        println("List[{i}] index out of bounds: {i}", list.count, index);
        exit(1);
    }

    return list.items[index];
}
