#include "dram_kvtable.h"
#include <iostream>

int main() {
    DramKVTable myMap(400, 10, 10);

    myMap.add("apple");
    myMap.add("banana");
    myMap.add("cherry");

    std::cout << "Current map:\n";
    myMap.print();

    myMap.remove("banana");

    std::cout << "\nAfter removing 'banana':\n";
    myMap.print();

    std::cout << "\nValue of 'apple': ";
    auto val = myMap.get("apple");
    if (val.has_value()) {
        std::cout << val.value() << std::endl;
    } else {
        std::cout << "Not found\n";
    }

    // 新 key 会复用之前删除的 value
    myMap.add("date");
    std::cout << "\nAfter adding 'date':\n";
    myMap.print();

    return 0;
}