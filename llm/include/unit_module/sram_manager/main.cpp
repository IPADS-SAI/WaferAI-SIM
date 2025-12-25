#include "sram_manager.h"
#include <iostream>
#include <cstdint> // For int

int main() {
    try {
        int sram_base = 0;
        size_t total_size = 1024; // bytes
        size_t blk_size = 64;     // bytes

        SramManager manager(sram_base, total_size, blk_size);
        std::cout << "Initial state:\n";
        manager.display_status();

        AllocationID alloc1 = manager.allocate(128); // Needs 2 blocks
        if (alloc1) {
            std::cout << "Allocated 128 bytes with ID " << alloc1 << " at address 0x"
                      << std::hex << manager.get_address(alloc1) << std::dec << std::endl;
        } else {
            std::cout << "Failed to allocate 128 bytes.\n";
        }
        manager.display_status();

        AllocationID alloc2 = manager.allocate(60); // Needs 1 block
        if (alloc2) {
            std::cout << "Allocated 60 bytes with ID " << alloc2 << " at address 0x"
                      << std::hex << manager.get_address(alloc2) << std::dec << std::endl;
        } else {
            std::cout << "Failed to allocate 60 bytes.\n";
        }
        manager.display_status();

        AllocationID alloc3 = manager.allocate(200); // Needs 4 blocks (200/64 ceil = 4)
        if (alloc3) {
            std::cout << "Allocated 200 bytes with ID " << alloc3 << " at address 0x"
                      << std::hex << manager.get_address(alloc3) << std::dec << std::endl;
        } else {
            std::cout << "Failed to allocate 200 bytes.\n";
        }
        manager.display_status();

        std::cout << "Deallocating ID " << alloc1 << "...\n";
        if (manager.deallocate(alloc1)) {
            std::cout << "Deallocated ID " << alloc1 << std::endl;
        } else {
            std::cout << "Failed to deallocate ID " << alloc1 << std::endl;
        }
        manager.display_status();

        AllocationID alloc4 = manager.allocate(150); // Needs 3 blocks. Should try to reuse space.
        if (alloc4) {
            std::cout << "Allocated 150 bytes with ID " << alloc4 << " at address 0x"
                      << std::hex << manager.get_address(alloc4) << std::dec << std::endl;
        } else {
            std::cout << "Failed to allocate 150 bytes.\n";
        }
        manager.display_status();

        AllocationID alloc5 = manager.allocate(512); // Needs 8 blocks
        if (alloc5) {
             std::cout << "Allocated 512 bytes with ID " << alloc5 << " at address 0x"
                      << std::hex << manager.get_address(alloc5) << std::dec << std::endl;
        } else {
            std::cout << "Failed to allocate 512 bytes.\n";
        }
        manager.display_status();

        std::cout << "Total free blocks: " << manager.get_free_blocks_count() << std::endl;
        std::cout << "Total used blocks: " << manager.get_used_blocks_count() << std::endl;

        AllocationID alloc_too_big = manager.allocate(2048); // Too big
        if (alloc_too_big) {
            std::cout << "Allocated 2048 bytes with ID " << alloc_too_big << std::endl;
        } else {
            std::cout << "Failed to allocate 2048 bytes (expected failure).\n";
        }
        manager.display_status();

        std::cout << "Deallocating ID " << alloc2 << "...\n";
        manager.deallocate(alloc2);
        manager.display_status();

        std::cout << "Deallocating ID " << alloc3 << "...\n";
        manager.deallocate(alloc3);
        manager.display_status();

        std::cout << "Deallocating ID " << alloc4 << "...\n";
        manager.deallocate(alloc4);
        manager.display_status();

        std::cout << "Deallocating ID " << alloc5 << "...\n";
        manager.deallocate(alloc5);
        manager.display_status();

        std::cout << "Final Total free blocks: " << manager.get_free_blocks_count() << std::endl;
        std::cout << "Final Total used blocks: " << manager.get_used_blocks_count() << std::endl;


        // Test with zero total size
        std::cout << "\nTesting with zero total SRAM size:\n";
        SramManager zero_manager(sram_base, 0, blk_size);
        zero_manager.display_status();
        AllocationID zero_alloc = zero_manager.allocate(64);
        if (zero_alloc) {
            std::cout << "Allocated on zero_manager (error)!\n";
        } else {
            std::cout << "Correctly failed to allocate on zero_manager.\n";
        }
        AllocationID alloc_id = manager.allocate(1024); // 分配 1KB
        manager.display_status();
        if (alloc_id != 0) {
            int base_addr = manager.get_address(alloc_id); // 获取起始地址

            int new_addr = manager.get_address_with_offset(alloc_id, base_addr, 768);
            std::cout << "Address at offset 768: 0x" << std::hex << new_addr << std::dec << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}