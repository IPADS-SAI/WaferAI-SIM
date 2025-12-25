#ifndef SRAM_MANAGER_H
#define SRAM_MANAGER_H

#include <vector>
#include <map>
#include <list>
#include <cstdint> // For int, int
#include <stdexcept> // For std::invalid_argument
#include "macros/macros.h"
#include <fstream>
#include <string>
#include <iomanip>
#include <systemc>
// Optional: Define a type for allocation IDs for clarity
using AllocationID = float;

class SramManager {
public:
    // Constructor
    SramManager(int sram_start_address, int cid, int total_sram_size, int block_size, int num_blocks);

    // Allocate memory
    // Returns an AllocationID on success, or 0 on failure.
    AllocationID allocate(int requested_size);
    AllocationID allocate_append(int requested_size, AllocationID id);
    // Allocate memory with a specific AllocationID

    int get_address_with_offset(AllocationID id, int current_address, int offset_bytes) const;

    // Deallocate memory associated with an AllocationID
    bool deallocate(AllocationID id);
    void log_free_block_ratio() const;

    // Optional: Get the actual SRAM address for an allocation ID
    int get_address(AllocationID id) const;
    int get_address_index(AllocationID id) const;

    // Optional: Display current SRAM allocation status
    void display_status() const;

    // Optional: Get number of free blocks
    int get_free_blocks_count() const;

    // Optional: Get number of used blocks
    int get_used_blocks_count() const;


    // 在 SramManager 类的 public 部分添加：
    void printAllAllocationIDs() const;

    int get_allocation_byte_capacity(AllocationID id) const;

public:
    int sram_start_address_;
    int total_sram_size_;
    int block_size_;
    int num_blocks_;
    int cid;

    std::vector<bool> block_status_; // Tracks if a block is used (true) or free (false)
    // Stores {allocation_id -> vector_of_block_indices}
    std::map<AllocationID, std::vector<int>> allocations_;

    // List of indices of free blocks, kept sorted.
    std::list<int> free_block_indices_;

    AllocationID next_allocation_id_; // Simple way to generate unique IDs
};

#endif // SRAM_MANAGER_H