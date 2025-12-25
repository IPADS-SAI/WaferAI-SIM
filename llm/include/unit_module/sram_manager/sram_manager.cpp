#include "sram_manager.h"
#include <numeric>   // For std::iota
#include <iostream>  // For basic output/debugging
#include <algorithm> // For std::sort, std::remove_if on list (indirectly)
#include <cassert>

// Constructor
SramManager::SramManager(int sram_start_address, int cid, int total_sram_size, int block_size, int num_blocks)
    : sram_start_address_(sram_start_address),
      cid(cid),
      total_sram_size_(total_sram_size),
      block_size_(block_size),
      num_blocks_(num_blocks), // Initialize num_blocks_
      next_allocation_id_(1) {

    if (block_size == 0) {
        throw std::invalid_argument("Block size cannot be zero.");
    }
    if (total_sram_size > 0 && total_sram_size < block_size) {
        // If total size is not zero but less than block size, it effectively means 0 usable blocks.
        // Or one could argue this is an invalid configuration.
        // Here, we allow it but num_blocks_ will be 0.
    }
    if (total_sram_size == 0) {
        num_blocks_ = 0;
        // block_status_ and free_block_indices_ will be empty, which is fine.
        return;
    }

    // We can only manage full blocks.
    // If total_sram_size is not a multiple of block_size, the remaining part is unusable.
    num_blocks_ = total_sram_size_ / block_size_;

    if (num_blocks_ > 0) {
        block_status_.resize(num_blocks_, false); // All blocks initially free

        free_block_indices_.resize(num_blocks_);
        std::iota(free_block_indices_.begin(), free_block_indices_.end(), 0); // Fill with 0, 1, 2...
    }
    assert(num_blocks_ >= 0 && "num_blocks_ must not be negative");
    // std::cout << "num_blocks_: " << num_blocks_ << std::endl;
    // If num_blocks_ is 0, vectors/lists remain empty.
}


void SramManager::log_free_block_ratio() const {
    // Calculate ratio of free blocks to total blocks
    double ratio = (num_blocks_ > 0) ? static_cast<double>(num_blocks_ - free_block_indices_.size()) / num_blocks_ : 0.0;
    
    // Create log filename based on cid
    std::string filename = "sram_util/sram_manager_cid_" + std::to_string(cid) + ".log";
    
    // Open file in append mode
    std::ofstream log_file(filename, std::ios_base::app);
    
    if (log_file.is_open()) {
        // Write timestamp and ratio to log file
        log_file << "Time: " << sc_core::sc_time_stamp().to_string() 
                 << " Free/Total Ratio: " << std::fixed << std::setprecision(4) << ratio 
                 << " (" << num_blocks_ - free_block_indices_.size() << "/" << num_blocks_ << ")\n";
        log_file.close();
    } else {
        std::cerr << "Error: Could not open log file " << filename << " for writing.\n";
    }
}
// Allocate memory
AllocationID SramManager::allocate(int requested_size) {


    if (requested_size == 0 || num_blocks_ == 0) {

    if (requested_size == 0) {
        std::cerr << "Allocation failed: requested_size is zero." << std::endl;
        assert(false);
        return 0;
    }

    if (num_blocks_ == 0) {
        std::cerr << "Allocation failed: num_blocks_ is zero (no available blocks)." << std::endl;
        assert(false);
        return 0;
    }    
}

    int blocks_needed = (requested_size + block_size_ - 1) / block_size_; // Ceiling division
#if DEBUG_SRAM_MANAGER == 1
    std::cout << "blocks_needed: " << blocks_needed << std::endl;
#endif    
    if (blocks_needed == 0 || blocks_needed > num_blocks_ || free_block_indices_.size() < blocks_needed) {
        if (blocks_needed == 0) {
        std::cerr << "Allocation failed: blocks_needed is zero (requested_size may be too small)." << std::endl;
        assert(false);
        return 0;
    }

    if (blocks_needed > num_blocks_) {
        std::cerr << "Allocation failed: blocks_needed exceeds total available blocks (" 
                << blocks_needed << " > " << num_blocks_ << ")." << std::endl;
        assert(false);
        return 0;
    }

    if (free_block_indices_.size() < blocks_needed) {
        std::cerr << "Allocation failed: not enough free blocks available (" 
                << free_block_indices_.size() << " < " << blocks_needed << ")." << std::endl;
        assert(false);
        return 0;
    }
        return 0; // Cannot allocate if no blocks needed, more than total, or not enough free blocks overall
    }

    // First-Fit approach using the sorted free_block_indices_ list
    auto it = free_block_indices_.begin();
    std::vector<int> potential_blocks;
    int blocks_needed_tmp = blocks_needed;
    while (it != free_block_indices_.end()) {
        
        int first_block_idx = *it;
#if DEBUG_SRAM_MANAGER == 1
        std::cout << "first_block_idx: " << first_block_idx << std::endl;
#endif
        potential_blocks.push_back(first_block_idx);

        // Try to find contiguous blocks
        it++;
        for (int k = 1; k < blocks_needed_tmp; ++k) {
            if (it != free_block_indices_.end() && *it == first_block_idx + static_cast<int>(k)) {
                potential_blocks.push_back(*it);
                it++;
            } else {
                break; // Not enough contiguous blocks starting from *it
            }
        }
#if DEBUG_SRAM_MANAGER == 1
        // Print potential_blocks for debugging
        std::cout << "potential_blocks: ";
        for (int block : potential_blocks) {
            std::cout << block << " ";
        }
        std::cout << std::endl;
#endif

        if (potential_blocks.size() == blocks_needed) {
            // Found a fit!
            for (int block_idx : potential_blocks) {
                if (static_cast<int>(block_idx) < num_blocks_) { // Boundary check
                    block_status_[block_idx] = true;
                } else {
                     // Should not happen if logic is correct and num_blocks_ is set right
                    std::cerr << "Error: Trying to mark out-of-bounds block " << block_idx << " as used." << std::endl;
                    assert(false);
                    return 0; // Critical error
                }
            }

            // Remove allocated blocks from free_block_indices_
            for (int block_idx_to_remove : potential_blocks) {
                // std::list::remove is O(N), doing it multiple times is not ideal.
                // A more optimized way would be to use list::erase with iterators if we knew them,
                // or rebuild the list without these elements.
                // For this example, list::remove is clear.
                free_block_indices_.remove(block_idx_to_remove);
            }
            AllocationID id = next_allocation_id_++;
            std::cout << "\033[1;31m" << "ID ALLOC " << id << "\033[0m" << std::endl;

            allocations_[id] = potential_blocks;

            return id;
        }else{

            blocks_needed_tmp = blocks_needed_tmp - potential_blocks.size();
        }
        // If not a fit, move to the next block in free_block_indices_
        // The block *it might be part of a smaller free segment.
        // We need to advance 'it' to the start of the next potential free segment.
        // The current 'it' was the start of the segment we just checked.
        // If potential_blocks.size() < blocks_needed, we continue from the *next* free block.
        // it++;
    }

    assert(false && "No suitable contiguous block sequence found");
    log_free_block_ratio();
    return 0; // No suitable contiguous block sequence found
}



AllocationID SramManager::allocate_append(int requested_size, AllocationID id) {


    if (requested_size == 0 || num_blocks_ == 0) {

    if (requested_size == 0) {
        std::cerr << "Allocation failed: requested_size is zero." << std::endl;
        assert(false);
        return 0;
    }

    if (num_blocks_ == 0) {
        std::cerr << "Allocation failed: num_blocks_ is zero (no available blocks)." << std::endl;
        assert(false);
        return 0;
    }    
}

    int blocks_needed = (requested_size + block_size_ - 1) / block_size_; // Ceiling division
#if DEBUG_SRAM_MANAGER == 1
    std::cout << "blocks_needed: " << blocks_needed << std::endl;
#endif    
    if (blocks_needed == 0 || blocks_needed > num_blocks_ || free_block_indices_.size() < blocks_needed) {
        if (blocks_needed == 0) {
        std::cerr << "Allocation failed: blocks_needed is zero (requested_size may be too small)." << std::endl;
        assert(false);
        return 0;
    }

    if (blocks_needed > num_blocks_) {
        std::cerr << "Allocation failed: blocks_needed exceeds total available blocks (" 
                << blocks_needed << " > " << num_blocks_ << ")." << std::endl;
        assert(false);
        return 0;
    }

    if (free_block_indices_.size() < blocks_needed) {
        std::cerr << "Allocation failed: not enough free blocks available (" 
                << free_block_indices_.size() << " < " << blocks_needed << ")." << std::endl;
        assert(false);
        return 0;
    }
        return 0; // Cannot allocate if no blocks needed, more than total, or not enough free blocks overall
    }

    // First-Fit approach using the sorted free_block_indices_ list
    auto it = free_block_indices_.begin();
    std::vector<int> potential_blocks;
    int blocks_needed_tmp = blocks_needed;
    while (it != free_block_indices_.end()) {
        
        int first_block_idx = *it;
#if DEBUG_SRAM_MANAGER == 1
        std::cout << "first_block_idx: " << first_block_idx << std::endl;
#endif
        potential_blocks.push_back(first_block_idx);

        // Try to find contiguous blocks
        it++;
        for (int k = 1; k < blocks_needed_tmp; ++k) {
            if (it != free_block_indices_.end() && *it == first_block_idx + static_cast<int>(k)) {
                potential_blocks.push_back(*it);
                it++;
            } else {
                break; // Not enough contiguous blocks starting from *it
            }
        }
#if DEBUG_SRAM_MANAGER == 1
        // Print potential_blocks for debugging
        std::cout << "potential_blocks: ";
        for (int block : potential_blocks) {
            std::cout << block << " ";
        }
        std::cout << std::endl;
#endif

        if (potential_blocks.size() == blocks_needed) {
            // Found a fit!
            for (int block_idx : potential_blocks) {
                if (static_cast<int>(block_idx) < num_blocks_) { // Boundary check
                    block_status_[block_idx] = true;
                } else {
                     // Should not happen if logic is correct and num_blocks_ is set right
                    std::cerr << "Error: Trying to mark out-of-bounds block " << block_idx << " as used." << std::endl;
                    assert(false);
                    return 0; // Critical error
                }
            }

            // Remove allocated blocks from free_block_indices_
            for (int block_idx_to_remove : potential_blocks) {
                // std::list::remove is O(N), doing it multiple times is not ideal.
                // A more optimized way would be to use list::erase with iterators if we knew them,
                // or rebuild the list without these elements.
                // For this example, list::remove is clear.
                free_block_indices_.remove(block_idx_to_remove);
            }
            // AllocationID id = next_allocation_id_++;
            std::cout << "\033[1;31m" << "ID ALLOC " << id << "\033[0m" << std::endl;

            allocations_[id].insert(allocations_[id].end(), potential_blocks.begin(), potential_blocks.end());

            return id;
        }else{

            blocks_needed_tmp = blocks_needed_tmp - potential_blocks.size();
        }
        // If not a fit, move to the next block in free_block_indices_
        // The block *it might be part of a smaller free segment.
        // We need to advance 'it' to the start of the next potential free segment.
        // The current 'it' was the start of the segment we just checked.
        // If potential_blocks.size() < blocks_needed, we continue from the *next* free block.
        // it++;
    }
    log_free_block_ratio();
    assert(false && "No suitable contiguous block sequence found");
    return 0; // No suitable contiguous block sequence found
}

// Deallocate memory
bool SramManager::deallocate(AllocationID id) {
    if (id == 0) return false; // Invalid ID

    auto alloc_it = allocations_.find(id);
    // std::cout << "Current Allocation IDs:" << std::endl;
    // for (const auto& pair : allocations_) {
    //     std::cout << "Allocation ID: " << pair.first << std::endl;
    // }
    std::cout << "deallocate id "  << id << std::endl;
    if (alloc_it == allocations_.end()) {
        std::cerr << "Error: Allocation ID not found!" << std::endl;
        exit(EXIT_FAILURE); // 主动终止
        return false; // Allocation ID not found
    }

    const std::vector<int>& blocks_to_free = alloc_it->second;
    bool all_valid = true;
    for (int block_idx : blocks_to_free) {
        if (block_idx >= 0 && static_cast<int>(block_idx) < num_blocks_) {
            if (block_status_[block_idx]) { // Only free if it was marked as used
                block_status_[block_idx] = false;
                // Add back to free_block_indices_. It will be sorted later.
                free_block_indices_.push_back(block_idx);
            } else {
                // This might indicate a double free or logic error
                std::cerr << "Warning: Block " << block_idx << " for ID " << id << " was already free." << std::endl;
                assert(false);
            }
        } else {
            std::cerr << "Error: Trying to free invalid block index " << block_idx << " for ID " << id << std::endl;
            all_valid = false; // Mark that an error occurred but try to free others
            assert(false);
        }
    }

    // Keep free_block_indices_ sorted for the allocation strategy
    free_block_indices_.sort();
    // Optional: remove duplicates if any arose from error conditions (should not happen in normal op)
    // free_block_indices_.unique();

    allocations_.erase(alloc_it);
    log_free_block_ratio();
    // printAllAllocationIDs();
    return all_valid; // Return true if all specified blocks were valid and processed,
                      // false if any block_idx was out of bounds.
}

// Get address
int SramManager::get_address(AllocationID id) const {
    auto it = allocations_.find(id);
    if (it != allocations_.end() && !it->second.empty()) {
        // Ensure block_size_ is not zero to prevent division by zero if somehow it was set to 0
        // after construction (though it's const, good practice for such calculations)
        if (block_size_ == 0) return 0;
        return sram_start_address_ + (static_cast<int>(it->second[0]) * block_size_);
    }
    return 0; // Invalid ID or empty allocation (should not happen for valid ID)
}


int SramManager::get_allocation_byte_capacity(AllocationID id) const {
    auto it = allocations_.find(id);
    if (it == allocations_.end()) {
        std::cerr << "\033[1;31mError: Allocation ID " << id << " not found in allocations_.\033[0m" << std::endl;
        assert(false && "Allocation ID not found");
        return 0; // Invalid ID
    } else if (it->second.empty()) {
        std::cerr << "\033[1;31mError: Allocation for ID " << id << " is empty (no associated blocks).\033[0m" << std::endl;
        assert(false && "Allocation is empty");
        return 0; // Empty allocation
    }
    if (block_size_ == 0) {
        std::cerr << "Error: Block size is zero." << std::endl;
        assert(false);
        return 0;
    }

    const std::vector<int>& block_indices = it->second;
    return block_indices.size() * block_size_;
}
int SramManager::get_address_index(AllocationID id) const {
    auto it = allocations_.find(id);
    if (it != allocations_.end() && !it->second.empty()) {
        // Ensure block_size_ is not zero to prevent division by zero if somehow it was set to 0
        // after construction (though it's const, good practice for such calculations)
        if (block_size_ == 0) return 0;
        return (sram_start_address_ + (static_cast<int>(it->second[0]) * block_size_))  * 8 / SRAM_BITWIDTH;
    }
    return 0; // Invalid ID or empty allocation (should not happen for valid ID)
}

void SramManager::printAllAllocationIDs() const {
    std::cout << "[SRAM Manager] All active Allocation IDs:\n";
    if (allocations_.empty()) {
        std::cout << "  <No active allocations>\n";
        return;
    }

    for (const auto& pair : allocations_) {
        std::cout << "  Allocation ID: " << pair.first << "\n";
    }
}

int SramManager::get_address_with_offset(AllocationID id, int current_address, int offset_bytes) const {
    auto it = allocations_.find(id);
    // std::cout << "get_address_with_offset id " << id << std::endl;
    if (it == allocations_.end() || it->second.empty()) {
        if (it == allocations_.end()) {

            std::cerr << "Error: Allocation ID not found in map." << std::endl;
        } else if (it->second.empty()) {
            std::cerr << "Error: Allocation for ID is empty (no associated blocks)." << std::endl;
        }

        
        std::cerr << "Error: Invalid AllocationID or empty allocation." << std::endl;
        assert(false);
        return 0; // 无效 ID
    }

    if (block_size_ == 0) {
        std::cerr << "Error: Block size is zero." << std::endl;
        assert(false);
        return 0;
    }

    const std::vector<int>& block_indices = it->second;

    // Step 1: Find which block contains the current_address
    int base_block_index = -1;
    for (int i = 0; i < block_indices.size(); ++i) {
        int block_idx = block_indices[i];
        uintptr_t block_base = sram_start_address_ + block_idx * block_size_;
        uintptr_t block_end = block_base + block_size_;

        if (current_address >= block_base && current_address < block_end) {
            base_block_index = static_cast<int>(i);
            break;
        }
    }

    if (base_block_index == -1) {
        std::cout << "get_address_with_offset id " << id << std::endl;
        std::cout << "current_address: " << current_address << std::endl;
        std::cerr << "Error: Current address not within allocation range." << std::endl;
        assert(false);
        return 0;
    }

    // Step 2: Now walk through blocks starting from base_block_index to find the target address
    int remaining_offset = offset_bytes;
    int tmp = -1;
    for (int i = base_block_index; i < block_indices.size(); ++i) {
        int block_idx = block_indices[i];
        uintptr_t block_base = sram_start_address_ + block_idx * block_size_;
        int block_remaining = 0;
        if (tmp == -1){
            block_remaining = block_size_ - (current_address - block_base);
        }else{
            block_remaining = block_size_;
            current_address = block_base;

        }
        // std::cout << "block_base: " << block_base << "block_remaining " << block_remaining << std::endl;
        // std::cout << "remaining_offset: " << remaining_offset << std::endl;
        if (remaining_offset < block_remaining) {
            return static_cast<int>(block_base + (current_address - block_base) + remaining_offset);
        }else if (remaining_offset == block_remaining) {

            if (i + 1 >= block_indices.size()) {
                assert(false);
            }

            block_idx = block_indices[i + 1];
            block_base = sram_start_address_ + block_idx * block_size_;
            return static_cast<int>(block_base);
        } 
        
        else {
            remaining_offset -= block_remaining;
            tmp =0;
        }
    }
    std::cout << "remaining_offset: " << remaining_offset << std::endl;

    
    // Step 3: Offset exceeds allocated memory
    std::cerr << "Offset exceeds allocation size!" << std::endl;
    assert(false);
    return 0;
}

// Display status
void SramManager::display_status() const {
    std::cout << "SRAM Status (Start: 0x" << std::hex << sram_start_address_
              << ", Total Size: " << std::dec << total_sram_size_
              << ", Block Size: " << block_size_
              << ", Num Blocks: " << num_blocks_ << ")\n";

    if (num_blocks_ == 0) {
        std::cout << "No blocks to display.\n";
    } else {
        for (int i = 0; i < num_blocks_; ++i) {
            std::cout << "[" << (block_status_[i] ? "U" : "F") << "]"; // U for Used, F for Free
            if ((i + 1) % 32 == 0 && i + 1 < num_blocks_) std::cout << "\n";
        }
        std::cout << "\n";
    }


    std::cout << "Allocations (" << allocations_.size() << " active):\n";
    for (const auto& pair : allocations_) {
        int addr = get_address(pair.first);
        std::cout << "  ID " << pair.first << " (Addr: 0x" << std::hex << addr << std::dec << "): Blocks ";
        for (int i = 0; i < pair.second.size(); ++i) {
            std::cout << pair.second[i] << (i == pair.second.size() - 1 ? "" : ", ");
        }
        std::cout << "\n";
    }

    std::cout << "Free Block Indices (" << free_block_indices_.size() << " blocks): ";
    // Limiting output if too many free blocks for brevity
    int count = 0;
    for (int idx : free_block_indices_) {
        std::cout << idx << " ";
        count++;
        if (count > 50 && free_block_indices_.size() > 60) { // Show first 50 then indicate more
            std::cout << "... and " << (free_block_indices_.size() - count) << " more";
            break;
        }
    }
    std::cout << "\n----------------------------------\n";
}

int SramManager::get_free_blocks_count() const {
    return free_block_indices_.size();
}

int SramManager::get_used_blocks_count() const {
    if (num_blocks_ == 0) return 0;
    return num_blocks_ - free_block_indices_.size();
}