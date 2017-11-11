// Copyright (c) 2015 The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

// bitcoin/src/consensus/merkle.h

#ifndef BUBBLEFS_UTILS_BITCOIN_MERKLE_H_
#define BUBBLEFS_UTILS_BITCOIN_MERKLE_H_

#include <stdint.h>
#include <vector>
#include "utils/bitcoin_uint256.h"

namespace bubblefs {
namespace mybitcoin {
  
uint256 ComputeMerkleRoot(const std::vector<uint256>& leaves, bool* mutated = nullptr);
std::vector<uint256> ComputeMerkleBranch(const std::vector<uint256>& leaves, uint32_t position);
uint256 ComputeMerkleRootFromBranch(const uint256& leaf, const std::vector<uint256>& branch, uint32_t position);

/*
 * Compute the Merkle root of the transactions in a block.
 * *mutated is set to true if a duplicated subtree was found.
 */
//uint256 BlockMerkleRoot(const CBlock& block, bool* mutated = nullptr);

/*
 * Compute the Merkle root of the witness transactions in a block.
 * *mutated is set to true if a duplicated subtree was found.
 */
//uint256 BlockWitnessMerkleRoot(const CBlock& block, bool* mutated = nullptr);

/*
 * Compute the Merkle branch for the tree of transactions in a block, for a
 * given position.
 * This can be verified using ComputeMerkleRootFromBranch.
 */
//std::vector<uint256> BlockMerkleBranch(const CBlock& block, uint32_t position);

} // namespace mybitcoin
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_BITCOIN_MERKLE_H_